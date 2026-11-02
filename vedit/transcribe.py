"""Speech-to-text with word timestamps (faster-whisper), with a repair pass for
the places Whisper gets Tamil-English speech wrong."""
import glob
import os
import site
import unicodedata
import wave

import numpy as np

from .util import HOME, log, run, save_json, load_json

SR = 16000
TEMPS = [0.0, 0.2, 0.4]  # higher temperatures turn Tamil into multi-script gibberish


def _add_cuda_dlls():
    # pip's nvidia-cublas / nvidia-cudnn wheels ship the DLLs ctranslate2 needs on Windows.
    for sp in site.getsitepackages():
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def extract_audio_16k(video, out_wav):
    if not out_wav.exists():
        run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", out_wav])
    return out_wav


def _load_model(cfg):
    _add_cuda_dlls()
    from faster_whisper import WhisperModel
    root = str(HOME / "models")
    try:
        return WhisperModel(cfg["model"], device=cfg["device"],
                            compute_type=cfg["compute_type"], download_root=root)
    except Exception as e:  # no GPU / driver problem
        log(f"GPU load failed ({e}); falling back to CPU int8 — this is much slower")
        return WhisperModel(cfg["model"], device="cpu", compute_type="int8", download_root=root)


def junk_frac(text):
    """Share of letters outside Latin and Tamil (Greek, Cyrillic, CJK, Vietnamese...)."""
    letters = [c for c in text if unicodedata.category(c)[0] == "L"]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if not (ord(c) < 0x0250 or 0x0B80 <= ord(c) <= 0x0BFF))
    return bad / len(letters)


def _loops(words, min_rep=4):
    """Indices of words in a repetition loop ("0.78, 0.78, 0.78, ..."), keeping the first copy."""
    toks = [w["text"].lower().strip(".,?!") for w in words]
    bad, n = set(), len(toks)
    for k in (1, 2, 3):
        i = 0
        while i + 2 * k <= n:
            j, reps = i, 1
            while j + 2 * k <= n and toks[j:j + k] == toks[j + k:j + 2 * k]:
                reps += 1
                j += k
            if reps >= min_rep:
                bad.update(range(i + k, j + k))
                i = j + k
            else:
                i += 1
    return bad


def _decode(model, samples, offset, language, prompt, **kw):
    """Transcribe samples; return segments as dicts with words and a quality score."""
    segs, _ = model.transcribe(samples, language=language, task="transcribe", beam_size=5,
                               word_timestamps=True, initial_prompt=prompt or None,
                               condition_on_previous_text=False, temperature=TEMPS, **kw)
    out = []
    for s in segs:
        words = [{"text": w.word.strip(), "start": round(w.start + offset, 3),
                  "end": round(w.end + offset, 3), "prob": round(w.probability, 3)}
                 for w in (s.words or []) if w.word.strip()]
        drop = _loops(words)
        words = [w for i, w in enumerate(words) if i not in drop]
        text = " ".join(w["text"] for w in words)
        jf = junk_frac(text)
        score = s.avg_logprob - 3.0 * jf - (0.6 if drop else 0) - (0.5 if s.compression_ratio > 2.4 else 0)
        out.append({"start": round(s.start + offset, 3), "end": round(s.end + offset, 3), "text": text,
                    "words": words, "score": round(score, 3), "junk": round(jf, 3)})
    return out


def _bad(seg):
    return seg["score"] < -0.85 or seg["junk"] > 0.08 or not seg["words"]


def transcribe(audio16k, cfg, out_json):
    """Returns {"language", "duration", "segments", "words": [{text,start,end,prob,lowconf?}]}."""
    settings = {k: cfg[k] for k in ("model", "language", "beam_size", "initial_prompt")}
    settings["repair"] = 2
    if out_json.exists():
        cached = load_json(out_json)
        if cached.get("settings") == settings:
            log(f"transcript cached: {out_json.name}")
            return cached
    model = _load_model(cfg)
    with wave.open(str(audio16k)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    lang = cfg["language"]
    if lang in (None, "auto"):
        lang = model.detect_language(samples[:SR * 60])[0]
    log(f"transcribing with {cfg['model']} (language={lang}) ...")
    prompt = cfg.get("initial_prompt")
    segs = _decode(model, samples, 0.0, lang, prompt, vad_filter=True,
                   vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200})
    for s in segs:
        log(f"  {s['start']:7.1f}s {'?' if _bad(s) else ' '} {s['text'][:90]}")
    segs, repaired = repair(model, samples, segs, lang, prompt)
    words = []
    for k, s in enumerate(segs):
        for w in s["words"]:
            w["seg"] = k
            if _bad(s):
                w["lowconf"] = True
            words.append(w)
    words.sort(key=lambda w: w["start"])
    result = {"settings": settings, "language": lang, "duration": len(samples) / SR,
              "segments": [{k: s[k] for k in ("start", "end", "text", "score", "junk")} for s in segs],
              "words": words, "repaired": repaired}
    save_json(out_json, result)
    low = sum(1 for w in words if w.get("lowconf"))
    log(f"transcript: {len(words)} words ({low} low-confidence, kept out of captions), "
        f"{len(repaired)} passages re-transcribed")
    return result


def repair(model, samples, segs, language, prompt):
    """Retry weak sentences with other settings and keep the best version; then
    transcribe speech that produced no words at all."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    total = len(samples) / SR
    repaired = []

    def window(a, b):
        a, b = max(0.0, a), min(total, b)
        return samples[int(a * SR):int(b * SR)], a

    out = []
    for s in segs:
        if not _bad(s):
            out.append(s)
            continue
        clip, a = window(s["start"] - 0.3, s["end"] + 0.3)
        cands = [[s]]
        for p in (None, prompt):
            got = _decode(model, clip, a, language, p, vad_filter=False, no_repeat_ngram_size=4)
            got = [g for g in got if s["start"] - 0.5 <= (g["start"] + g["end"]) / 2 <= s["end"] + 0.5]
            if got:
                cands.append(got)

        def quality(c):
            n = sum(len(g["words"]) for g in c) or 1
            return sum(g["score"] * len(g["words"]) for g in c) / n

        best = max(cands, key=quality)
        if best is not cands[0]:
            text = " ".join(g["text"] for g in best)
            repaired.append({"start": s["start"], "end": s["end"], "was": s["text"], "now": text})
            log(f"  retried {s['start']:6.1f}s: {text[:80]}")
        out.extend(best)

    # speech the voice detector hears but no word covers
    for w in (w for s in out for w in s["words"]):
        if w["end"] - w["start"] > 1.2:  # stretched over a gap
            w["end"] = round(w["start"] + 1.2, 3)
    covered = sorted((w["start"] - 0.3, w["end"] + 0.3) for s in out for w in s["words"])
    holes = []
    for sp in get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100)):
        s, e = sp["start"] / SR, sp["end"] / SR
        cur = s
        for a, b in covered:
            if b <= cur or a >= e:
                continue
            if a - cur >= 1.0:
                holes.append([cur, a])
            cur = max(cur, b)
        if e - cur >= 1.0:
            holes.append([cur, e])
    for s, e in holes:
        clip, a = window(s - 0.4, e + 0.4)
        got = _decode(model, clip, a, language, None, vad_filter=False, no_repeat_ngram_size=4)
        for g in got:
            g["words"] = [w for w in g["words"] if s - 0.25 <= (w["start"] + w["end"]) / 2 <= e + 0.25
                          and w["end"] - w["start"] <= 2.0]
            if g["words"]:
                out.append(g)
                repaired.append({"start": round(s, 2), "end": round(e, 2), "was": "", "now": g["text"]})
                log(f"  filled  {s:6.1f}s: {g['text'][:80]}")
    out.sort(key=lambda g: g["start"])
    return out, repaired
