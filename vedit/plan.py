"""Turn a word-timestamped transcript into an edit plan:
what to keep, what to cut and why, captions, zooms and B-roll."""
import math
import re
import unicodedata
import wave
from bisect import bisect_right

import numpy as np

from .util import log

HOP = 0.01  # envelope resolution (s)
TAMIL = re.compile(r"[஀-௿]")
FILLER_SHAPES = re.compile(r"(u+h+m*|u+m+|h+m+|m{2,}|a+h+|e+h+|e+r+m*|a{2,})")


def norm(text):
    t = unicodedata.normalize("NFC", str(text)).lower().strip()
    return "".join(ch for ch in t if ch == "%" or unicodedata.category(ch)[0] not in "PSZ")


def is_junk(text):
    """Whisper sometimes drops Greek / Cyrillic / Vietnamese words into Tamil speech.
    Those are recognition noise: keep the audio, leave them out of captions."""
    for ch in text:
        o = ord(ch)
        if unicodedata.category(ch)[0] == "L" and not (o < 0x0250 or 0x0B80 <= o <= 0x0BFF):
            return True
    return False


# ---------------------------------------------------------------- audio

def envelope(wav16k):
    with wave.open(str(wav16k)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    hop = int(sr * HOP)
    m = len(x) // hop
    rms = np.sqrt((x[:m * hop].reshape(m, hop) ** 2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    return np.convolve(db, np.ones(3) / 3, mode="same")


def _span(env, t0, t1):
    return env[max(0, int(t0 / HOP)):max(int(t0 / HOP) + 1, int(t1 / HOP))]


# ---------------------------------------------------------------- removals

def mark_removals(words, c, fill):
    # low-confidence words are guesses: never cut on them
    toks = ["" if w.get("lowconf") else norm(w["text"]) for w in words]
    n = len(words)
    hard = {norm(x) for x in fill.get("hard", [])}
    soft = [tuple(norm(p) for p in str(x).split()) for x in fill.get("soft", [])]
    whitelist = {norm(x) for x in fill.get("repeat_whitelist", [])}
    stop = {norm(x) for x in fill.get("stutter_stoplist", [])}
    removed = {}

    def gap(i, j):
        return words[j]["start"] - words[i]["end"]

    if c.get("fillers"):
        for i, t in enumerate(toks):
            if t and (t in hard or (not TAMIL.search(t) and FILLER_SHAPES.fullmatch(t))):
                removed[i] = "filler"

    mode = str(c.get("soft_fillers", "off")).lower()
    if mode in ("isolated", "all"):
        for i in range(n):
            for phrase in soft:
                k = len(phrase)
                if tuple(toks[i:i + k]) != phrase:
                    continue
                before = gap(i - 1, i) if i > 0 else 9.0
                after = gap(i + k - 1, i + k) if i + k < n else 9.0
                if mode == "all" or (before >= 0.25 and after >= 0.25):
                    for j in range(i, i + k):
                        removed.setdefault(j, "soft filler")

    def kept():
        return [i for i in range(n) if i not in removed]

    if c.get("repeats"):
        idx = kept()
        kt = [toks[i] for i in idx]
        for p in range(len(idx)):
            for k in range(c.get("max_repeat_ngram", 3), 0, -1):
                if p + 2 * k > len(idx):
                    continue
                a, b = kt[p:p + k], kt[p + k:p + 2 * k]
                if a != b or not all(a):
                    continue
                if k == 1 and a[0] in whitelist:
                    continue
                if gap(idx[p + k - 1], idx[p + k]) > c.get("repeat_max_gap", 1.0):
                    continue
                for j in idx[p:p + k]:
                    removed.setdefault(j, "repeat")
                break

    if c.get("stutters"):
        idx = kept()
        for a, b in zip(idx, idx[1:]):
            ta, tb = toks[a], toks[b]
            frag = words[a]["text"].rstrip().endswith(("-", "–", "—")) and gap(a, b) < 0.8
            prefix = (ta and tb and ta != tb and tb.startswith(ta) and ta not in stop
                      and len(ta) <= max(3, 0.6 * len(tb)) and gap(a, b) <= 0.6)
            if frag or prefix:
                removed.setdefault(a, "stutter")
    return removed


# ---------------------------------------------------------------- keep segments

def build_segments(words, removed, env, duration, c, levels):
    floor = levels["floor"]
    kept = [i for i in range(len(words)) if i not in removed]
    if not kept:
        return [[0.0, duration]], []

    def extend_end(t, limit):
        i = int(t / HOP)
        cap = min(limit, t + 0.18)
        while i < len(env) and (i + 1) * HOP < cap and env[i] > floor:
            i += 1
        return max(t, min(i * HOP, cap))

    def extend_start(t, limit):
        i = int(t / HOP)
        cap = max(limit, t - 0.12)
        while i > 0 and (i - 1) * HOP > cap and env[i - 1] > floor:
            i -= 1
        return min(t, max(i * HOP, cap))

    def voiced(t0, t1):
        return float((_span(env, t0, t1) > floor + 6).sum()) * HOP

    segs, kept_noise = [], []
    first = words[kept[0]]
    start = first["start"] - c["pad_before"]
    before = [j for j in range(kept[0]) if j in removed]
    if before:
        start = max(start, words[before[-1]]["end"])
    if not c.get("trim_edges", True):
        start = 0.0
    segs.append([max(0.0, start), first["end"] + c["pad_after"]])

    for p, i in zip(kept, kept[1:]):
        pw, w = words[p], words[i]
        between = [j for j in range(p + 1, i) if j in removed]
        g = w["start"] - pw["end"]
        if not between and g <= c["max_pause"]:
            segs[-1][1] = max(segs[-1][1], w["end"] + c["pad_after"])
            continue
        if not between and voiced(pw["end"] + 0.1, w["start"] - 0.1) >= c["keep_untranscribed_speech"]:
            kept_noise.append([pw["end"], w["start"]])
            segs[-1][1] = max(segs[-1][1], w["end"] + c["pad_after"])
            continue
        end_prev = pw["end"] + c["pad_after"]
        start_next = w["start"] - c["pad_before"]
        if between:
            end_prev = max(pw["end"], min(end_prev, words[between[0]]["start"]))
            start_next = min(w["start"], max(start_next, words[between[-1]]["end"]))
            limit_e, limit_s = words[between[0]]["start"], words[between[-1]]["end"]
        else:
            limit_e, limit_s = start_next, end_prev
        end_prev = extend_end(end_prev, limit_e)
        start_next = extend_start(start_next, limit_s)
        segs[-1][1] = max(segs[-1][1], end_prev)
        if start_next - segs[-1][1] < c["min_cut"]:
            segs[-1][1] = max(segs[-1][1], w["end"] + c["pad_after"])
        else:
            segs.append([start_next, w["end"] + c["pad_after"]])

    last = words[kept[-1]]
    segs[-1][1] = extend_end(segs[-1][1], duration)
    if not c.get("trim_edges", True):
        segs[-1][1] = duration
    segs = [[max(0.0, s), min(duration, e)] for s, e in segs]
    return segs, kept_noise


def snap(segs, fps, duration):
    """Snap to whole frames and merge overlaps. Returns [[start_frame, end_frame), ...]."""
    total = int(math.floor(duration * fps))
    out = []
    for s, e in segs:
        a, b = int(math.floor(s * fps)), min(total, int(math.ceil(e * fps)))
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Timeline:
    """Maps source time <-> edited time for a list of kept [start, end) frame ranges."""

    def __init__(self, frames, fps):
        self.fps = fps
        self.src = [(a / fps, b / fps) for a, b in frames]
        self.off = []
        acc = 0.0
        for s, e in self.src:
            self.off.append(acc)
            acc += e - s
        self.duration = acc
        self.starts = [s for s, _ in self.src]

    def to_new(self, t):
        k = max(0, bisect_right(self.starts, t) - 1)
        s, e = self.src[k]
        return self.off[k] + min(max(t, s), e) - s

    def to_src(self, nt):
        k = max(0, bisect_right(self.off, nt) - 1)
        return self.src[k][0] + (nt - self.off[k])

    def inside(self, t):
        k = bisect_right(self.starts, t) - 1
        return k >= 0 and t < self.src[k][1]


# ---------------------------------------------------------------- captions & emphasis

def word_levels(words, env):
    lv = []
    for w in words:
        s = _span(env, w["start"], w["end"])
        lv.append(float(np.percentile(s, 75)) if len(s) else -90.0)
    return lv


def visible_len(text):
    """Characters as they appear on screen: Tamil vowel signs and viramas
    combine with the letter before them, so they don't add width."""
    return sum(1 for ch in text if unicodedata.category(ch) not in ("Mn", "Mc"))


def merge_numbers(words):
    """Whisper splits "0.618" into "0" + ".618" and "61.8%" into "61" + ".8" + "%"."""
    out = []
    for w in words:
        t = w["text"]
        if out and t[:1] in (".", "%") and out[-1]["text"][-1:].isdigit() and not out[-1].get("removed"):
            prev = out[-1]
            prev["text"] += t
            prev["end"] = w["end"]
            prev["prob"] = min(prev.get("prob", 1), w.get("prob", 1))
            continue
        out.append(w)
    return out


def apply_corrections(words, table):
    if not table:
        return
    fixes = {norm(k): str(v) for k, v in table.items()}
    for w in words:
        key = norm(w["text"])
        if key in fixes:
            tail = "".join(ch for ch in w["text"][::-1] if unicodedata.category(ch)[0] == "P")[::-1]
            w.setdefault("heard", w["text"])
            w["text"] = fixes[key] + (tail if not fixes[key].endswith(tail) else "")


def caption_groups(nwords, c):
    groups, cur = [], []
    min_dur = c.get("min_duration", 0.7)
    for w in nwords:
        if cur:
            chars = sum(visible_len(x["text"]) + 1 for x in cur) + visible_len(w["text"])
            if (len(cur) >= c["max_words"] or chars > c["max_chars"]
                    or w["ns"] - cur[-1]["ne"] > c["max_gap"]
                    or cur[-1]["text"].endswith((".", "?", "!"))):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    # fold captions too short to read into the one before
    merged = []
    for g in groups:
        if merged:
            prev = merged[-1]
            short = g[-1]["ne"] - g[0]["ns"] < min_dur or prev[-1]["ne"] - prev[0]["ns"] < min_dur
            chars = sum(visible_len(x["text"]) + 1 for x in prev + g)
            if short and chars <= c["max_chars"] * 1.5 and g[0]["ns"] - prev[-1]["ne"] < c["max_gap"]:
                prev.extend(g)
                continue
        merged.append(list(g))
    groups = merged
    out = []
    for gi, g in enumerate(groups):
        end = g[-1]["ne"]
        if gi + 1 < len(groups):
            nxt = groups[gi + 1][0]["ns"]
            end = nxt if nxt - end < 0.35 else end + 0.25
        out.append({"start": round(g[0]["ns"], 3), "end": round(end, 3),
                    "words": [w["i"] for w in g]})
    return out


# ---------------------------------------------------------------- zoom

def plan_zooms(nwords, groups, cuts_new, total, z, motion, timeline, base_focus, full_frame, segments):
    events = []
    if not z.get("enabled"):
        return events

    def free(s, e, margin=0.4):
        return all(e + margin <= ev["start"] or s >= ev["end"] + margin
                   for ev in events if ev["type"] != "hold")

    def focus(nt0, nt1):
        if z.get("focus") == "auto" and motion is not None:
            f = motion.focus(timeline.to_src(nt0), timeline.to_src(nt1))
            if f:
                return f
        return base_focus

    if z.get("zoom_out_reveal"):
        reveal_at = [0.0] + [cn for cn, gap in cuts_new if gap >= z["topic_break_gap"]]
        for t in reveal_at:
            s, e = t, t + z["reveal_duration"]
            if e < total and free(s, e, 1.0):
                fx, fy = focus(s, s + 2.0)
                events.append({"type": "reveal", "start": round(s, 3), "end": round(e, 3),
                               "level": z["reveal_from"], "fx": round(fx, 3), "fy": round(fy, 3)})

    group_end = {}
    for g in groups:
        for wi in g["words"]:
            group_end[wi] = g["end"]
    cands = [w for w in nwords if w.get("kw") or w.get("emph")]
    cands.sort(key=lambda w: -(w.get("emph_db", 0) + (3 if w.get("kw") else 0)))
    per_min = {}
    for w in cands:
        s = w["ns"] - z["ease_in"] * 0.8
        hold_end = min(max(group_end.get(w["i"], w["ne"]), w["ns"] + z["hold_min"]), w["ns"] + z["hold_max"])
        e = hold_end + z["ease_out"]
        if s < 0.5 or e > total - 0.3:
            continue
        minute = int(w["ns"] // 60)
        if per_min.get(minute, 0) >= z["max_per_minute"]:
            continue
        if not free(s, e, z["min_interval"] / 2):
            continue
        fx, fy = focus(w["ns"] - 1.0, w["ns"] + 1.5)
        events.append({"type": "punch", "start": round(s, 3), "end": round(e, 3),
                       "level": z["punch_in"], "ease_in": z["ease_in"], "ease_out": z["ease_out"],
                       "fx": round(fx, 3), "fy": round(fy, 3), "word": w["text"]})
        per_min[minute] = per_min.get(minute, 0) + 1

    on_cuts = z.get("on_cuts", "auto")
    if on_cuts is True or (on_cuts == "auto" and full_frame):
        for k, (s, e) in enumerate(segments):
            if k % 2 == 1:
                events.append({"type": "hold", "start": round(s, 3), "end": round(e, 3),
                               "level": z["cut_punch"], "fx": base_focus[0], "fy": base_focus[1]})
    events.sort(key=lambda ev: ev["start"])
    return events


# ---------------------------------------------------------------- b-roll

def plan_broll(nwords, library, b, total):
    if not b.get("enabled") or not library:
        return []
    keys = []
    for item in library:
        for k in item.get("keywords", []):
            keys.append((tuple(norm(p) for p in str(k).split()), item))
    toks = [norm(w["text"]) for w in nwords]
    events, last_by_id, per_min = [], {}, {}
    last = -1e9
    for i, w in enumerate(nwords):
        for phrase, item in keys:
            if tuple(toks[i:i + len(phrase)]) != phrase:
                continue
            s = max(0.0, w["ns"] - b["lead"])
            e = min(total, s + b["duration"])
            minute = int(s // 60)
            if (s - last < b["min_interval"] or per_min.get(minute, 0) >= b["max_per_minute"]
                    or s - last_by_id.get(item["id"], -1e9) < 45 or e - s < 1.5):
                continue
            events.append({"start": round(s, 3), "end": round(e, 3), "id": item["id"],
                           "file": item["file"], "word": w["text"]})
            last, last_by_id[item["id"]] = s, s
            per_min[minute] = per_min.get(minute, 0) + 1
            break
    return events


# ---------------------------------------------------------------- main

def subtract(segs, ranges, min_len=0.25):
    """Remove time ranges from kept segments."""
    out = [list(s) for s in segs]
    for a, b in sorted(ranges):
        nxt = []
        for s, e in out:
            if b <= s or a >= e:
                nxt.append([s, e])
                continue
            if a - s >= min_len:
                nxt.append([s, a])
            if e - b >= min_len:
                nxt.append([b, e])
        out = nxt
    return out


def snap_to_phrases(segs, ph, words, removed):
    """A kept piece that starts or ends inside a sound burst should take the whole burst,
    unless a removed word sits in the part being added (a stutter cut inside a phrase)."""
    def removed_between(a, b):
        return any(i in removed and a - 0.02 <= (w["start"] + w["end"]) / 2 <= b + 0.02
                   for i, w in enumerate(words))

    out = []
    for s, e in segs:
        for p in ph.items:
            if p["on"] + 0.05 < s < p["off"] and s - p["on"] <= 0.8 and not removed_between(p["on"] - 0.05, s):
                s = p["on"] - 0.06
            if p["on"] < e < p["off"] - 0.05 and p["off"] - e <= 0.5 and not removed_between(e, p["off"]):
                e = p["off"] + 0.10
        out.append([max(0.0, s), e])
    return out


def make_cues(ph, tl, words, removed):
    """The speech that survives the edit, phrase by phrase, in edited time. These are the
    units that get an English caption: id, when it is on screen, and what he said (Tamil)."""
    cues = []
    for p in ph.items:
        for s0, e0 in tl.src:
            a, b = max(p["on"], s0), min(p["off"], e0)
            if b - a < 0.25:
                continue
            txt = " ".join(w["text"] for i, w in enumerate(words)
                           if i not in removed and a - 0.05 <= (w["start"] + w["end"]) / 2 <= b + 0.05)
            cues.append({"id": len(cues) + 1, "t0": round(tl.to_new(a), 3), "t1": round(tl.to_new(b), 3),
                         "src0": round(a, 3), "src1": round(b, 3), "tamil": txt})
    return cues


def make_plan(transcript, wav16k, info, cfg, layout, motion, library, title):
    c, fps = cfg["cuts"], info.get("fps_out", 30)
    words = merge_numbers([dict(w) for w in transcript["words"]])
    apply_corrections(words, cfg.get("corrections"))
    env = envelope(wav16k)
    duration = min(info["duration"], len(env) * HOP)
    in_words = np.concatenate([_span(env, w["start"], w["end"]) for w in words]) if words else env
    speech = float(np.median(in_words))
    noise = float(np.percentile(env, 10))
    levels = {"speech": speech, "noise": noise, "floor": max(noise + 6, speech - 26)}
    log(f"levels: speech {speech:.1f} dB, noise {noise:.1f} dB, floor {levels['floor']:.1f} dB")

    from .acoustic import Phrases, load_wav
    from . import restarts as R
    ph = Phrases(load_wav(wav16k), env, levels["floor"])
    R.fix_word_times(words, ph)
    removed = mark_removals(words, c, cfg["fillers"])
    decisions = cfg.get("decisions") or {}
    cands = R.find_restarts(words, removed, ph, cfg)
    R.decide(cands, c.get("restart", {}).get("policy", "normal"), decisions.get("restarts"), duration)
    ranges = R.cut_ranges(cands, ph, c["pad_before"], c["pad_after"], decisions) + R.manual_ranges(decisions, env)
    # long pauses, found from the audio itself (Whisper's timestamps drift): leave a natural breath
    keep = c.get("keep_pause", 0.30)
    pauses = [(a + keep / 2, b - keep / 2) for a, b in ph.silences if b - a > c.get("pause_threshold", 0.55)]
    for i, w in enumerate(words):
        mid = (w["start"] + w["end"]) / 2
        if any(a <= mid <= b for a, b in ranges):
            removed.setdefault(i, "restart")
    raw, kept_noise = build_segments(words, removed, env, duration, c, levels)
    raw = snap_to_phrases(subtract(raw, ranges), ph, words, removed)
    raw = subtract(raw, ranges + pauses)
    frames = snap(raw, fps, duration)
    tl = Timeline(frames, fps)

    # cuts between kept pieces, with the reasons for each
    cuts = []
    prev_end = 0.0
    for (s, e) in tl.src + [(duration, duration)]:
        if s - prev_end > 0.01:
            inside = [i for i, w in enumerate(words) if w["start"] >= prev_end - 0.05 and w["end"] <= s + 0.05]
            reasons = sorted({removed[i] for i in inside if i in removed}) or ["pause"]
            cuts.append({"start": round(prev_end, 3), "end": round(s, 3), "dur": round(s - prev_end, 3),
                         "reasons": reasons, "text": " ".join(words[i]["text"] for i in inside)})
        prev_end = e

    kw = {norm(k) for k in cfg.get("keywords", [])}
    levels_w = word_levels(words, env)
    cap = cfg["captions"]
    nwords = []
    for i, w in enumerate(words):
        w.pop("removed", None)
        if i in removed:
            w["removed"] = removed[i]
            continue
        if not tl.inside((w["start"] + w["end"]) / 2):
            w["removed"] = "outside kept audio"
            continue
        if is_junk(w["text"]) or w.get("lowconf"):
            w["junk"] = True
            continue
        ns, ne = tl.to_new(w["start"]), tl.to_new(w["end"])
        t = norm(w["text"])
        lo, hi = max(0, i - 20), min(len(words), i + 20)
        local = float(np.median(levels_w[lo:hi]))
        emph_db = levels_w[i] - local
        w.update({"i": i, "ns": round(ns, 3), "ne": round(max(ne, ns + 0.06), 3),
                  "kw": t in kw or (cap["highlight_numbers"] and any(ch.isdigit() for ch in t)),
                  "emph": bool(cap["highlight_emphasis"] and emph_db >= cap["emphasis_db"] and len(t) > 2),
                  "emph_db": round(emph_db, 1)})
        nwords.append(w)

    groups = caption_groups(nwords, cap)
    cuts_new = [(tl.to_new(cu["end"]), cu["dur"]) for cu in cuts if cu["end"] < duration]
    full_frame = layout["mode"] == "full"
    base_focus = (layout.get("face_focus") or [0.5, 0.45]) if full_frame else [0.5, 0.5]
    seg_new = [(tl.off[k], tl.off[k] + (e - s)) for k, (s, e) in enumerate(tl.src)]
    zooms = plan_zooms(nwords, groups, cuts_new, tl.duration, cfg["zoom"], motion, tl,
                       base_focus, full_frame, seg_new)
    broll = plan_broll(nwords, library, cfg["broll"], tl.duration)

    reasons = {}
    for r in removed.values():
        reasons[r] = reasons.get(r, 0) + 1
    stats = {"source_duration": round(duration, 2), "edited_duration": round(tl.duration, 2),
             "removed_seconds": round(duration - tl.duration, 2),
             "removed_pct": round(100 * (duration - tl.duration) / max(duration, 1e-6), 1),
             "cuts": len(cuts), "removed_words": reasons, "zooms": len(zooms), "broll": len(broll),
             "captions": len(groups), "kept_untranscribed": len(kept_noise)}
    restarts = [{k: r[k] for k in ("id", "level", "decision", "score", "cov", "cost", "truncated", "unique",
                                    "t0", "t1", "b0", "removed_text", "kept_text") if k in r} for r in cands]
    phrases = [{"i": n, "on": p["on"], "off": p["off"], "text": p.get("text", "")} for n, p in enumerate(ph.items)]
    return {"version": 2, "fps": fps, "title": title, "language": transcript.get("language"),
            "restarts": restarts, "phrases": phrases, "cut_ranges": [list(r) for r in ranges],
            "cues": make_cues(ph, tl, words, removed),
            "info": info, "levels": levels, "layout": layout, "segments": frames,
            "duration": round(tl.duration, 3), "cuts": cuts, "words": words,
            "captions": groups, "zooms": zooms, "broll": broll, "stats": stats}
