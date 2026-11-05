"""Let a model that can watch AND listen (Google Gemini) judge what he is actually saying.

The rest of the pipeline finds repeats and pauses from the sound. It cannot tell that a sentence
stalled ("definitely, what happens is a ..."), that the same point is made twice, or that a stretch
is a filler or an aside, because that needs understanding Tamil. This step sends the raw clip plus
the table of speech phrases the pipeline already found (accurate start/end times from the audio)
and asks, phrase by phrase: what did he say, is it a complete thought, is it a false start, a
stall, a filler, a repeat of another phrase, an aside; keep or cut.

Cuts are decided by phrase id (the audio-detected windows), never by the model's own timestamps,
which drift by up to a second. The answer is stored in understanding.json, shown as a table for
review, and only then written into decisions.json (cut_at / keep_at / cuts), which the planner
already understands.

Two ways to run it, same result:
  * with an API key (GEMINI_API_KEY, GEMINI_API, GOOGLE_API_KEY or ~/.video-edit-skill/gemini.key) the clip is
    uploaded, judged, and the upload is deleted again;
  * without one: --prompt-only writes the prompt, you paste it into Google AI Studio with the clip
    attached, save the JSON reply and run --from-json."""
import json
import os
import re
import time
from pathlib import Path

from .util import HOME, load_json, log, save_json

KEY_FILE = HOME / "gemini.key"
KINDS = ("complete", "false_start", "stalled", "filler", "repeat", "aside", "ramble", "camera")

PROMPT = """You are a senior short-form video editor and a native Tamil speaker.

The attached video is a raw phone clip of Naga Rajan, a Tamil stock-market educator. He speaks Tamil
mixed with English finance words. He often starts a sentence, stops (sometimes mid-word) and starts it
again, uses filler phrases, and sometimes makes the same point twice. The clip will be cut down to a
{target} second Instagram Reel / YouTube Short for viewers who read English captions.

LISTEN to the audio and watch the picture. Do NOT trust the automatic text in the table below: it is
often wrong. The table only tells you where each speech phrase is (seconds on the video's own clock).

PHRASES
{table}

STEP 1 - message. In two or three English sentences: what is he actually trying to say in this clip,
and how is it structured? Do this before judging any phrase.

STEP 2 - for EVERY phrase id in the table, in order, return:
  id, tamil (what he really says, Tamil script with English words in Latin letters), english (what it
  means, natural English, by meaning: do not make it nicer or more complete than what he said),
  first_words and last_words (the first and last three words you actually hear in that window, in Latin
  letters: this proves you listened to the right place), kind, complete_thought (true/false),
  repeat_of (id of the earlier phrase making the same point, or null), keep (true/false), reason (one
  short sentence) and trim (null, or {{"drop_from_s", "drop_to_s", "drop_words"}} when only the start or
  the middle of a phrase should go, for example a stalled lead-in before a good sentence).

kind is one of:
  complete     a finished thought that moves the story forward
  false_start  a sentence he abandons and then says again (cut the abandoned attempt)
  stalled      a half-sentence that trails off or never lands, e.g. "kandippa enna pannunna oru ..."
  filler       only filler words ("athu vandhu", "ennanna", "so", "okay", "mm") or a lead-in with no content
  repeat       a point he already made; keep the better delivery, normally the last complete one, unless
               the earlier one contains information the later one lacks
  aside        something off the point of this clip
  ramble       going round in circles without adding anything
  camera       handling the phone, talking to someone off camera, or the picture is unusable

Rules for keep:
  - First priority: the argument stays complete and every sentence that remains makes sense on its own.
  - Then remove repeats, stalls, fillers, asides and rambles.
  - Then tighten toward {target} seconds. If tightening would cut a point the message needs, keep it.
  - Two kept phrases must never say the same thing. Never reorder anything: the edit only deletes.
  - If a repeated point adds information the kept one lacks, say so in reason.

STEP 3 - fillers: individual filler words or sounds INSIDE phrases you keep that should be trimmed:
a list of {{"t": seconds, "text": word}}.

STEP 4 - warnings: anything you are unsure of (unclear audio, a number you could not hear, a phrase
that might be two sentences).

Reply with JSON only, exactly this shape:
{{"message": "...", "structure": "...",
 "phrases": [{{"id": 0, "tamil": "...", "english": "...", "first_words": "...", "last_words": "...",
   "kind": "complete", "complete_thought": true, "repeat_of": null, "keep": true, "reason": "...",
   "trim": null}}],
 "fillers": [{{"t": 0.0, "text": "..."}}], "warnings": ["..."]}}
"""

REFINE = """You are a senior short-form video editor. Below is a per-phrase analysis of a Tamil trading
clip (JSON). Produce the final edit as deletions only (never reorder).

Rules: keep the argument complete first; group phrases that make the same point and keep the best take
(normally the last complete one; keep an earlier one if it holds information the later one lacks and
say which); remove stalls, fillers, asides; aim for {target} seconds of kept speech but never drop a point
the message needs.
Then SELF-CHECK: read only the kept english in order, as one script, and list every place where a kept
phrase reads as a fragment, starts mid-thought, or dangles.

Reply with JSON only: {{"keep": [ids], "cut": [ids], "repeat_groups": [{{"ids": [..], "kept": id,
"dropped_has_new_info": false}}], "dangling": [{{"id": 0, "problem": "..."}}], "notes": "..."}}

ANALYSIS
{analysis}
"""


# ------------------------------------------------------------------ key and prompt

def api_key():
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API"):    # Windows names are case-insensitive: Gemini_API works
        v = os.environ.get(name, "").strip().strip('"').strip("'")
        if v:
            return v
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return None


def phrase_table(plan):
    rows = [f"P{p['i']:02d}  {p['on']:7.2f} - {p['off']:7.2f} s   auto text (unreliable): {p.get('text', '')[:110]}"
            for p in plan["phrases"]]
    return "\n".join(rows)


def build_prompt(plan, target="60-80"):
    return PROMPT.format(table=phrase_table(plan), target=target)


# ------------------------------------------------------------------ calling the model

def _client(key):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=900_000)), types


def list_models(key):
    client, _ = _client(key)
    out = []
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            out.append(m.name)
    return out


def _pick_model(key, want):
    """The configured model if it exists, else the best-looking Pro-class one the key can use."""
    names = list_models(key)
    short = [n.split("/")[-1] for n in names]
    if want in short:
        return want
    bad = ("image", "tts", "live", "embedding", "audio", "robotics", "computer", "thinking-exp", "vision")
    pro = [n for n in short if "pro" in n and not any(b in n for b in bad)]
    flash = [n for n in short if "flash" in n and "lite" not in n and not any(b in n for b in bad)]
    pick = (sorted(pro) or sorted(flash) or short or [want])[-1]
    log(f"model {want!r} not available to this key; using {pick!r}")
    return pick


def ask(video, prompt, model, key):
    """Upload the clip, ask, delete the upload. Returns the reply text."""
    client, types = _client(key)
    model = _pick_model(key, model)
    log(f"uploading {Path(video).name} to Gemini ({model}) ...")
    f = client.files.upload(file=str(video))
    try:
        while f.state.name == "PROCESSING":
            time.sleep(3)
            f = client.files.get(name=f.name)
        if f.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini could not process the video (state {f.state.name})")
        log("asking (a 2-3 minute clip usually takes 1-3 minutes) ...")
        r = client.models.generate_content(
            model=model, contents=[f, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
        return r.text
    finally:
        try:
            client.files.delete(name=f.name)          # do not leave the client's video on Google's side
            log("upload deleted")
        except Exception as e:                        # noqa: BLE001
            log(f"could not delete the upload ({e}); it expires on its own within 48 hours")


def ask_text(prompt, model, key):
    client, types = _client(key)
    model = _pick_model(key, model)
    r = client.models.generate_content(
        model=model, contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
    return r.text


def parse(text):
    """The reply as a dict, tolerating code fences and stray text around the JSON."""
    t = (text or "").strip()
    if not t:
        raise ValueError("Gemini returned no text (the answer may have been blocked or cut off)")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            raise
        out = json.loads(t[a:b + 1])
    if isinstance(out, list):                        # some models answer with the bare list of phrases
        out = {"phrases": out}
    return out


# ------------------------------------------------------------------ checking the answer

def normalise(result, plan):
    """Phrases keyed by id as ints, kinds validated, missing phrases noted. Returns (by_id, problems)."""
    by, problems = {}, []
    for p in result.get("phrases", []):
        try:
            i = int(str(p.get("id")).lstrip("Pp"))
        except ValueError:
            problems.append(f"unreadable id {p.get('id')!r}")
            continue
        if p.get("kind") not in KINDS:
            problems.append(f"P{i:02d}: unknown kind {p.get('kind')!r}")
        by[i] = p
    known = {p["i"]: p for p in plan["phrases"]}
    for i in known:
        if i not in by:
            problems.append(f"P{i:02d}: no answer (treated as keep)")
    for i in by:
        if i not in known:
            problems.append(f"P{i:02d}: not in the table (ignored)")
    for i, p in by.items():
        if i in known:
            dur = max(known[i]["off"] - known[i]["on"], 0.1)
            words = len(str(p.get("tamil", "")).split())
            rate = words / dur
            if words and (rate < 0.7 or rate > 6.0):
                p["check"] = f"{words} words in {dur:.1f} s: the answer may be for another place in the clip"
            r = p.get("repeat_of")
            if r is not None:
                try:
                    r = int(str(r).lstrip("Pp"))
                except ValueError:
                    r = None
                # A retake may come LATER in the clip (he re-says it after an interruption): fine when that later
                # phrase is kept. Flag only a repeat of itself or of a phrase that is not kept either.
                if r is None or r == i or r not in by or not by[r].get("keep", True):
                    p["check"] = (p.get("check", "") + " repeat_of points at a phrase that is not kept").strip()
    return {i: p for i, p in by.items() if i in known}, problems


def review_text(result, plan, by, problems):
    known = {p["i"]: p for p in plan["phrases"]}
    lines = [f"MESSAGE: {result.get('message', '')}", f"STRUCTURE: {result.get('structure', '')}", "",
             " id   time          keep  kind          english  (reason)"]
    kept = 0.0
    for i in sorted(known):
        p = by.get(i)
        k = known[i]
        if p is None:
            lines.append(f"P{i:02d} {k['on']:6.1f}-{k['off']:6.1f}   ?     (no answer)")
            continue
        keep = bool(p.get("keep", True))
        kept += (k["off"] - k["on"]) if keep else 0.0
        flag = "  [CHECK: " + p["check"] + "]" if p.get("check") else ""
        trim = p.get("trim")
        tr = ""
        if trim:
            try:
                tr = f"  [trim {float(trim.get('drop_from_s')):.1f}-{float(trim.get('drop_to_s')):.1f}: {trim.get('drop_words')}]"
            except (TypeError, ValueError):
                tr = f"  [trim {trim}]"
        lines.append(f"P{i:02d} {k['on']:6.1f}-{k['off']:6.1f}   {'KEEP' if keep else 'cut '}  {str(p.get('kind', '')):12s}  "
                     f"{str(p.get('english', ''))[:90]}  ({str(p.get('reason', ''))[:80]}){tr}{flag}")
    lines += ["", f"kept speech about {kept:.0f} s of {sum(k['off'] - k['on'] for k in known.values()):.0f} s"]
    if result.get("fillers"):
        lines.append("fillers inside kept phrases: " + ", ".join(f"{f.get('t')}s {f.get('text')}" for f in result["fillers"][:20]))
    for w in result.get("warnings", []):
        lines.append(f"WARNING: {w}")
    for pr in problems:
        lines.append(f"PROBLEM: {pr}")
    return "\n".join(lines)


# ------------------------------------------------------------------ turning it into decisions

def to_decisions(by, plan):
    """cut_at / keep_at (a second inside each phrase) and hand cuts for trims inside a phrase."""
    known = {p["i"]: p for p in plan["phrases"]}
    cut_at, keep_at, cuts = [], [], []
    for i, p in sorted(by.items()):
        k = known[i]
        mid = round((k["on"] + k["off"]) / 2, 3)
        if not p.get("keep", True):
            cut_at.append(mid)
        elif p.get("kind") == "complete" and not p.get("check"):
            keep_at.append(mid)            # a finished thought: overrides the sound-based false-start guess
        t = p.get("trim")
        if p.get("keep", True) and t and t.get("drop_from_s") is not None and t.get("drop_to_s") is not None:
            a = min(max(float(t["drop_from_s"]), k["on"] - 0.5), k["off"])
            b = min(max(float(t["drop_to_s"]), a), k["off"] + 0.2)
            if b - a >= 0.15:
                cuts.append({"t0": round(a, 3), "t1": round(b, 3)})
    return {"cut_at": cut_at, "keep_at": keep_at, "cuts": cuts}


def apply(wd, plan, by):
    path = wd / "decisions.json"
    cur = load_json(path) if path.exists() else {}
    if path.exists():
        save_json(wd / "decisions.before_gemini.json", cur)
    d = to_decisions(by, plan)
    # An earlier hand-made trim INSIDE a phrase the model keeps is finer than its phrase-level judgment: keep it
    # (unless the model trims the same stretch itself). Trims inside phrases the model cuts are moot.
    known = {p["i"]: p for p in plan["phrases"]}
    for c in cur.get("cuts", []):
        mid = (c["t0"] + c["t1"]) / 2
        i = next((i for i, k in known.items() if k["on"] - 0.3 <= mid <= k["off"] + 0.3), None)
        if i is not None and by.get(i, {}).get("keep", True) and not any(c["t0"] < n["t1"] and n["t0"] < c["t1"] for n in d["cuts"]):
            d["cuts"].append(c)
            log(f"kept the earlier trim {c['t0']}-{c['t1']} s (inside P{i:02d}, which is kept)")
    # replace the earlier phrase-level decisions with the model's; keep the restart-id decisions
    cur["cut_at"], cur["keep_at"], cur["cuts"] = d["cut_at"], d["keep_at"], d["cuts"]
    save_json(path, cur)
    return cur


def english_for_cues(plan, by):
    """{cue id: english} from the model's per-phrase English, where a cue is one whole phrase."""
    known = {p["i"]: p for p in plan["phrases"]}
    out, shared = {}, {}
    for c in plan["cues"]:
        mid = (c["src0"] + c["src1"]) / 2
        i = next((i for i, k in known.items() if k["on"] - 0.1 <= mid <= k["off"] + 0.1), None)
        if i is not None and i in by:
            shared.setdefault(i, []).append(c["id"])
    for i, ids in shared.items():
        if len(ids) == 1:
            out[str(ids[0])] = str(by[i].get("english", "")).strip()
    multi = [i for i, ids in shared.items() if len(ids) > 1]
    return out, multi


def refine(by, plan, target, model, key):
    """Optional second call (text only): repeat groups, best takes and a self-check of the kept script."""
    analysis = json.dumps([{"id": i, "kind": p.get("kind"), "english": p.get("english"), "repeat_of": p.get("repeat_of"),
                            "keep": p.get("keep"), "seconds": round(next(k["off"] - k["on"] for k in plan["phrases"] if k["i"] == i), 1)}
                           for i, p in sorted(by.items())], ensure_ascii=False)
    return parse(ask_text(REFINE.format(target=target, analysis=analysis), model, key))
