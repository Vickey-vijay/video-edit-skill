"""False starts: he begins a sentence, stops (sometimes mid-word), then says it again.

The aborted attempt is what gets cut; in a chain of attempts only the last one stays.
A phrase counts as aborted when a later phrase (within a few seconds) opens with
the same words, judged two ways that back each other up:

  text   - Whisper's words, compared on a phonetic skeleton so that "டெக்னிய",
           "technical" and "ரீசன்" / "recent" still line up
  audio  - the first moments of the two phrases sound alike (see acoustic.py),
           which also works when Whisper dropped the attempt or mis-timed it

Each candidate gets a score and a level:
  high    cut automatically
  medium  listed for review (kept unless restarts.policy = aggressive)
Decisions made at review time ({"restarts": {"3": "cut"}}) override both."""
import re
import unicodedata
from difflib import SequenceMatcher

from .plan import norm

TAMIL_CONS = dict(zip("கஙசஞடணதநபமரலவழளறனஜஷஸஹ", "knsntnt" "npm" "rlvlllr" "nsssh"))
LATIN_DIGRAPHS = (("ph", "p"), ("th", "t"), ("sh", "s"), ("kh", "k"), ("gh", "k"), ("ck", "k"))
LATIN_CONS = {"c": "k", "g": "k", "d": "t", "b": "p", "z": "s", "j": "s", "q": "k", "f": "p", "w": "v"}
TAIL = re.compile(r"^([A-Za-z0-9.%]+)([஀-௿]{0,3})$")
ENDS = (".", "?", "!", "…")


def skeleton_keys(token):
    """Phonetic consonant skeletons for a word, in any script. Latin words get two
    variants because 'ch' is a 'k' in technical and an 's' in much else."""
    m = TAIL.match(token)
    if m and m.group(2):
        token = m.group(1)           # 260ல -> 260, ratioல -> ratio
    if any(c.isdigit() for c in token):
        return [token.replace("%", "")]
    out = []
    latin = bool(re.search(r"[A-Za-z]", token))
    for ch_val in (("k", "s") if latin else ("",)):
        buf = []
        s = token.lower() if latin else token
        if latin:
            s = s.replace("ch", ch_val)
            for a, b in LATIN_DIGRAPHS:
                s = s.replace(a, b)
        for c in s:
            if c in TAMIL_CONS:
                buf.append(TAMIL_CONS[c])
            elif latin and c.isalpha() and c not in "aeiouyh":
                buf.append(LATIN_CONS.get(c, c) if c != "x" else "ks")
        key = re.sub(r"(.)\1+", r"\1", "".join(buf))
        if key:
            out.append(key)
    return out or [""]


def keys_match(ka, kb):
    for a in ka:
        for b in kb:
            if not a or not b:
                continue
            if a == b:
                return True
            if a.isdigit() or b.isdigit():
                continue
            lo = min(len(a), len(b))
            if lo >= 3 and (a.startswith(b) or b.startswith(a)):
                return True
            if lo >= 3 and SequenceMatcher(None, a, b).ratio() >= 0.78:
                return True
    return False


def weight(keys, common):
    k = keys[0]
    if len(k) < 2:
        return 0.0
    if k.isdigit():
        return 0.2
    return 0.35 if k in common else 1.0


def coverage(A, B, common):
    """Weighted longest-common-subsequence of A inside the start of B, as a share of A."""
    Bp = B[:len(A) + 3]
    total = sum(weight(a, common) for a in A)
    if total < 0.8:
        return 0.0, False
    n, m = len(A), len(Bp)
    L = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            w = weight(A[i], common) if keys_match(A[i], Bp[j]) else 0.0
            L[i + 1][j + 1] = max(L[i][j + 1], L[i + 1][j], L[i][j] + w if w else 0.0)
    first = 0
    while first < min(2, n) and weight(A[first], common) == 0:
        first += 1                                   # skip a leading function word
    start_ok = first < n and any(keys_match(A[first], Bp[j]) for j in range(min(3, m)))
    return L[n][m] / total, start_ok


def fix_word_times(words, ph):
    """Whisper sometimes stretches a word across a long silence. Pull its edges back
    to where the sound actually is."""
    fixed = 0
    for w in words:
        for a, b in ph.silences:
            if b - a < 0.2:
                continue
            if w["start"] < a < w["end"] - 0.1 and b < w["end"] and a > w["start"] + 0.1 and b - a >= 0.3:
                w["end"] = round(a, 3)
                fixed += 1
            elif a - 0.02 <= w["start"] < b and b < w["end"] - 0.1:
                w["start"] = round(b, 3)
                fixed += 1
    return fixed


def phrase_words(words, skip, p):
    return [i for i, w in enumerate(words)
            if p["on"] - 0.12 <= (w["start"] + w["end"]) / 2 <= p["off"] + 0.12
            and i not in skip and not w.get("lowconf")]


def find_restarts(words, skip, ph, cfg):
    c = cfg["cuts"].get("restart", {})
    if not c.get("enabled", True):
        return []
    items = ph.items
    for p in items:
        p["words"] = phrase_words(words, skip, p)
        p["keys"] = [skeleton_keys(norm(words[i]["text"]) or words[i]["text"]) for i in p["words"]]
        p["text"] = " ".join(words[i]["text"] for i in p["words"])
    opener = {}
    for p in items:
        for k in p["keys"][:2]:
            opener[k[0]] = opener.get(k[0], 0) + 1
    common = {k for k, n in opener.items() if n >= max(4, 0.2 * len(items))}

    gap_max, win = c.get("gap_max", 4.5), c.get("window", 8.0)
    found = {}
    for i, A in enumerate(items):
        dA = A["off"] - A["on"]
        if not A["keys"] and dA > 2.5:
            continue
        best = None
        for j in range(i + 1, min(i + 5, len(items))):
            B = items[j]
            gap = B["on"] - A["off"]
            if gap > gap_max or B["on"] - A["on"] > win:
                break
            k2 = j                                  # the restart may run on across small pauses
            while k2 + 1 < len(items) and items[k2 + 1]["on"] - items[k2]["off"] < 0.6                     and items[k2 + 1]["off"] - B["on"] < 8.0:
                k2 += 1
            dB = items[k2]["off"] - B["on"]
            cov, start_ok = coverage(A["keys"], B["keys"], common) if A["keys"] and B["keys"] else (0.0, False)
            cost = ph.opening_cost(A, B)
            s = 0.0
            s += 0.0 if cov < 0.4 else (1.0 if cov < 0.6 else 2.0) + (0.5 if start_ok else -1.0)
            if cov < 0.4 and A["keys"] and B["keys"] and keys_match(A["keys"][0], B["keys"][0]) \
                    and len(A["keys"][0][0]) >= 2 and weight(A["keys"][0], common) > 0 and len(A["keys"]) <= 4:
                s += 1.0                                    # same first word, short attempt
            s += 3.0 if cost < 30 else 2.0 if cost < 33 else 1.2 if cost < 38 else 0.5 if cost < 42 else 0.0
            s += 0.6 if dA <= dB * 1.15 else 0.0
            s += 0.4 if gap >= 0.4 else 0.0
            s += 0.6 if (gap >= 1.0 and len(A["keys"]) <= 4) else 0.0
            tail = A["keys"][-1] if A["keys"] else [""]
            truncated = A["text"].rstrip().endswith(("...", "…")) or any(
                len(t) >= 3 and len(k) > len(t) and k.startswith(t)
                for t in tail for kk in B["keys"][:6] for k in kk)
            s += 1.4 if truncated else 0.0
            later = {k for q in items[j:j + 3] for kk in q["keys"] for k in kk}
            unique = [A["words"][n] for n, kk in enumerate(A["keys"])
                      if weight(kk, common) >= 1.0 and not any(k in later for k in kk)
                      and not any(keys_match(kk, [x]) for x in later)]
            ends = A["text"].rstrip().endswith(ENDS) and len(A["keys"]) >= 5 and not truncated
            if dB < 0.8 * dA and not truncated:
                continue                       # a restart is not shorter than the attempt it replaces
            if not B["keys"] and dB < 1.0:
                continue
            s -= 2.5 if (cost >= 50 and not truncated and cov < 0.9) else 1.5 if (cost >= 47 and not truncated and cov < 0.9) else 0.0
            s -= 0.9 if (len(unique) >= 2 and not truncated and cost >= 34) else 0.0
            s -= 0.7 if ends else 0.0
            if dA > c.get("max_dur", 6.0) or len(A["keys"]) > c.get("max_tokens", 15):
                s -= 2.0
            cand = {"A": i, "B": j, "score": round(s, 2), "cov": round(cov, 2), "cost": round(cost, 1),
                    "truncated": bool(truncated), "unique": len(unique), "gap": round(gap, 2),
                    "pause_ok": gap >= 0.25}
            if best is None or s > best["score"]:
                best = cand
        if best and best["score"] >= c.get("medium", 1.8):
            found[i] = best

    # a short phrase caught between an aborted attempt and its restart is part of the chain
    for i, cd in list(found.items()):
        if cd["score"] < c.get("high", 3.0):
            continue
        for x in range(i + 1, cd["B"]):
            X = items[x]
            if x in found and found[x]["B"] == cd["B"]:
                found[x]["score"] = max(found[x]["score"], c.get("high", 3.0))
                found[x]["chain"] = True
                continue
            if x in found or X["off"] - X["on"] > 2.5:
                continue
            cost = ph.opening_cost(X, items[cd["B"]])
            if cost < 45 or (X["keys"] and items[cd["B"]]["keys"]
                             and keys_match(X["keys"][0], items[cd["B"]]["keys"][0])):
                found[x] = {"A": x, "B": cd["B"], "score": round(c.get("high", 3.0), 2), "cov": 0.0,
                            "cost": round(cost, 1), "truncated": False, "unique": 0,
                            "gap": round(items[cd["B"]]["on"] - X["off"], 2), "pause_ok": True, "chain": True}

    for i, A in enumerate(items[:-1]):
        if i in found or A["off"] - A["on"] > 0.55 or len(A["keys"]) > 1:
            continue
        gap = items[i + 1]["on"] - A["off"]
        if gap >= 0.8:
            opener = bool(A["keys"]) and weight(A["keys"][0], common) == 0 or not A["keys"]
            found[i] = {"A": i, "B": i + 1, "score": 3.1 if opener else 2.0, "cov": 0.0, "cost": 99.0,
                        "truncated": False, "unique": 0, "gap": round(gap, 2), "pause_ok": True,
                        "fragment": True}
    out = []
    for i, cd in sorted(found.items()):
        A, B = items[i], items[cd["B"]]
        level = "high" if cd["score"] >= c.get("high", 3.0) else "medium"
        if level == "high" and cd["unique"] >= 2 and not cd["truncated"] and cd["cost"] >= 34:
            level = "medium"                     # the earlier attempt says something the later one does not
        if level == "high" and not cd["pause_ok"]:
            level = "medium"
        out.append({**cd, "level": level, "t0": A["on"], "t1": A["off"], "b0": B["on"],
                    "removed_text": A["text"] or "(not transcribed)", "kept_text": B["text"],
                    "words": A["words"]})
    for n, r in enumerate(out, 1):
        r["id"] = n
    return out


def phrase_at(items, t):
    """Index of the phrase that contains time t (or the nearest one within 0.4 s)."""
    best, gap = None, 0.4
    for i, p in enumerate(items):
        d = 0.0 if p["on"] <= t <= p["off"] else min(abs(t - p["on"]), abs(t - p["off"]))
        if d < gap:
            best, gap = i, d
    return best


def decide(cands, policy, decisions, total, cap=0.15):
    """Fill in each candidate's decision. Automatic cutting is capped at a share of the
    runtime, best candidates first; the rest wait for review."""
    spent, order = 0.0, sorted(cands, key=lambda r: -r["score"])
    for r in order:
        d = (decisions or {}).get(str(r["id"]))
        if d is None:
            auto = r["level"] == "high" or policy == "aggressive"
            if auto and spent + (r["t1"] - r["t0"]) > cap * total:
                auto, r["capped"] = False, True
            d = "cut" if auto else "keep"
        if d == "cut":
            spent += r["t1"] - r["t0"]
        r["decision"] = d
    return cands


def cut_ranges(cands, ph, pad_before, pad_after, decisions=None):
    """Time ranges to remove: each run of cut phrases, from just after the speech
    before it to just before the speech after it, so no stray pause is left behind.
    decisions: {"cut_at": [seconds...], "keep_at": [seconds...]} act on whole phrases."""
    cut = {r["A"] for r in cands if r["decision"] == "cut"}
    for key, add in (("cut_at", True), ("keep_at", False)):
        for tt in (decisions or {}).get(key, []):
            k = phrase_at(ph.items, float(tt))
            if k is not None:
                (cut.add if add else cut.discard)(k)
    items, out, i = ph.items, [], 0
    for _ in range(2):                       # a short sound caught between two cut phrases is part of the stumble
        for k in range(1, len(items) - 1):
            if k not in cut and k - 1 in cut and k + 1 in cut and items[k]["off"] - items[k]["on"] <= 1.2                     and items[k + 1]["on"] - items[k - 1]["off"] <= 4.0:
                cut.add(k)
    while i < len(items):
        if i in cut:
            j = i
            while j + 1 in cut:
                j += 1
            prev_off = items[i - 1]["off"] if i > 0 else 0.0
            nxt_on = items[j + 1]["on"] if j + 1 < len(items) else items[j]["off"]
            t0 = max(prev_off + pad_after, items[i]["on"] - 0.35) if i > 0 else 0.0
            t1 = nxt_on - pad_before
            if t1 > t0 + 0.05:
                out.append((round(t0, 3), round(t1, 3)))
            i = j + 1
        else:
            i += 1
    return out


def valley(env, t, radius=0.2, hop=0.01):
    """Time of the quietest moment within radius of t (smoothed), so a hand-placed cut lands in a pause."""
    import numpy as np
    a, b = max(0, int((t - radius) / hop)), min(len(env), int((t + radius) / hop) + 1)
    if b - a < 3:
        return t
    s = np.convolve(env[a:b], np.ones(5) / 5, mode="same")
    return round((a + int(np.argmin(s))) * hop, 3)


def manual_ranges(decisions, env):
    """Hand-placed cuts from review: [{"t0":..,"t1":..}] snapped into nearby pauses."""
    out = []
    for c in (decisions or {}).get("cuts", []):
        t0, t1 = valley(env, float(c["t0"])), valley(env, float(c["t1"]))
        if t1 > t0 + 0.05:
            out.append((t0, t1))
    return out
