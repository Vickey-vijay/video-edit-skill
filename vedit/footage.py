"""Checks on hand-held phone footage: the things a viewer sees as "the camera is shaking".

* whip-pans and swings (the camera is thrown from one place to another)
* motion blur and focus hunting (sharpness collapses for a few frames)
* exposure jumps (auto-exposure catching up after a swing, a hand over the lens)
* lost tracking (the picture changed so much that nothing can be matched)

and the camera path that stabilisation needs. Everything is measured on the conformed clip
(constant frame rate), one pass, cached in the work folder.

Frames judged bad are cut when no speech lives in them, or held (the last good frame, slowly
pushed in) when he is talking over them, and flagged for review."""
import cv2
import numpy as np

from .util import log

SCALE = 0.5          # tracking is done at half size: fast, and sub-pixel accurate enough
VERSION = 2


# ------------------------------------------------------------------ measuring

def analyze(conform, info, wd, force=False):
    """Per conformed frame: similarity motion between this frame and the one before
    (tx, ty in source pixels, rotation in radians, log-scale), how many tracked points
    agreed (inl), sharpness, mean brightness. Cached as footage.npz."""
    out = wd / "footage.npz"
    key = f"{conform.stat().st_size}|{VERSION}"
    if out.exists() and not force:
        z = np.load(out)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files if k != "key"}
    cap = cv2.VideoCapture(str(conform))
    w, h = info["width"], info["height"]
    sw, sh = int(w * SCALE), int(h * SCALE)
    c = np.array([sw / 2.0, sh / 2.0], np.float32)
    rows = []
    prev = None
    log(f"footage check: tracking the camera over the whole clip ({sw}x{sh}) ...")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        tx = ty = rot = ls = 0.0
        inl, nfeat = 1.0, 0
        if prev is not None:
            p0 = cv2.goodFeaturesToTrack(prev, 250, 0.01, 7, blockSize=7)
            nfeat = 0 if p0 is None else len(p0)
            inl = 0.0
            if nfeat >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                a, b = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
                if len(a) >= 8:
                    M, mask = cv2.estimateAffinePartial2D(a - c, b - c, method=cv2.RANSAC,
                                                          ransacReprojThreshold=1.2, maxIters=300)
                    if M is not None:
                        s = float(np.hypot(M[0, 0], M[1, 0]))
                        tx, ty = float(M[0, 2]) / SCALE, float(M[1, 2]) / SCALE
                        rot, ls = float(np.arctan2(M[1, 0], M[0, 0])), float(np.log(max(s, 1e-3)))
                        inl = float(mask.sum()) / max(len(a), 1)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        rows.append((tx, ty, rot, ls, inl, nfeat, float(lap.var()), float(g.mean())))
        prev = g
    a = np.array(rows, np.float32)
    m = {"tx": a[:, 0], "ty": a[:, 1], "rot": a[:, 2], "ls": a[:, 3], "inl": a[:, 4],
         "nfeat": a[:, 5], "sharp": a[:, 6], "luma": a[:, 7]}
    np.savez(out, key=key, **m)
    log(f"footage check: {len(a)} frames measured")
    return m


def _rolling_median(x, k):
    pad = np.pad(x, (k, k), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    return np.median(win, axis=1)


def _min_run(flag, k):
    """Keep only stretches of True that last at least k frames (one soft frame is not a problem)."""
    out = np.zeros_like(flag)
    i, n = 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if j - i >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------ bad frames

def bad_frames(m, fps, c=None):
    """Boolean per frame (before padding) with the reason code: 1 whip, 2 blur, 3 exposure, 4 lost."""
    c = c or {}
    n = len(m["tx"])
    mag = np.hypot(m["tx"], m["ty"])
    med = _rolling_median(m["sharp"], int(fps))
    ratio = m["sharp"] / np.maximum(med, 1e-3)
    # a slow, deliberate pan is up to ~10 px a frame (at 480 px wide) and stays sharp; a whip is
    # far faster, or fast and smeared
    whip = (mag > float(c.get("whip_px", 14.0))) | ((mag > 5.0) & (ratio < 0.45))
    lost = (m["inl"] < 0.30) & (m["nfeat"] >= 20)
    blur = _min_run(ratio < float(c.get("blur_ratio", 0.55)), int(c.get("blur_min_frames", 4)))
    dl = np.abs(np.diff(m["luma"], prepend=m["luma"][0]))
    jump = (dl > float(c.get("luma_jump", 7.0))) & (mag > 0.8)
    why = np.zeros(n, np.int8)
    for code, flag in ((4, lost), (3, jump), (2, blur), (1, whip)):
        why[flag] = code
    return why


def bad_ranges(m, fps, c=None):
    """Padded, merged [f0, f1) frame ranges that a viewer would see as a swing, blur or jolt."""
    c = c or {}
    why = bad_frames(m, fps, c)
    pad = int(c.get("pad_frames", 3))
    gap = int(c.get("merge_frames", 6))
    idx = np.flatnonzero(why > 0)
    out = []
    for i in idx:
        a, b = max(0, i - pad), min(len(why), i + pad + 1)
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] = min(out[-1][2], int(why[i])) if out[-1][2] else int(why[i])
        else:
            out.append([a, b, int(why[i])])
    return [(int(a), int(b), {1: "whip", 2: "blur", 3: "exposure", 4: "lost"}[int(r)]) for a, b, r in out]


def add_interludes(ranges, talk, fps, c=None):
    """A stretch with nobody in frame that is short and sits between two talking shots is not a
    deliberate cutaway (a screen he is explaining is long): it is the camera being moved or put
    down. Treat the whole stretch, with the swings around it, as one bad range ("interlude")."""
    c = c or {}
    maxlen = int(float(c.get("interlude_max", 6.0)) * fps)
    n = len(talk)
    out = [list(r) for r in ranges]
    i = 0
    while i < n:
        if talk[i]:
            i += 1
            continue
        j = i
        while j < n and not talk[j]:
            j += 1
        if i > 0 and j < n and j - i <= maxlen:
            lo, hi = i, j
            keep = []
            for r in out:
                if r[0] <= hi + 25 and r[1] >= lo - 25:
                    lo, hi = min(lo, r[0]), max(hi, r[1])
                else:
                    keep.append(r)
            out = keep + [[lo, hi, "interlude"]]
        i = j
    return [(int(a), int(b), why) for a, b, why in sorted(out)]


# ------------------------------------------------------------------ stabilising

def _gauss(x, sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="reflect", reflect_type="odd"), k, mode="valid")


def corrections(m, foot, bad, width, c=None):
    """Per frame (dx, dy, rot, dlogscale) that steadies the camera on footage shots.

    Within each uninterrupted stretch of footage (no swings), the camera path is the running
    sum of the frame-to-frame motion; the correction is the smoothed path minus the real one,
    which removes shake and keeps slow, deliberate movement. Talk shots are left alone (the
    camera is steady there and the subject's own movement must not be fought)."""
    c = c or {}
    n = len(m["tx"])
    sigma = float(c.get("stab_sigma", 15))
    lim_px = float(c.get("stab_limit", 0.03)) * width
    lim_rot = np.radians(float(c.get("stab_rot_limit", 2.5)))
    corr = np.zeros((n, 4), np.float32)
    blocked = bad.copy()
    for a, b, _ in bad_ranges(m, 25.0, c):
        blocked[a:b] = True
    ok = foot & ~blocked
    runs, i = [], 0
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= 12:
                runs.append((i, j))
            i = j
        else:
            i += 1
    for a, b in runs:
        steps = np.stack([m["tx"][a:b], m["ty"][a:b], m["rot"][a:b], m["ls"][a:b]], axis=1).copy()
        steps[0] = 0
        path = np.cumsum(steps, axis=0)
        for k in range(4):
            smooth = _gauss(path[:, k], sigma)
            corr[a:b, k] = smooth - path[:, k]
    corr[:, 0] = np.clip(corr[:, 0], -lim_px, lim_px)
    corr[:, 1] = np.clip(corr[:, 1], -lim_px, lim_px)
    corr[:, 2] = np.clip(corr[:, 2], -lim_rot, lim_rot)
    corr[:, 3] = np.clip(corr[:, 3], -0.04, 0.04)
    stab = np.zeros(n, bool)
    for a, b in runs:
        stab[a:b] = True
    return corr, stab


def jitter(m, mask, sigma=15):
    """rms deviation (source pixels) of the camera path from its smooth version over `mask`:
    the number to compare before and after stabilising."""
    n = len(m["tx"])
    mag = np.hypot(m["tx"], m["ty"])
    idx = np.flatnonzero(mask & (mag < 8))
    if len(idx) < 25:
        return 0.0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    tot, cnt = 0.0, 0
    for r in runs:
        if len(r) < 25:
            continue
        for k in ("tx", "ty"):
            p = np.cumsum(m[k][r])
            d = p - _gauss(p, sigma)
            tot += float((d ** 2).sum())
            cnt += len(d)
    return float(np.sqrt(tot / max(cnt, 1) * 2)) if cnt else 0.0
