"""Finishing of the speaker's cut-out so it sits in a new scene instead of looking pasted on.

Phone footage is small and soft, and when he moves his head the matte trails behind the movement
as a translucent band that shows the old room. Three things fix that at the compositing stage:

1. `firm_alpha`: the matte is made decisive (a smoothstep between lo and hi) and then feathered a
   pixel or so, so the soft ghost is gone but the edge is still anti-aliased.
2. `interior_colour`: every edge pixel takes its colour from the inside of the subject (the colour
   of his hair or cheek pushed outward), so no pixel at the border carries the old wall.
3. `skin_balance`: the phone lights one cheek blue-grey and the other warm. Skin-like pixels in
   the face zone are pulled toward a healthy brown, leaving the hoodie and hair alone.

Everything works per pixel on a single frame (no state), so frames can be rendered in any order
and in parallel pieces without flicker."""
import cv2
import numpy as np


def smooth(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def interior_colour(rgb, core, scale=3, sigmas=(2.0, 5.0, 12.0)):
    """Colour of the subject's interior pushed outward (normalised convolution at 1/scale size).
    Returns (colour, have): `have` is 1 where an interior colour was found nearby."""
    h, w = core.shape
    sw, sh = max(1, w // scale), max(1, h // scale)
    c_s = cv2.resize(core, (sw, sh), interpolation=cv2.INTER_AREA)
    r_s = cv2.resize(rgb * core[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    est = np.zeros_like(r_s)
    have = np.zeros_like(c_s)
    for s in sigmas:
        num = cv2.GaussianBlur(r_s, (0, 0), s)
        den = cv2.GaussianBlur(c_s, (0, 0), s)
        ok = ((den > 0.03) & (have < 0.5)).astype(np.float32)
        est = est * (1 - ok[..., None]) + (num / np.maximum(den[..., None], 1e-4)) * ok[..., None]
        have = np.maximum(have, ok)
    return (cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR),
            np.clip(cv2.resize(have, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1))


def refine(rgb, a, lo=0.32, hi=0.88, feather=1.0, core_thr=0.9, core_erode=5):
    """rgb float32 HxWx3 (0..255), a float32 HxW (0..1) -> (colour, alpha) with a decisive edge and interior colour."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_erode, core_erode))
    core = cv2.erode((a > core_thr).astype(np.uint8), k).astype(np.float32)
    ext, have = interior_colour(rgb, core)
    ext = ext * have[..., None] + rgb * (1 - have[..., None])
    trust = smooth(a, 0.80, 0.97)[..., None]                     # believe the picture only where he is solid
    fg = rgb * trust + ext * (1 - trust)
    al = cv2.GaussianBlur(smooth(a, lo, hi), (0, 0), feather) if feather > 0 else smooth(a, lo, hi)
    return fg, al


def skin_balance(rgb, y0, y1, strength=0.65, target=(13.0, 19.0), lift=4.0):
    """Pull skin-like pixels between rows y0..y1 toward a healthy brown (Lab a*, b*), gently lift shadows.
    rgb is BGR float32 0..255, edited in place. Hair (dark), whites of eyes and teeth (bright), and the
    strongly coloured hoodie are left alone."""
    y0, y1 = max(0, int(y0)), min(rgb.shape[0], int(y1))
    if y1 <= y0:
        return rgb
    crop = np.ascontiguousarray(rgb[y0:y1] / 255.0)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    w = smooth(L, 31, 42) * (1 - smooth(L, 74, 88)) * (1 - smooth(chroma, 20, 34))
    rows = np.arange(y0, y1, dtype=np.float32)
    zone = smooth(rows, y0, y0 + 0.12 * (y1 - y0)) * (1 - smooth(rows, y0 + 0.88 * (y1 - y0), y1))
    w = w * zone[:, None]
    lab[..., 1] = a + w * strength * 0.6 * (target[0] - a)
    lab[..., 2] = b + w * strength * (target[1] - b)
    lab[..., 0] = L + w * lift * (1 - L / 100.0)
    rgb[y0:y1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255)
    return rgb


def tame_yellow(rgb, y0, amount=0.12):
    """Take a little chroma out of strongly yellow pixels (the mustard hoodie) from row y0 down, keeping their hue,
    so a saturated garment does not pull the eye away from his face. rgb is BGR float32 0..255, edited in place."""
    if amount <= 0:
        return rgb
    y0 = max(0, int(y0))
    crop = np.ascontiguousarray(rgb[y0:] / 255.0)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    k = 1 - amount * smooth(lab[..., 2], 30.0, 48.0)
    lab[..., 1] *= k
    lab[..., 2] *= k
    rgb[y0:] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255)
    return rgb
