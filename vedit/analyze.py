"""Look at the picture: find the Meet layout (shared screen + presenter tile),
the presenter's face, and where on screen things are moving (zoom focus)."""
import subprocess

import cv2
import numpy as np

from .util import log


def grab_frame(video, t):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True)
    if not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def sample_frames(video, duration, n=9):
    frames = (grab_frame(video, duration * (i + 0.5) / n) for i in range(n))
    return [f for f in frames if f is not None]


def _even(v):
    return int(v) // 2 * 2


def detect_layout(frames):
    """Return {"mode", "screen", "tiles"}; rects are [x, y, w, h] in source pixels.

    A Meet / Zoom recording is a flat dark background with a big shared-screen
    rectangle and smaller participant tiles. Anything else is treated as full frame.
    """
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    H, W = med.shape[:2]
    edge = np.concatenate([med[:6].reshape(-1, 3), med[-6:].reshape(-1, 3),
                           med[:, :6].reshape(-1, 3), med[:, -6:].reshape(-1, 3)])
    bgc = np.median(edge, axis=0)
    flat = (np.abs(edge.astype(int) - bgc).sum(axis=1) < 24).mean()
    if flat < 0.8:  # border isn't a flat colour: not a meeting layout
        return {"mode": "full", "screen": None, "tiles": [], "size": [W, H]}
    diff = np.abs(med.astype(np.int16) - bgc.astype(np.int16)).sum(axis=2)
    mask = (diff > 24).astype(np.uint8) * 255
    # Meet leaves ~0.7% of the width between the shared screen and the tiles, so the
    # kernels must stay smaller than that gap or the regions merge into one.
    k = max(3, W // 240)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgmask = diff <= 24
    rects = []
    for c in cnts:
        rects += _split_gutters(list(cv2.boundingRect(c)), bgmask)
    rects = [r for r in rects if r[2] * r[3] > 0.015 * W * H and 0.4 < r[2] / r[3] < 3.5]
    if not rects:
        return {"mode": "full", "screen": None, "tiles": [], "size": [W, H]}
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    screen = rects[0]
    if screen[2] * screen[3] > 0.85 * W * H:
        return {"mode": "full", "screen": None, "tiles": [], "size": [W, H]}
    tiles = [r for r in rects[1:]
             if r[0] >= screen[0] + screen[2] * 0.8 or r[1] >= screen[1] + screen[3] * 0.8]
    tiles.sort(key=lambda r: (r[1], r[0]))
    screen = [_even(screen[0]), _even(screen[1]), _even(screen[2]), _even(screen[3])]
    tiles = [[_even(v) for v in r] for r in tiles]
    return {"mode": "screen_face" if tiles else "screen", "screen": screen, "tiles": tiles,
            "size": [W, H]}


def _split_gutters(rect, bgmask, depth=0):
    """Split a region along any full-width/full-height strip of background colour
    (the gap between two stacked tiles that the morphology bridged)."""
    x, y, w, h = rect
    if depth > 3 or w < 40 or h < 40:
        return [rect]
    sub = bgmask[y:y + h, x:x + w]
    for axis, size in ((1, h), (0, w)):  # rows first, then columns
        frac = sub.mean(axis=axis)
        lo, hi = int(size * 0.08), int(size * 0.92)
        gut = np.where(frac[lo:hi] > 0.9)[0] + lo
        if len(gut) >= 2:
            a, b = int(gut.min()), int(gut.max())
            if axis == 1:
                parts = [[x, y, w, a], [x, y + b + 1, w, h - b - 1]]
            else:
                parts = [[x, y, a, h], [x + b + 1, y, w - b - 1, h]]
            return [r for p in parts for r in _split_gutters(p, bgmask, depth + 1)]
    return [rect]


def auto_trim(frames, rect, max_top=0.25, max_bottom=0.14):
    """Fractions [top, bottom, 0, 0] of static rows (browser tabs, ribbons, taskbar)
    at the edges of the shared screen. Rows where nothing ever changes are UI chrome."""
    x, y, w, h = rect
    g = np.stack([cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.int16)
    changed = (np.abs(np.diff(g, axis=0)) > 25).mean(axis=0).mean(axis=1)
    active = np.where(changed > 0.03)[0]  # the taskbar clock alone stays under 3%
    if len(active) < 0.4 * h:
        return [0, 0, 0, 0]
    top = min(active[0] / h, max_top)
    bottom = min((h - 1 - active[-1]) / h, max_bottom)
    return [round(float(top), 4), round(float(bottom), 4), 0, 0]


def face_center(frames, rect=None):
    """Median face centre (fx, fy as 0..1 of rect) over sampled frames, or None."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    pts = []
    for f in frames:
        x, y, w, h = rect or [0, 0, f.shape[1], f.shape[0]]
        roi = cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(roi, 1.1, 5, minSize=(max(24, w // 14), max(24, w // 14)))
        if len(found):
            fx, fy, fw, fh = max(found, key=lambda r: r[2] * r[3])
            pts.append(((fx + fw / 2) / w, (fy + fh / 2) / h, fh / h))
    if not pts:
        return None
    a = np.median(np.array(pts), axis=0)
    return {"x": float(a[0]), "y": float(a[1]), "size": float(a[2])}


def face_crop(tile, face, aspect, fill=0.9):
    """Crop inside the presenter tile at the card aspect, keeping the face in the upper middle."""
    tx, ty, tw, th = tile
    ch = th * fill
    cw = ch * aspect
    if cw > tw:
        cw, ch = tw, tw / aspect
    fx = (face or {}).get("x", 0.5) * tw
    fy = (face or {}).get("y", 0.4) * th
    x0 = min(max(fx - cw / 2, 0), tw - cw)
    y0 = min(max(fy - ch * 0.42, 0), th - ch)
    return [_even(tx + x0), _even(ty + y0), _even(cw), _even(ch)]


class Motion:
    """Low-res greyscale frames of one region, used to aim zooms at the activity."""

    def __init__(self, video, region, fps=4, width=256):
        x, y, w, h = region
        hh = max(2, int(round(width * h / w / 2)) * 2)
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                            f"fps={fps},crop={w}:{h}:{x}:{y},scale={width}:{hh},format=gray",
                            "-f", "rawvideo", "-"], capture_output=True)
        self.fps = fps
        self.frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, hh, width)
        # Pixels that change all the time (live price axis, clock, ticker) say nothing
        # about where he is pointing; weight them down.
        busy = np.zeros(self.frames.shape[1:], np.float64)
        for i in range(1, len(self.frames)):
            busy += np.abs(self.frames[i].astype(np.int16) - self.frames[i - 1]) > 20
        self.weight = 1.0 / (1.0 + 40.0 * busy / max(1, len(self.frames) - 1))
        log(f"motion map: {len(self.frames)} frames of {width}x{hh}")

    def focus(self, t0, t1, clamp=(0.22, 0.78)):
        i0, i1 = max(0, int(t0 * self.fps)), min(len(self.frames), int(t1 * self.fps) + 1)
        if i1 - i0 < 2:
            return None
        d = np.abs(np.diff(self.frames[i0:i1].astype(np.int16), axis=0)).sum(axis=0)
        d = np.where(d > 40, d, 0).astype(np.float64) * self.weight
        total = d.sum()
        if total < 2000:
            return None
        if (d > 0).mean() > 0.35:  # scrolling / page change: whole frame moved, stay centred
            return None
        ys, xs = np.indices(d.shape)
        fx = float((d * xs).sum() / total / d.shape[1])
        fy = float((d * ys).sum() / total / d.shape[0])
        return (min(max(fx, clamp[0]), clamp[1]), min(max(fy, clamp[0]), clamp[1]))


def layout_preview(frame, layout, out_png):
    img = frame.copy()
    if layout.get("screen"):
        x, y, w, h = layout["screen"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (220, 198, 113), 4)
        cv2.putText(img, "SCREEN", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 198, 113), 3)
    if layout.get("content"):
        x, y, w, h = layout["content"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (71, 179, 255), 3)
        cv2.putText(img, "CONTENT (after trim)", (x + 10, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (71, 179, 255), 2)
    for i, t in enumerate(layout.get("tiles", [])):
        x, y, w, h = t
        cv2.rectangle(img, (x, y), (x + w, y + h), (177, 198, 92), 3)
        cv2.putText(img, f"TILE {i}", (x + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (177, 198, 92), 2)
    if layout.get("face"):
        x, y, w, h = layout["face"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (126, 142, 255), 3)
        cv2.putText(img, "FACE CROP", (x + 10, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (126, 142, 255), 2)
    for gx in range(0, img.shape[1], 100):
        cv2.line(img, (gx, 0), (gx, 12), (255, 255, 255), 1)
        cv2.putText(img, str(gx), (gx + 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for gy in range(0, img.shape[0], 100):
        cv2.line(img, (0, gy), (12, gy), (255, 255, 255), 1)
        cv2.putText(img, str(gy), (14, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out_png), img)
