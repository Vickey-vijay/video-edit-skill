"""Animated procedural backgrounds that sit behind the matted subject.

They are drawn at half resolution and upscaled, which also gives them the soft,
out-of-focus depth a real lens would. All of them stay dark around the subject so
his bright clothing keeps its separation; the colour lives in light, not in fills."""
import math

import cv2
import numpy as np

from .fx import gradient_fill


def _radial(w, h, cx, cy, radius, color, strength):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    falloff = np.exp(-(d ** 2) * 2.2)[..., None]
    return falloff * np.array(color, np.float32) * strength


class _Bokeh:
    def __init__(self, rng, w, h, n, rmin, rmax, colors, alpha=(0.10, 0.34)):
        self.w, self.h = w, h
        self.x = rng.uniform(0, w, n)
        self.y = rng.uniform(0, h, n)
        self.r = rng.uniform(rmin, rmax, n)
        self.vx = rng.uniform(-6, 6, n) * (w / 540)
        self.vy = rng.uniform(-14, -3, n) * (w / 540)
        self.ph = rng.uniform(0, 6.28, n)
        self.a = rng.uniform(*alpha, n)
        self.col = np.array([colors[i % len(colors)] for i in range(n)], np.float32)

    def draw(self, layer, t, depth_blur=1.8):
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t) % (self.w + 2 * self.r[i]) - self.r[i]
            y = (self.y[i] + self.vy[i] * t) % (self.h + 2 * self.r[i]) - self.r[i]
            a = self.a[i] * (0.65 + 0.35 * math.sin(t * 0.7 + self.ph[i]))
            cv2.circle(layer, (int(x), int(y)), int(self.r[i]), (self.col[i] * a).tolist(), -1, cv2.LINE_AA)
        return cv2.GaussianBlur(layer, (0, 0), depth_blur)


class Gold:
    """Near-black amber with a warm halo behind the head, bokeh at two depths, drifting dust, slow light rays."""

    GOLDS = [(60, 194, 255), (34, 150, 240), (90, 215, 255), (20, 110, 200)]

    def __init__(self, w, h, seed=7):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        base = gradient_fill(h, w, [(0.0, (4, 7, 12)), (0.40, (8, 18, 34)), (0.72, (5, 11, 20)), (1.0, (3, 5, 9))])
        base += _radial(w, h, w * 0.5, h * 0.34, h * 0.46, (24, 92, 180), 0.24)
        base += _radial(w, h, w * 1.0, h * 0.0, h * 0.40, (36, 130, 230), 0.20)
        self.base = base
        self.far = _Bokeh(rng, w, h, 18, 8 * w / 540, 24 * w / 540, self.GOLDS, (0.10, 0.24))
        self.near = _Bokeh(rng, w, h, 6, 34 * w / 540, 78 * w / 540, self.GOLDS, (0.05, 0.12))
        n = 130
        self.dx = rng.uniform(0, w, n)
        self.dy = rng.uniform(0, h, n)
        self.dv = rng.uniform(5, 26, n) * (w / 540)
        self.dz = rng.uniform(0.6, 2.0, n)
        self.dph = rng.uniform(0, 6.28, n)
        # light rays fanning from the upper right
        self.rays = []
        for ang, wid, ph in ((118, 0.07, 0.0), (132, 0.10, 2.0), (146, 0.06, 4.0), (104, 0.05, 1.0)):
            m = np.zeros((h, w), np.float32)
            ox, oy = w * 1.08, -h * 0.1
            a0, a1 = math.radians(ang - wid * 60), math.radians(ang + wid * 60)
            pts = np.array([[ox, oy], [ox + math.cos(a0) * h * 2, oy + math.sin(a0) * h * 2],
                            [ox + math.cos(a1) * h * 2, oy + math.sin(a1) * h * 2]], np.int32)
            cv2.fillPoly(m, [pts], 1.0)
            self.rays.append((cv2.GaussianBlur(m, (0, 0), w * 0.035), ph))

    def frame(self, t):
        w, h = self.w, self.h
        out = self.base.copy()
        out += self.near.draw(np.zeros((h, w, 3), np.float32), t, w * 0.012)
        out += self.far.draw(np.zeros((h, w, 3), np.float32), t, w * 0.004)
        rays = sum(m * (0.045 + 0.03 * math.sin(t * 0.45 + ph)) for m, ph in self.rays)
        out += rays[..., None] * np.array((50, 160, 255), np.float32)
        dust = np.zeros((h, w, 3), np.float32)
        ys = (self.dy - self.dv * t) % h
        for i in range(len(ys)):
            b = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 2.2 * self.dz[i] + self.dph[i])) ** 2
            x = self.dx[i] + 6 * math.sin(t * 0.5 + self.dph[i])
            cv2.circle(dust, (int(x) % w, int(ys[i])), 1 if self.dz[i] < 1.4 else 2,
                       (np.array((110, 215, 255)) * b).tolist(), -1, cv2.LINE_AA)
        out += dust * 1.25 + cv2.GaussianBlur(dust, (0, 0), 3.0) * 1.4
        return out


class Market:
    """Heavily defocused candles scrolling past, with a glowing gold price line. Charts without the cliché grid."""

    def __init__(self, w, h, seed=11):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        n = 260
        steps = rng.normal(0.18, 1.0, n).cumsum()
        steps += np.sin(np.arange(n) / 14.0) * 5
        o = steps + rng.normal(0, 0.35, n)
        c = steps + rng.normal(0, 0.35, n)
        self.o, self.c = o, c
        self.hi = np.maximum(o, c) + np.abs(rng.normal(0.5, 0.35, n))
        self.lo = np.minimum(o, c) - np.abs(rng.normal(0.5, 0.35, n))
        self.cw = w / 20.0
        self.base = gradient_fill(h, w, [(0.0, (7, 10, 9)), (0.5, (12, 18, 15)), (1.0, (5, 8, 7))])
        self.base += _radial(w, h, w * 0.5, h * 0.38, h * 0.55, (40, 110, 150), 0.22)

    def frame(self, t):
        w, h, cw = self.w, self.h, self.cw
        speed = 0.9                                   # candles per second
        pos = t * speed
        i0 = int(pos)
        vis = range(i0, i0 + 24)
        lo = min(self.lo[k] for k in vis)
        hi = max(self.hi[k] for k in vis)
        sc = (h * 0.42) / max(hi - lo, 1e-3)
        y = lambda p: int(h * 0.74 - (p - lo) * sc)
        layer = np.zeros((h, w, 3), np.float32)
        line = []
        for k in vis:
            x = int((k - pos) * cw + cw * 0.5)
            up = self.c[k] >= self.o[k]
            col = (90, 200, 80) if up else (70, 75, 225)
            cv2.line(layer, (x, y(self.hi[k])), (x, y(self.lo[k])), col, 2, cv2.LINE_AA)
            cv2.rectangle(layer, (x - int(cw * 0.32), y(max(self.o[k], self.c[k]))),
                          (x + int(cw * 0.32), y(min(self.o[k], self.c[k]))), col, -1, cv2.LINE_AA)
            line.append((x, y(self.c[k])))
        layer = cv2.GaussianBlur(layer, (0, 0), w * 0.0055) * 1.25
        glow = np.zeros((h, w, 3), np.float32)
        cv2.polylines(glow, [np.array(line, np.int32)], False, (60, 194, 255), 2, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 1.6) * 0.55 + cv2.GaussianBlur(glow, (0, 0), 8) * 0.45
        return self.base + layer + glow


class Studio:
    """Quiet warm charcoal with a soft pool of light and a few slow bokeh. The calm default."""

    def __init__(self, w, h, seed=5):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        base = gradient_fill(h, w, [(0.0, (16, 15, 16)), (0.5, (30, 28, 30)), (1.0, (12, 11, 12))])
        base += _radial(w, h, w * 0.5, h * 0.4, h * 0.55, (45, 80, 125), 0.30)
        self.base = base
        self.b = _Bokeh(rng, w, h, 9, 20 * w / 540, 60 * w / 540, [(190, 220, 245), (60, 150, 235)], (0.04, 0.10))

    def frame(self, t):
        return self.base + self.b.draw(np.zeros((self.h, self.w, 3), np.float32), t, self.w * 0.012)


THEMES = {"gold": Gold, "market": Market, "studio": Studio}


def make(theme, w, h, seed=0):
    cls = THEMES.get(theme, Studio)
    return cls(w, h, seed or 7)
