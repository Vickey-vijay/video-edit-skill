"""Backgrounds that sit behind the matted subject: one calm "After Hours" world.

A deep blue-black room with one warm practical light low on the right, a cool fill on the left, a
single soft key streak from the top left and slowly evolving haze. The three themes keep that rig,
grade and camera and change only one mid-layer element:

  market  frosted glass with a few defocused price levels and one price path that draws itself once
  gold    a brass guilloche rosette, like the engraving on a banknote or a share certificate
  studio  nothing but light and haze

So a topic change (the plan's theme schedule) moves the mood without changing the brand.

Rules that keep it expensive rather than "screensaver": everything is a pure function of the time
t and the cut index k (frames can be rendered in any order and in parallel pieces), nothing is
re-randomised per frame, every motion is slower than about 1% of the frame width per second, the
camera drifts instead of objects floating, and every edit cut steps the camera a little (a new
angle, as in a multi-camera shoot). Frames are drawn at half size and enlarged: soft, like a lens.
All values are float32 BGR 0..255."""
import math

import cv2
import numpy as np


def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# Two palettes of the same room. "graphite" (default) is warm neutral slate with amber light and brass: no blue, because a
# navy/cyan look was rejected earlier as "AI-looking". "ink" is the blue-black variant.
PALETTES = {
    "graphite": {"ink": hex_bgr("0A0A0B"), "mid": hex_bgr("15181B"), "haze": hex_bgr("1E2328"), "pool_hi": hex_bgr("6B4A20"),
                 "pool_lo": hex_bgr("2C2010"), "fill": hex_bgr("26303A"), "key": hex_bgr("F2E6D2"), "data": hex_bgr("D2C09A"),
                 "brass": hex_bgr("C9A15B")},
    "ink": {"ink": hex_bgr("070B10"), "mid": hex_bgr("0E1620"), "haze": hex_bgr("0F1923"), "pool_hi": hex_bgr("5E4222"),
            "pool_lo": hex_bgr("2A1F12"), "fill": hex_bgr("163048"), "key": hex_bgr("F2E6D2"), "data": hex_bgr("7FA3B8"),
            "brass": hex_bgr("C9A15B")},
}


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


class _Noise:
    """Smooth noise that evolves slowly and is a pure function of t: every octave is a random field
    A and a second random field B blended as A cos(wt) + B sin(wt), which keeps its strength constant."""

    def __init__(self, rng, w, h, cells=((4, 7), (9, 16), (20, 36), (44, 80)), gains=(0.55, 0.6, 0.45, 0.3),
                 speeds=(0.11, 0.17, 0.23, 0.31)):
        self.w, self.h = w, h
        self.A = [rng.normal(size=(cy, cx)).astype(np.float32) for cx, cy in cells]
        self.B = [rng.normal(size=(cy, cx)).astype(np.float32) for cx, cy in cells]
        self.g, self.s = gains, speeds
        self.norm = 1.0 / math.sqrt(sum(g * g for g in gains))

    def field(self, t):
        acc = np.zeros((self.h, self.w), np.float32)
        for A, B, g, s in zip(self.A, self.B, self.g, self.s):
            acc += g * cv2.resize(A * math.cos(s * t) + B * math.sin(s * t), (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        return acc * self.norm


def _rounded_box(w, h, x0, y0, x1, y1, r):
    """Signed distance (pixels, negative inside) to a rounded rectangle."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx, qy = np.abs(x - cx) - hx, np.abs(y - cy) - hy
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r


class AfterHours:
    """The shared rig. `variant` chooses the mid-layer: levels | rosette | plain."""

    variant = "plain"

    def __init__(self, w, h, seed=7, palette="graphite"):
        self.w, self.h = w, h
        self.P = P = PALETTES.get(palette, PALETTES["graphite"])
        rng = np.random.default_rng(seed)
        self.noise = _Noise(rng, w, h)
        self.warp = _Noise(rng, w, h, cells=((2, 3), (4, 7)), gains=(1.0, 0.6), speeds=(0.07, 0.13))
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        self.xx, self.yy = x, y
        # --- base: ink room, lighter behind the head so dark hair separates, warm pool low right, cool fill left
        base = np.zeros((h, w, 3), np.float32)
        ys = (y / h)[..., None]
        top, mid, bot = (np.array(c, np.float32) for c in (P["mid"], P["haze"], P["ink"]))
        base[:] = top * (1 - _smooth(ys, 0.0, 0.5)) + mid * _smooth(ys, 0.0, 0.5)
        base = base * (1 - _smooth(ys, 0.55, 1.0)) + bot * _smooth(ys, 0.55, 1.0)

        def pool(cx, cy, rad, col, k):
            d = np.sqrt(((x - cx * w) / (rad * h)) ** 2 + ((y - cy * h) / (rad * h)) ** 2)
            return (np.exp(-d * d * 2.4)[..., None] * np.array(col, np.float32) * k)

        base += pool(0.82, 0.68, 0.40, P["pool_hi"], 0.62)
        base += pool(0.80, 0.76, 0.66, P["pool_lo"], 0.60)
        base += pool(0.04, 0.26, 0.40, P["fill"], 0.60)
        self.base = base
        # busy detail (price lines, the path, the rosette) fades out around where his head is: nothing may touch his face or hair
        r = np.sqrt(((x - 0.5 * w) / (0.30 * w)) ** 2 + ((y - 0.27 * h) / (0.20 * h)) ** 2)
        self.clear = (0.10 + 0.90 * _smooth(r, 0.80, 1.55)).astype(np.float32)
        # --- key streak from the top left, anisotropic and soft
        ang = math.radians(28)
        u = (x - 0.05 * w) * math.cos(ang) + (y - 0.0 * h) * math.sin(ang)      # along the streak
        v = -(x - 0.05 * w) * math.sin(ang) + (y - 0.0 * h) * math.cos(ang)     # across it
        band = np.exp(-((v - 0.16 * h) / (0.085 * h)) ** 2) * (1 - _smooth(u, 0.25 * h, 1.05 * h))
        self.streak = (band * _smooth(u, -0.1 * h, 0.12 * h)).astype(np.float32)[..., None]
        # --- mid-layer, static parts
        self.glass = None
        if self.variant == "levels":
            self._build_glass(rng)
        elif self.variant == "rosette":
            self._build_rosette()

    # ------------------------------------------------------------------ mid layers
    def _build_glass(self, rng):
        w, h = self.w, self.h
        x0, y0, x1, y1 = -0.15 * w, -0.10 * h, 1.15 * w, 0.585 * h
        sd = _rounded_box(w, h, x0, y0, x1, y1, 0.05 * w)
        inside = _smooth(-sd, -0.8, 0.8) * (1 - 0.8 * _smooth(self.yy, 0.40 * h, 0.58 * h))
        frost = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 6.0)
        frost = frost / (frost.std() + 1e-6)
        edge = np.exp(-(sd / 1.1) ** 2)
        # the edge catches the key light: brighter toward the top left
        light = (1.25 - 0.75 * (self.xx / w) - 0.5 * (self.yy / h)).clip(0.25, 1.3)
        self.glass = {"fill": (inside * (1.0 + 0.10 * frost)).astype(np.float32)[..., None],
                      "edge": (edge * light).astype(np.float32)[..., None], "box": (x0, y0, x1, y1)}
        # price levels (seeded) and one smooth price path across the glass
        self.levels = np.sort(rng.uniform(y0 + 0.12 * (y1 - y0), y1 - 0.10 * (y1 - y0), 6))
        self.level_a = rng.uniform(0.55, 1.0, 6)
        n = 160
        walk = np.cumsum(rng.normal(0.0, 1.0, n + 80))
        walk = cv2.GaussianBlur(walk.astype(np.float32).reshape(1, -1), (0, 0), 9.0).ravel()[40:40 + n]
        walk = (walk - walk.min()) / max(walk.max() - walk.min(), 1e-6)
        self.path = np.stack([np.linspace(x0 + 0.05 * (x1 - x0), x1 - 0.05 * (x1 - x0), n),
                              y1 - 0.14 * (y1 - y0) - walk * 0.62 * (y1 - y0)], 1).astype(np.float32)

    def _build_rosette(self):
        """Guilloche: many rose curves with shifted phases make the moire of banknote engraving."""
        w, h = self.w, self.h
        size = int(1.7 * w)
        layer = np.zeros((size, size), np.float32)
        c = size / 2
        # hypotrochoids (the Spirograph curve), many of them with a slowly changing pen distance: the
        # overlap makes the fine interference weave of a banknote rosette
        for (R, r, d0, d1, n, sc, turns) in ((100.0, 8.0, 14.0, 42.0, 13, 0.0033, 2), (100.0, 12.0, 8.0, 30.0, 9, 0.0033, 3)):
            t = np.linspace(0, 2 * math.pi * turns, 5200)
            for j in range(n):
                d = d0 + (d1 - d0) * j / max(1, n - 1)
                x = (R - r) * np.cos(t) + d * np.cos((R - r) / r * t)
                y = (R - r) * np.sin(t) - d * np.sin((R - r) / r * t)
                pts = np.stack([c + x * size * sc, c + y * size * sc], 1).astype(np.float32)
                cv2.polylines(layer, [np.round(pts * 16).astype(np.int32)], True, 1.0, 1, cv2.LINE_AA, 4)
        self.rosette = cv2.GaussianBlur(layer, (0, 0), 0.9)
        self.rosette = np.clip(self.rosette / max(float(self.rosette.max()), 1e-6) * 1.6, 0, 1)

    # ------------------------------------------------------------------ camera
    def camera(self, t, k):
        """Slow drift plus a small deterministic step for each edit cut (a new angle)."""
        r = np.random.default_rng(9000 + int(k))
        sx = r.choice([-1.0, 1.0]) * r.uniform(0.6, 1.0) * 16.0          # half-size pixels (about 32-34 px at 1080)
        sy = r.uniform(-1.0, 1.0) * 6.0
        sc = 1.0 + r.choice([-1.0, 1.0]) * r.uniform(0.015, 0.03)
        return (10.0 * math.sin(2 * math.pi * t / 41.0 + 0.7) + sx, 5.0 * math.sin(2 * math.pi * t / 53.0 + 2.1) + sy, sc)

    def _xf(self, img, cam, depth):
        dx, dy, sc = cam
        s = 1.0 + (sc - 1.0) * depth
        c = np.array([self.w / 2, self.h / 2])
        M = np.array([[s, 0, c[0] - s * c[0] + dx * depth], [0, s, c[1] - s * c[1] + dy * depth]], np.float32)
        return cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------ frame
    def frame(self, t, k=0):
        w, h = self.w, self.h
        cam = self.camera(t, k)
        out = self._xf(self.base, cam, 0.35)
        # haze: domain-warped, slow, a few percent of luminance with a cool tint
        n = self.noise.field(t)
        wf = self.warp.field(t) * 14.0
        n = cv2.remap(n, self.xx + wf, self.yy + np.roll(wf, 97, axis=0), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        out *= (1.0 + 0.026 * n)[..., None]
        out += (np.maximum(n, 0) * 0.9)[..., None] * np.array(self.P["haze"], np.float32) * 0.05
        # key streak, breathing very slowly
        out += self._xf(self.streak, cam, 0.5).reshape(h, w, 1) * np.array(self.P["key"], np.float32) * (0.040 + 0.006 * math.sin(2 * math.pi * t / 20.0))
        if self.variant == "levels":
            self._levels(out, t, cam)
        elif self.variant == "rosette":
            self._rosette(out, t, cam)
        return out

    def _levels(self, out, t, cam):
        w, h = self.w, self.h
        g = self.glass
        fill = self._xf(g["fill"][..., 0], cam, 0.7)[..., None]
        edge = self._xf(g["edge"][..., 0], cam, 0.7)[..., None]
        out += fill * np.array(self.P["haze"], np.float32) * 0.30 + edge * np.array(self.P["key"], np.float32) * 0.055
        dx, dy = cam[0] * 1.0, cam[1] * 1.0
        x0, y0, x1, y1 = g["box"]
        layer = np.zeros((h, w), np.float32)
        for yv, a in zip(self.levels, self.level_a):
            yy = int(round(yv + dy))
            cv2.line(layer, (int(x0 + 0.03 * (x1 - x0) + dx), yy), (int(x1 - 0.03 * (x1 - x0) + dx), yy), float(a), 1, cv2.LINE_AA)
        lines = cv2.GaussianBlur(layer, (0, 0), 2.6) * self.clear
        out += lines[..., None] * np.array(self.P["data"], np.float32) * 0.24
        # the price path draws itself once (ease-out over 8 s), then holds; it sits further back: blurred, less parallax
        p = 1.0 - (1.0 - min(1.0, max(0.0, t / 8.0))) ** 3
        m = max(2, int(len(self.path) * p))
        pts = (self.path[:m] + np.array([dx * 0.7, dy * 0.7], np.float32))
        pl = np.zeros((h, w), np.float32)
        cv2.polylines(pl, [np.round(pts * 8).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA, 3)
        core = cv2.GaussianBlur(pl, (0, 0), 2.0)
        glow = cv2.GaussianBlur(pl, (0, 0), 7.0)
        out += ((core * 0.85 + glow * 1.8) * self.clear)[..., None] * np.array(self.P["data"], np.float32) * 0.62

    def _rosette(self, out, t, cam):
        w, h = self.w, self.h
        size = self.rosette.shape[0]
        ang = 0.5 * t
        cx, cy = 0.70 * w + cam[0] * 0.8, 0.28 * h + cam[1] * 0.8
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0 * cam[2] ** 0.8)
        M[0, 2] += cx - size / 2
        M[1, 2] += cy - size / 2
        lay = cv2.warpAffine(self.rosette, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # fades out toward the frame edges so it stays an engraving behind him, not a pattern over the frame
        fade = np.exp(-(((self.xx - cx) / (0.62 * w)) ** 2 + ((self.yy - cy) / (0.50 * h)) ** 2) * 1.6)
        out += (lay * fade * self.clear)[..., None] * np.array(self.P["brass"], np.float32) * 0.15


class Market(AfterHours):
    variant = "levels"


class Gold(AfterHours):
    variant = "rosette"


class Studio(AfterHours):
    variant = "plain"


THEMES = {"gold": Gold, "market": Market, "studio": Studio}


def make(theme, w, h, seed=0, palette="graphite"):
    return THEMES.get(theme, Studio)(w, h, seed or 7, palette)
