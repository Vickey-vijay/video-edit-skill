"""Animated graphic elements for short-form video: hook type that sits behind the
subject, stat stickers, tag banners and an end card. Each element knows when it is
on screen, what to draw behind the subject (back) and in front of it (front), and
which sound effects it wants.

Times are edited-video seconds. Elements are built from the per-video graphics spec."""
import math

import cv2
import numpy as np

from . import fx
from .fx import (GOLD, GOLD_HI, GOLD_LO, INK, IVORY, blit, ease_in_out, ease_out_cubic, spring,
                 text_sprite, text_size, rect_sprite)

W, H = 1080, 1920
GOLD_GRAD = [(0.0, GOLD_HI), (0.5, GOLD), (1.0, GOLD_LO)]


def _life(t, t0, t1, tin=0.42, tout=0.28):
    """(entry progress 0..1, exit progress 0..1) for an element alive between t0 and t1."""
    return (t - t0) / tin, (t1 - t) / tout


class Element:
    sfx = ()

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1

    def alive(self, t):
        return self.t0 <= t <= self.t1

    def back(self, canvas, t):
        pass

    def front(self, canvas, t):
        pass


def _fit(sprite, max_w):
    h, w = sprite.shape[:2]
    if w <= max_w:
        return sprite
    k = max_w / w
    return cv2.resize(sprite, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def _ink(sprite, thr=0.04):
    """Crop a sprite to where it actually has ink (text sprites carry padding)."""
    ys, xs = np.where(sprite[..., 3] > thr)
    if not len(ys):
        return sprite
    return sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _stack(parts, gaps, padx, pady):
    """Centre sprites in a column on one transparent sprite; returns (sprite, w, h)."""
    wmax = max(p.shape[1] for p in parts)
    h = int(sum(p.shape[0] for p in parts) + sum(gaps[:len(parts) - 1]) + 2 * pady)
    w = int(wmax + 2 * padx)
    out = np.zeros((h, w, 4), np.float32)
    y = pady
    for i, p in enumerate(parts):
        ph, pw = p.shape[:2]
        x = (w - pw) // 2
        roi = out[int(y):int(y) + ph, x:x + pw]
        a = p[..., 3:4]
        roi[..., :3] = p[..., :3] + roi[..., :3] * (1 - a)
        roi[..., 3:4] = a + roi[..., 3:4] * (1 - a)
        y += ph + (gaps[i] if i < len(gaps) else 0)
    return out, w, h


class Hook(Element):
    """The opening, in the headroom above his head: a letter-spaced brass kicker with a hairline under it,
    then a two-line ivory headline (the accent line in brass). It fades up slowly and leaves the same way."""

    def __init__(self, t0, t1, behind="", lines=(), accent=None):
        super().__init__(t0, t1)
        self.accent = accent
        self.sfx = (("whoosh", t0 - 0.1),)
        self._scrim = None
        # (sprite with halo, height of the ink) so spacing follows the letters, not the halo padding
        def mk(txt, size, col, trk, blur):
            ink = _ink(text_sprite(txt, "sans", size, fill=col, tracking=trk)).shape[0]
            return fx.text_halo(txt, "sans", size, col, tracking=trk, blur=blur, a=0.55), ink
        self.kicker = mk(behind.upper(), 28, fx.BRASS_HI, 11, 8) if behind else None
        self.lines = [mk(ln.upper(), 64 if len(ln) < 15 else 54, fx.BRASS_HI if i == accent else fx.IVORY2, 3, 10)
                      for i, ln in enumerate(lines)]

    def back(self, canvas, t):
        if not getattr(self, "talk", True):          # over footage there is no subject: darken the top for legibility
            a_in, a_out = _life(t, self.t0, self.t1, 0.55, 0.35)
            if self._scrim is None:
                self._scrim = np.clip((np.linspace(1.0, 0.0, H)[:, None, None] - 0.62) / 0.38, 0, 1).astype(np.float32)
            canvas *= 1 - 0.55 * ease_out_cubic(a_in) * ease_in_out(a_out) * self._scrim

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0 + 0.1, self.t1, 0.8, 0.45)
        k, out = ease_out_cubic(a_in), ease_in_out(a_out)
        if k <= 0 or out <= 0:
            return
        y = H * 0.034
        if self.kicker is not None:
            sp, ink = self.kicker
            blit(canvas, sp, W / 2, y + ink / 2 + (1 - k) * 14, alpha=k * out)
            wipe = ease_out_cubic((t - self.t0 - 0.2) / 0.7)
            if wipe > 0.01:
                blit(canvas, rect_sprite(max(2, int(130 * wipe)), 3, fx.BRASS, 0.9), W / 2, y + ink + 20, alpha=out)
            y += ink + 52
        for i, (sp, ink) in enumerate(self.lines):
            kk = ease_out_cubic((t - self.t0 - 0.1 - 0.12 * (i + 1)) / 0.8)
            if kk > 0:
                blit(canvas, sp, W / 2, y + ink / 2 + (1 - kk) * 26, alpha=kk * out)
            y += ink * 1.55


class Stat(Element):
    """A frosted-glass card with a brass hairline: small letter-spaced label, a big ivory figure, a quiet line below.
    It slides in from its side and fades; no tilt, no bounce."""

    def __init__(self, t0, t1, kicker="", value="", sub="", side="left", y=0.33, tilt=None):
        super().__init__(t0, t1)
        self.side, self.y = side, y
        self.kicker, self.value, self.sub = kicker, value, sub
        self.sfx = (("whoosh", t0 - 0.1),)
        self._cache = {}

    def _build(self, s):
        key = int(round(s * 20))
        if key not in self._cache:
            parts = []
            if self.kicker:
                parts.append(_ink(text_sprite(self.kicker.upper(), "sans", int(27 * s), fill=fx.BRASS_HI, tracking=6)))
            parts.append(_ink(text_sprite(self.value, "black", int(112 * s), fill=fx.IVORY2, tracking=1)))
            if self.sub:
                parts.append(_ink(text_sprite(self.sub.upper(), "sans", int(24 * s), fill=fx.MUTED, tracking=4)))
            gaps = [int(18 * s)] + [int(16 * s)] * 2
            content, w, h = _stack(parts, gaps, int(44 * s), int(34 * s))
            self._cache[key] = (fx.Panel(w, h, radius=8), content)
        return self._cache[key]

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.6, 0.35)
        k, out = ease_out_cubic(a_in), ease_in_out(a_out)
        if k <= 0 or out <= 0:
            return
        s, head = 1.0, getattr(self, "avoid", None)       # (left, right) of his head in this frame, when he is cut out
        if head is not None:
            room = (head[0] if self.side == "left" else W - head[1]) - 56 - 24
            s = max(0.70, min(1.0, room / self._build(1.0)[0].w))
        s = round(s * 20) / 20
        panel, content = self._build(s)
        if head is not None:
            x = 24 + panel.w / 2 if self.side == "left" else W - 24 - panel.w / 2
        else:
            x = W * (0.22 if self.side == "left" else 0.78)
        slide = (1 - k) * 44 * (-1 if self.side == "left" else 1)
        cy = H * self.y
        panel.draw(canvas, x + slide, cy, alpha=k * out)
        blit(canvas, content, x + slide, cy, alpha=k * out)


class Tag(Element):
    """A short statement on a frosted pill with a small brass dot."""

    def __init__(self, t0, t1, text="", y=0.215, accent=""):
        super().__init__(t0, t1)
        self.y = y
        self.sfx = (("whoosh", t0 - 0.05),)
        s = _ink(text_sprite(text.upper(), "sans", 40, fill=fx.IVORY2, tracking=5))
        s = _fit(s, W * 0.74)
        th = s.shape[0]
        dot = 14
        padx, pady = 50, 26
        w, h = s.shape[1] + dot + 22 + 2 * padx, th + 2 * pady
        content = np.zeros((h, w, 4), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.clip(dot / 2 + 0.5 - np.hypot(xx - (padx + dot / 2), yy - h / 2), 0, 1)
        content[..., :3] = np.array(fx.BRASS, np.float32) * d[..., None]
        content[..., 3] = d
        x0 = padx + dot + 22
        roi = content[pady:pady + th, x0:x0 + s.shape[1]]
        a = s[..., 3:4]
        roi[..., :3] = s[..., :3] + roi[..., :3] * (1 - a)
        roi[..., 3:4] = a + roi[..., 3:4] * (1 - a)
        self.content = content
        self.panel = fx.Panel(w, h, radius=8)

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.6, 0.35)
        k, out = ease_out_cubic(a_in), ease_in_out(a_out)
        if k <= 0 or out <= 0:
            return
        cy = H * self.y + (1 - k) * 22
        self.panel.draw(canvas, W / 2, cy, alpha=k * out)
        blit(canvas, self.content, W / 2, cy, alpha=k * out)


class Cta(Element):
    """End card: the frame dims, a brass hairline draws, one elegant italic question, a quiet line below."""

    def __init__(self, t0, t1, text="", sub=""):
        super().__init__(t0, t1)
        self.sfx = (("whoosh", t0 - 0.1),)
        face, txt = ("black", text.upper()) if "₹" in text else ("serif", text[:1].upper() + text[1:].lower())
        self.s = _fit(fx.text_halo(txt, face, 150 if face == "serif" else 120, fx.IVORY2, tracking=2 if face == "serif" else 4, blur=12, a=0.5), W * 0.92)
        self.sub = fx.text_halo(sub.upper(), "sans", 34, fx.BRASS_HI, tracking=8, blur=8, a=0.5) if sub else None

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.7, 0.01)
        k = ease_out_cubic(a_in)
        canvas *= 1 - 0.55 * k
        p = ease_out_cubic((t - self.t0 - 0.15) / 0.9)
        cy = H * 0.44
        wipe = ease_out_cubic((t - self.t0 - 0.1) / 0.8)
        if wipe > 0.01:
            blit(canvas, rect_sprite(max(2, int(190 * wipe)), 3, fx.BRASS, 0.9), W / 2, cy - self.s.shape[0] * 0.5 - 18, alpha=1.0)
        blit(canvas, self.s, W / 2, cy + (1 - p) * 30, alpha=p)
        if self.sub is not None:
            q = ease_out_cubic((t - self.t0 - 0.55) / 0.8)
            blit(canvas, self.sub, W / 2, cy + self.s.shape[0] * 0.5 + 40 + (1 - q) * 16, alpha=q)


class Interlude(Element):
    """Words on the background while the picture was unusable (the camera was being moved): centred ivory type, the accent line in brass."""

    def __init__(self, t0, t1, lines=(), accent=None):
        super().__init__(t0, t1)
        self.accent = accent
        self.sfx = (("whoosh", t0 - 0.05),)
        self.s_lines = [_fit(fx.text_halo(ln.upper(), "black", 116 if len(ln) < 12 else 88, fx.BRASS_HI if i == accent else fx.IVORY2,
                                          tracking=4, blur=12, a=0.5), W * 0.9) for i, ln in enumerate(lines)]

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.7, 0.4)
        out = ease_in_out(a_out)
        if out <= 0:
            return
        total = sum(sp.shape[0] * 0.8 for sp in self.s_lines)
        y = H * 0.40 - total / 2
        for i, sp in enumerate(self.s_lines):
            p = ease_out_cubic((t - self.t0 - 0.12 * i) / 0.8)
            h = sp.shape[0]
            if p > 0:
                blit(canvas, sp, W / 2, y + h * 0.5 + (1 - p) * 30, alpha=p * out)
            y += h * 0.8


def build(spec, tl, duration):
    """Create elements from the graphics spec; times given in SOURCE seconds are mapped
    through the edit, and anything that sat in a cut-out stretch is dropped."""
    out = []

    def at(src_t):
        return tl.to_new(src_t) if tl.inside(src_t) else None

    h = (spec or {}).get("hook")
    if h:
        t0 = float(h.get("at_edit", 0.1))
        out.append(Hook(t0, t0 + float(h.get("dur", 3.0)), h.get("behind", ""), h.get("lines", []), h.get("accent")))
    for s in (spec or {}).get("stats", []):
        t0 = at(float(s["at"]))
        if t0 is not None:
            out.append(Stat(t0, min(duration - 0.1, t0 + float(s.get("dur", 3.0))), s.get("kicker", ""),
                            s.get("value", ""), s.get("sub", ""), s.get("side", "left"), s.get("y", 0.33)))
    for s in (spec or {}).get("tags", []):
        t0 = at(float(s["at"]))
        if t0 is not None:
            out.append(Tag(t0, min(duration - 0.1, t0 + float(s.get("dur", 2.6))), s["text"], s.get("y", 0.108)))
    for s in (spec or {}).get("cards", []):
        t0 = at(float(s["at"]))
        if t0 is not None:
            out.append(Interlude(t0, min(duration - 0.1, t0 + float(s.get("dur", 2.0))), s.get("lines", []), s.get("accent")))
    c = (spec or {}).get("cta")
    if c:
        d = float(c.get("dur", 2.4))
        out.append(Cta(max(0.0, duration - d), duration, c.get("text", ""), c.get("sub", "")))
    return out


class Captions:
    """English captions: Montserrat ExtraBold, white, a soft dark shadow, over the darkened bottom of the frame. No box."""

    def __init__(self, events, size=54, y=0.765):
        self.events, self.size, self.y = events, size, y
        self.cache = {}

    def _build(self, text):
        if text in self.cache:
            return self.cache[text]
        parts = [_ink(text_sprite(line, "sans", self.size, fill=(252, 251, 250), tracking=0.6)) for line in text.split(chr(92) + "N")]
        lh = int(self.size * 1.30)
        content, w, h = _stack(parts, [lh - p.shape[0] for p in parts], 20, 14)
        content = fx.with_halo(content, blur=7, a=0.85)
        content = fx.with_halo(content, blur=14, a=0.55)
        self.cache[text] = content
        return content

    def draw(self, canvas, t):
        for a, b, text in self.events:
            if a <= t <= b:
                k = ease_out_cubic((t - a) / 0.10)
                out = ease_in_out((b - t) / 0.08)
                blit(canvas, self._build(text), W / 2, H * self.y + (1 - k) * 10, alpha=min(1.0, k * 1.4) * out)
                return
