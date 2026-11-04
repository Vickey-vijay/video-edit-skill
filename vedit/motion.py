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


class Hook(Element):
    """The opening: a giant word behind his head, a kicker above it, a gold marker line."""

    def __init__(self, t0, t1, behind="", lines=(), accent=None):
        super().__init__(t0, t1)
        self.behind = behind
        self.lines = list(lines)
        self.accent = accent
        self._scrim = None
        self.sfx = (("impact", t0 + 0.05), ("whoosh", t0 - 0.15), ("pop", t0 + 0.42))
        if behind:
            size = 660 if len(behind) <= 4 else int(660 * 4 / len(behind))
            self.s_back = text_sprite(behind.upper(), "display", size, gradient=GOLD_GRAD, stroke=0,
                                      shadow=(10, 14, (6, 14, 24)), tracking=6)
            self.s_back = self._fit(self.s_back, W * 0.96)
        self.s_lines = []
        for i, ln in enumerate(self.lines):
            marker = self.accent == i
            self.s_lines.append(text_sprite(ln.upper(), "display", 124 if len(ln) < 15 else 100,
                                            fill=INK if marker else IVORY,
                                            shadow=None if marker else (7, 9, INK), tracking=2))
            self.s_lines[-1] = self._fit(self.s_lines[-1], W * 0.92)

    @staticmethod
    def _fit(sprite, max_w):
        h, w = sprite.shape[:2]
        if w <= max_w:
            return sprite
        k = max_w / w
        return cv2.resize(sprite, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)

    def back(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.55, 0.35)
        k = ease_out_cubic(a_in)
        if not getattr(self, "talk", True):           # over footage there is no subject: darken the top for legibility
            if self._scrim is None:
                ys = np.linspace(1.0, 0.0, H)[:, None, None] ** 1.5
                self._scrim = np.clip(ys * (np.linspace(1, 0, H)[:, None, None] > 0.55), 0, 1).astype(np.float32)
                self._scrim = np.clip((np.linspace(1.0, 0.0, H)[:, None, None] - 0.55) / 0.45, 0, 1).astype(np.float32)
            canvas *= 1 - 0.6 * k * ease_in_out(a_out) * self._scrim
        if not self.behind:
            return
        drift = 1.0 + 0.035 * ((t - self.t0) / max(self.t1 - self.t0, 0.1))
        scale = (1.22 - 0.22 * k) * drift * (0.92 + 0.08 * ease_in_out(a_out))
        blit(canvas, self.s_back, W / 2, H * 0.175, scale=scale, alpha=k * ease_in_out(a_out) * 0.97)

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0 + 0.18, self.t1, 0.5, 0.3)
        y0 = H * (0.50 if self.behind else 0.115)
        for i, s in enumerate(self.s_lines):
            d = i * 0.14
            p = spring((t - self.t0 - 0.2 - d) / 0.5)
            out = ease_in_out(a_out)
            if p <= 0 or out <= 0:
                continue
            h, w = s.shape[:2]
            y = y0 + i * 168
            if self.accent == i:           # gold marker wiping in behind the text
                wipe = ease_out_cubic((t - self.t0 - 0.28 - d) / 0.32)
                bar = rect_sprite(w * 0.97, h * 0.80, GOLD)
                shade = rect_sprite(w * 0.97, h * 0.80, INK)
                blit(canvas, shade, W / 2 - w * 0.485 * (1 - wipe) + 9, y + 11, sx=wipe, sy=1.0, rot=-1.4, alpha=out)
                blit(canvas, bar, W / 2 - w * 0.485 * (1 - wipe), y, sx=wipe, sy=1.0, rot=-1.4, alpha=out,
                     anchor=(0.0 if False else 0.5, 0.5))
            blit(canvas, s, W / 2, y + (1 - min(p, 1.0)) * 40, scale=0.86 + 0.14 * min(p, 1.25), rot=-1.4 if self.accent == i else 0.6,
                 alpha=min(1.0, p * 1.6) * out)


class Stat(Element):
    """A flat gold sticker with a small label and a big figure, printed-looking: hard shadow, slight tilt."""

    def __init__(self, t0, t1, kicker="", value="", sub="", side="left", y=0.33, tilt=None):
        super().__init__(t0, t1)
        self.side, self.y = side, y
        self.tilt = tilt if tilt is not None else (-4.0 if side == "left" else 3.5)
        self.sfx = (("pop", t0 + 0.05), ("whoosh", t0 - 0.1))
        k = text_sprite(kicker.upper(), "sans", 36, fill=INK, tracking=3) if kicker else None
        v = text_sprite(value, "display", 150, fill=INK, tracking=1)
        s = text_sprite(sub.upper(), "sans", 34, fill=INK, tracking=2) if sub else None
        parts = [p for p in (k, v, s) if p is not None]
        wmax = max(p.shape[1] for p in parts) + 70
        hsum = sum(p.shape[0] for p in parts) * 0.86 + 40
        card = np.zeros((int(hsum) + 24, int(wmax) + 24, 4), np.float32)
        # hard shadow then card
        card[14:14 + int(hsum), 14:14 + int(wmax)] = rect_sprite(wmax, hsum, INK)
        card[0:int(hsum), 0:int(wmax)] = rect_sprite(wmax, hsum, GOLD)
        yy = 20
        for p in parts:
            ph, pw = p.shape[:2]
            x = int((wmax - pw) / 2)
            roi = card[yy:yy + ph, x:x + pw]
            a = p[..., 3:4]
            roi[..., :3] = p[..., :3] + roi[..., :3] * (1 - a)
            roi[..., 3:4] = a + roi[..., 3:4] * (1 - a)
            yy += int(ph * 0.86)
        self.card = card

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.5, 0.25)
        p = spring(a_in)
        out = ease_in_out(a_out)
        if p <= 0 or out <= 0:
            return
        x = W * (0.20 if self.side == "left" else 0.80)
        base = 0.82
        cw = self.card.shape[1]
        head = getattr(self, "avoid", None)          # (left, right) of his head in this frame, when he is cut out
        if head is not None:
            gap, edge = 70, 22                                  # a tilted card's corners reach further than its width
            room = (head[0] if self.side == "left" else W - head[1]) - gap - edge
            base = max(0.55, min(base, room / cw))              # shrink to fit beside him, but not below 55%
            half = cw * base / 2
            x = edge + half if self.side == "left" else W - edge - half     # live in the margin, away from him
        bob = math.sin((t - self.t0) * 1.6) * 5
        blit(canvas, self.card, x, H * self.y + bob, scale=base * (0.5 + 0.5 * min(p, 1.2)) * (0.9 + 0.1 * out),
             rot=self.tilt * (0.6 + 0.4 * min(p, 1.0)), alpha=min(1.0, p * 2) * out)


class Tag(Element):
    """A short statement on an ink bar with a gold underline that draws itself."""

    def __init__(self, t0, t1, text="", y=0.215, accent=""):
        super().__init__(t0, t1)
        self.y = y
        self.sfx = (("whoosh", t0 - 0.05), ("pop", t0 + 0.18))
        self.s = text_sprite(text.upper(), "display", 92, fill=IVORY, tracking=2)
        w = self.s.shape[1]
        if w > W * 0.9:
            self.s = cv2.resize(self.s, (int(W * 0.9), int(self.s.shape[0] * W * 0.9 / w)), interpolation=cv2.INTER_AREA)
        h, w = self.s.shape[:2]
        self.bar = rect_sprite(w + 60, h * 0.86, INK, 0.9)
        self.w, self.h = w, h

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.4, 0.25)
        p = ease_out_cubic(a_in)
        out = ease_in_out(a_out)
        if p <= 0 or out <= 0:
            return
        cx, cy = W / 2, H * self.y
        blit(canvas, self.bar, cx, cy, sx=p, sy=1.0, alpha=out)
        blit(canvas, self.s, cx, cy + (1 - p) * 30, scale=1.0, alpha=min(1.0, p * 1.5) * out)
        u = rect_sprite((self.w + 60) * 0.5, 7, GOLD)
        wipe = ease_out_cubic((t - self.t0 - 0.2) / 0.45)
        blit(canvas, u, cx - (self.w + 60) * 0.25 * (1 - wipe), cy + self.h * 0.5, sx=wipe, sy=1.0, alpha=out)


class Cta(Element):
    """End card: darkens the frame, gold question in huge type, a quiet line below."""

    def __init__(self, t0, t1, text="", sub=""):
        super().__init__(t0, t1)
        self.sfx = (("impact", t0 + 0.1), ("whoosh", t0 - 0.1))
        self.s = text_sprite(text.upper(), "display", 190, gradient=GOLD_GRAD, shadow=(9, 12, INK), tracking=3)
        w = self.s.shape[1]
        if w > W * 0.94:
            self.s = cv2.resize(self.s, (int(W * 0.94), int(self.s.shape[0] * W * 0.94 / w)), interpolation=cv2.INTER_AREA)
        self.sub = text_sprite(sub, "sans", 50, fill=IVORY, tracking=1, shadow=(4, 5, INK)) if sub else None

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.5, 0.01)
        k = ease_out_cubic(a_in)
        canvas *= 1 - 0.52 * k
        p = spring((t - self.t0 - 0.1) / 0.55)
        blit(canvas, self.s, W / 2, H * 0.44 + (1 - min(p, 1)) * 60, scale=0.8 + 0.2 * min(p, 1.2), alpha=min(1.0, p * 1.5))
        if self.sub is not None:
            q = ease_out_cubic((t - self.t0 - 0.45) / 0.5)
            blit(canvas, self.sub, W / 2, H * 0.44 + 190 + (1 - q) * 30, alpha=q)


class Interlude(Element):
    """Words on the animated background while the picture was unusable (the camera was being moved):
    big type in the hook style, centred, with a gold marker line behind one of the lines."""

    def __init__(self, t0, t1, lines=(), accent=None):
        super().__init__(t0, t1)
        self.accent = accent
        self.sfx = (("whoosh", t0 - 0.05), ("pop", t0 + 0.3))
        self.s_lines = []
        for i, ln in enumerate(lines):
            marker = accent == i
            sp = text_sprite(ln.upper(), "display", 190 if len(ln) < 12 else 140, fill=INK if marker else IVORY,
                             shadow=None if marker else (8, 10, INK), tracking=3)
            h, w = sp.shape[:2]
            if w > W * 0.9:
                sp = cv2.resize(sp, (int(W * 0.9), int(h * W * 0.9 / w)), interpolation=cv2.INTER_AREA)
            self.s_lines.append(sp)

    def front(self, canvas, t):
        a_in, a_out = _life(t, self.t0, self.t1, 0.45, 0.3)
        out = ease_in_out(a_out)
        if out <= 0:
            return
        y0 = H * 0.36
        for i, sp in enumerate(self.s_lines):
            p = spring((t - self.t0 - 0.08 * i) / 0.5)
            if p <= 0:
                continue
            h, w = sp.shape[:2]
            y = y0 + i * 230
            if self.accent == i:
                wipe = ease_out_cubic((t - self.t0 - 0.2 - 0.08 * i) / 0.32)
                bar = rect_sprite(w * 1.04, h * 0.82, GOLD)
                shade = rect_sprite(w * 1.04, h * 0.82, INK)
                blit(canvas, shade, W / 2 - w * 0.52 * (1 - wipe) + 10, y + 12, sx=wipe, sy=1.0, rot=-1.4, alpha=out)
                blit(canvas, bar, W / 2 - w * 0.52 * (1 - wipe), y, sx=wipe, sy=1.0, rot=-1.4, alpha=out)
            blit(canvas, sp, W / 2, y + (1 - min(p, 1.0)) * 50, scale=0.85 + 0.15 * min(p, 1.25),
                 rot=-1.4 if self.accent == i else 0.6, alpha=min(1.0, p * 1.6) * out)


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
            out.append(Tag(t0, min(duration - 0.1, t0 + float(s.get("dur", 2.6))), s["text"], s.get("y", 0.215)))
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
    """English captions, drawn like everything else: Montserrat ExtraBold, white, dark outline."""

    def __init__(self, events, size=60, y=0.745):
        self.events, self.size, self.y = events, size, y
        self.cache = {}

    def _sprite(self, text):
        if text in self.cache:
            return self.cache[text]
        parts = [text_sprite(line, "sans", self.size, fill=IVORY if False else (250, 250, 250), stroke=5,
                             stroke_fill=(6, 6, 8), shadow=(3, 5, (0, 0, 0)), tracking=0.4)
                 for line in text.split(chr(92) + "N")]
        lh = int(self.size * 1.18)
        wmax = max(p.shape[1] for p in parts)
        out = np.zeros((lh * len(parts) + 30, wmax, 4), np.float32)
        for i, p in enumerate(parts):
            ph, pw = p.shape[:2]
            x, y = (wmax - pw) // 2, i * lh
            roi = out[y:y + ph, x:x + pw]
            a = p[..., 3:4]
            h2 = min(ph, out.shape[0] - y)
            roi[:h2, :, :3] = p[:h2, :, :3] + roi[:h2, :, :3] * (1 - a[:h2])
            roi[:h2, :, 3:4] = a[:h2] + roi[:h2, :, 3:4] * (1 - a[:h2])
        self.cache[text] = out
        return out

    def draw(self, canvas, t):
        for a, b, text in self.events:
            if a <= t <= b:
                k = ease_out_cubic((t - a) / 0.12)
                out = ease_in_out((b - t) / 0.08)
                spr = self._sprite(text)
                blit(canvas, spr, W / 2, H * self.y + (1 - k) * 16, alpha=min(1.0, k * 1.4) * out)
                return
