"""Motion-graphics toolkit: easing, text sprites, sprite blitting, grade, grain, vignette.

Everything works on float32 BGR canvases (0..255) and premultiplied BGRA sprites."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .util import REPO

FONTS = REPO / "assets" / "fonts"
SS = 2  # text is drawn at 2x then reduced, for clean edges

# warm editorial palette (BGR): ink, ivory, gold family. Red/green are for up/down only.
INK = (8, 8, 10)
IVORY = (226, 239, 246)
GOLD = (60, 194, 255)
GOLD_HI = (138, 224, 255)
GOLD_LO = (20, 120, 210)
BRASS = (91, 161, 201)          # #C9A15B, the one accent
BRASS_HI = (140, 196, 232)
IVORY2 = (238, 246, 250)
MUTED = (196, 188, 170)
UP = (110, 200, 70)
DOWN = (60, 70, 225)


# ------------------------------------------------------------------ easing

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def smooth01(x):
    """smoothstep of an array already scaled so 0..1 is the ramp."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.9):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def spring(x, damp=5.0, freq=11.0):
    """0 -> 1 with a couple of decaying overshoots (like a physical pop)."""
    if x <= 0:
        return 0.0
    if x >= 1.6:
        return 1.0
    return 1 - math.exp(-damp * x) * math.cos(freq * x)


# ------------------------------------------------------------------ fonts & text

@lru_cache(maxsize=64)
def _font(name, size):
    return ImageFont.truetype(str(FONTS / name), int(size))


FONT_SETS = {
    "display": ("anton-latin-400-normal.ttf", "anton-latin-ext-400-normal.ttf"),
    "sans": ("montserrat-latin-800-normal.ttf", "montserrat-latin-ext-800-normal.ttf"),
    "black": ("montserrat-latin-900-normal.ttf", "montserrat-latin-ext-800-normal.ttf"),
    "serif": ("playfair-display-latin-800-italic.ttf", "playfair-display-latin-800-italic.ttf"),
}
EXT_ONLY = "₹"  # rupee sign lives in the latin-ext files


def _runs(text):
    out, cur, ext = [], "", False
    for ch in text:
        e = ch in EXT_ONLY
        if cur and e != ext:
            out.append((cur, ext))
            cur = ""
        cur, ext = cur + ch, e
    if cur:
        out.append((cur, ext))
    return out


def text_size(text, face, size, tracking=0):
    f_main, f_ext = FONT_SETS[face]
    w = 0
    for run, ext in _runs(text):
        font = _font(f_ext if ext else f_main, size * SS)
        w += font.getlength(run) + tracking * SS * len(run)
    return w / SS, size * 1.25


def gradient_fill(h, w, stops):
    """Vertical gradient image (h, w, 3) BGR from [(pos, color), ...]."""
    ys = np.linspace(0, 1, h)[:, None]
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(ys, [p for p, _ in stops], [col[c] for _, col in stops])
    return out


def text_sprite(text, face="display", size=120, fill=IVORY, gradient=None, stroke=0, stroke_fill=INK,
                shadow=None, tracking=0, pad=None):
    """Premultiplied BGRA float32 sprite (h, w, 4), alpha 0..1, drawn crisp at 2x then reduced.
    shadow = (dx, dy, color) draws a hard, flat offset shadow (no blur: it reads as printed, not AI-soft)."""
    f_main, f_ext = FONT_SETS[face]
    pad = pad if pad is not None else int(size * 0.18 + stroke + (max(abs(shadow[0]), abs(shadow[1])) if shadow else 0))
    w, h = text_size(text, face, size, tracking)
    W, H = int(w + 2 * pad), int(h + 2 * pad)
    mask = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(mask)
    x = pad * SS
    for run, ext in _runs(text):
        font = _font(f_ext if ext else f_main, size * SS)
        if tracking:
            for ch in run:
                d.text((x, pad * SS), ch, font=font, fill=255)
                x += font.getlength(ch) + tracking * SS
        else:
            d.text((x, pad * SS), run, font=font, fill=255)
            x += font.getlength(run)
    mask = np.asarray(mask.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    layers = np.zeros((H, W, 4), np.float32)           # BGR (premultiplied) + A

    def over(color_img, a):
        a3 = a[..., None]
        layers[..., :3] = color_img * a3 + layers[..., :3] * (1 - a3)
        layers[..., 3] = a + layers[..., 3] * (1 - a)

    if shadow:
        dx, dy, col = shadow
        sh = np.roll(np.roll(mask, int(dy), 0), int(dx), 1)
        if stroke:
            sh = np.roll(np.roll(cv2.dilate(mask, np.ones((stroke * 2 + 1,) * 2, np.uint8)), int(dy), 0), int(dx), 1)
        over(np.full((H, W, 3), col, np.float32), sh)
    if stroke:
        big = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke * 2 + 1,) * 2))
        over(np.full((H, W, 3), stroke_fill, np.float32), big)
    if gradient:
        ys, ye = np.where(mask.max(axis=1) > 0.1)[0][[0, -1]] if mask.max() > 0.1 else (0, H - 1)
        col = np.zeros((H, W, 3), np.float32)
        col[:] = fill
        gh = max(1, ye - ys + 1)
        col[ys:ye + 1] = gradient_fill(gh, W, gradient)
        over(col, mask)
    else:
        over(np.full((H, W, 3), fill, np.float32), mask)
    return layers


def rect_sprite(w, h, color, alpha=1.0):
    s = np.zeros((int(h), int(w), 4), np.float32)
    s[..., :3] = np.array(color, np.float32) * alpha
    s[..., 3] = alpha
    return s


# ------------------------------------------------------------------ blitting

def blit(canvas, sprite, cx, cy, scale=1.0, rot=0.0, alpha=1.0, sx=None, sy=None, anchor=(0.5, 0.5)):
    """Composite a premultiplied BGRA sprite onto a BGR float canvas, scaled and rotated about an anchor."""
    if alpha <= 0.003 or scale <= 0.001:
        return
    sx = scale if sx is None else sx
    sy = scale if sy is None else sy
    if abs(sx) < 1e-3 or abs(sy) < 1e-3:       # a wipe that has not started yet
        return
    h, w = sprite.shape[:2]
    ax, ay = anchor[0] * w, anchor[1] * h
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    # destination bounding box
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) - [ax, ay]
    corners = corners * [sx, sy]
    rc = np.stack([corners[:, 0] * c - corners[:, 1] * s, corners[:, 0] * s + corners[:, 1] * c], 1) + [cx, cy]
    x0, y0 = np.floor(rc.min(0)).astype(int) - 2
    x1, y1 = np.ceil(rc.max(0)).astype(int) + 2
    H, W = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    # affine: dest(px) -> sprite(px)
    m = np.array([[c / sx, s / sx], [-s / sy, c / sy]], np.float32)
    off = np.array([ax, ay], np.float32) - m @ np.array([cx, cy], np.float32)
    M = np.hstack([m, off[:, None]])
    M[:, 2] += m @ np.array([X0, Y0], np.float32)
    warped = cv2.warpAffine(sprite, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4] * alpha
    roi = canvas[Y0:Y1, X0:X1]
    roi *= (1 - a)
    roi += warped[..., :3] * alpha


def add_glow(canvas, layer, strength=0.6, sigma=8):
    canvas += cv2.GaussianBlur(layer, (0, 0), sigma) * strength


# ------------------------------------------------------------------ post: grade, vignette, grain

def vignette(w, h, strength=0.38):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h * 0.46) / (h / 2)) ** 2)
    return (1 - strength * np.clip(r - 0.35, 0, 1) ** 1.6).astype(np.float32)[..., None]


class Grain:
    """Pre-rendered film grain tiles, cycled per frame so it moves like real stock."""

    def __init__(self, w, h, n=6, amount=7.0, seed=3):
        rng = np.random.default_rng(seed)
        self.tiles = []
        for _ in range(n):
            g = rng.normal(0, 1, (h // 2, w // 2)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.7)
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
            self.tiles.append((g * amount)[..., None])

    def __call__(self, canvas, frame_no):
        canvas += self.tiles[frame_no % len(self.tiles)]


def grade(img, contrast=1.10, saturation=1.06, warmth=0.035, lift=2.0):
    """Gentle S-curve, a touch of colour and warmth. img float32 BGR 0..255, in place."""
    x = img / 255.0
    x = x + contrast * 0.0 + (x - 0.5) * (contrast - 1.0) * 1.6 * (1 - np.abs(2 * x - 1) ** 2) * 1.0
    x = np.clip(x, 0, 1)
    luma = (x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299)[..., None]
    x = luma + (x - luma) * saturation
    x[..., 2] *= 1 + warmth
    x[..., 0] *= 1 - warmth * 0.9
    img[:] = np.clip(x * 255.0 + lift, 0, 255)
    return img


def finish(img, halation=0.05, saturation=0.94):
    """The last look over the whole composite (in place): shadows lean cool, highlights warm, a touch less
    colour so the yellow hoodie does not shout, and a faint warm bloom around the brightest areas."""
    x = img / 255.0
    luma = (x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299)
    sh = 1 - smooth01(luma / 0.45)
    hi = smooth01((luma - 0.55) / 0.45)
    x[..., 0] += 0.010 * sh - 0.007 * hi
    x[..., 2] += -0.003 * sh + 0.009 * hi
    l3 = luma[..., None]
    x = l3 + (x - l3) * saturation
    if halation > 0:
        bright = np.clip(luma - 0.72, 0, 1)
        h, w = bright.shape
        sm = cv2.resize(bright, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3.0), (w, h), interpolation=cv2.INTER_LINEAR)
        x += sm[..., None] * np.array((0.55, 0.75, 1.0), np.float32) * halation * 3.0
    img[:] = np.clip(x * 255.0, 0, 255)
    return img


# ------------------------------------------------------------------ premium kit: halo text, frosted panels

def with_halo(sprite, blur=9, a=0.55):
    """Premultiplied BGRA sprite with a soft dark halo behind it (legible without an outline)."""
    pad = int(blur * 3)
    h, w = sprite.shape[:2]
    big = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    big[pad:pad + h, pad:pad + w] = sprite
    halo = cv2.GaussianBlur(big[..., 3], (0, 0), blur) * a
    big[..., 3] = big[..., 3] + halo * (1 - big[..., 3])
    return big


class Panel:
    """A label: solid near-black fill, a brass hairline and a soft shadow, small corner radius (printed, not UI).
    `frost=True` gives the frosted-glass variant instead (backdrop blurred and tinted); it reads as a UI card, so it is
    off by default. Draw text sprites over it yourself."""

    def __init__(self, w, h, radius=8, fill=(10, 10, 11), fill_a=0.92, edge=BRASS, edge_a=0.60, shadow=0.38, pad=44,
                 frost=False, dark=0.46, blur=14, tint=(24, 16, 10)):
        self.w, self.h, self.pad, self.blur, self.dark, self.frost = int(w), int(h), pad, blur, dark, frost
        self.tint = np.array(tint, np.float32)
        S = 2
        W_, H_ = self.w + 2 * pad, self.h + 2 * pad
        m = Image.new("L", (W_ * S, H_ * S), 0)
        ImageDraw.Draw(m).rounded_rectangle([pad * S, pad * S, (pad + self.w) * S - 1, (pad + self.h) * S - 1], radius=radius * S, fill=255)
        full = np.asarray(m.resize((W_, H_), Image.LANCZOS), np.float32) / 255.0
        self.mask = full[pad:pad + self.h, pad:pad + self.w].copy()
        ring = np.clip(full - cv2.erode(full, np.ones((3, 3), np.uint8)), 0, 1)
        sh = cv2.GaussianBlur(np.roll(full, 12, axis=0), (0, 0), 20) * shadow * (1 - full)
        sp = np.zeros((H_, W_, 4), np.float32)
        if not frost:
            pa = full * fill_a
            sp[..., :3] = np.array(fill, np.float32) * pa[..., None]
            sp[..., 3] = pa + sh * (1 - pa)
        else:
            sp[..., 3] = sh
        ea = ring * edge_a
        sp[..., :3] = np.array(edge, np.float32) * ea[..., None] + sp[..., :3] * (1 - ea[..., None])
        sp[..., 3] = ea + sp[..., 3] * (1 - ea)
        self.sprite = sp

    def draw(self, canvas, cx, cy, alpha=1.0):
        """(Frosted: backdrop blur + tint inside the panel,) then the label with its rim and shadow."""
        x0, y0 = int(round(cx - self.w / 2)), int(round(cy - self.h / 2))
        if self.frost:
            H_, W_ = canvas.shape[:2]
            X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + self.w, W_), min(y0 + self.h, H_)
            if X1 > X0 and Y1 > Y0 and alpha > 0.003:
                roi = canvas[Y0:Y1, X0:X1]
                m = self.mask[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None] * alpha
                glass = cv2.GaussianBlur(roi, (0, 0), self.blur) * (1 - self.dark) + self.tint * self.dark
                roi[:] = roi * (1 - m) + glass * m
        blit(canvas, self.sprite, x0 + self.w / 2, y0 + self.h / 2, alpha=alpha)
        return x0, y0


def text_halo(text, face="sans", size=60, fill=IVORY2, tracking=0, blur=9, a=0.5):
    return with_halo(text_sprite(text, face, size, fill=fill, tracking=tracking), blur, a)


def unsharp(img, amount=0.55, sigma=1.4):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + (img - blur) * amount, 0, 255)
