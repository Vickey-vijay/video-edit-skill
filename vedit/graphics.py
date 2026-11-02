"""Brand-styled stills drawn with Pillow: the frame around the video panels
(background grid, panel borders, labels, title) and B-roll concept cards."""
import textwrap
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .util import REPO

SS = 2  # supersampling for smooth edges


def rgba(h, alpha=None):
    h = h.lstrip("#")
    if len(h) == 6:
        h += "FF"
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
    return c if alpha is None else c[:3] + (alpha,)


class Brand:
    def __init__(self, brand):
        self.b = brand
        self.fontdir = REPO / "assets" / "fonts"

    def font(self, key, size):
        return ImageFont.truetype(str(self.fontdir / self.b["fonts"][key]["file"]), int(size))

    def c(self, key, alpha=None):
        return rgba(self.b["colors"][key], alpha)


def background(W, H, br):
    img = Image.new("RGBA", (W, H), br.c("bg"))
    glow = Image.new("RGBA", (W // 4, H // 4), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    r = max(W, H) // 4
    gd.ellipse([-r * 0.7, -r * 0.8, r * 0.8, r * 0.55], fill=br.c("accent_soft", 170))
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.22)).resize((W, H), Image.BICUBIC)
    img = Image.alpha_composite(img, glow)
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(grid)
    step = br.b.get("grid_step", 48)
    for x in range(0, W, step):
        d.line([(x, 0), (x, H)], fill=br.c("grid"), width=1)
    for y in range(0, H, step):
        d.line([(0, y), (W, y)], fill=br.c("grid"), width=1)
    return Image.alpha_composite(img, grid)


def _rr(d, rect, r, **kw):
    x, y, w, h = rect
    d.rounded_rectangle([x * SS, y * SS, (x + w) * SS - 1, (y + h) * SS - 1], radius=r * SS, **kw)


def _down(img, W, H):
    return img.resize((W, H), Image.LANCZOS)


def spaced(text, n=2):
    return (" " * (n - 1)).join(text) if n > 1 else text


def frame_overlay(W, H, br, geo, title, series, name):
    """Opaque background with transparent rounded holes where the videos show through."""
    r = br.b.get("radius", 22)
    panels = [p for p in (geo.get("content"), geo.get("face")) if p]
    img = background(W, H, br)

    shadow = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for p in panels:
        _rr(sd, (p[0], p[1] + 12, p[2], p[3]), r, fill=(0, 0, 0, 150))
    shadow = _down(shadow, W, H).filter(ImageFilter.GaussianBlur(22))
    img = Image.alpha_composite(img, shadow)

    holes = Image.new("L", (W * SS, H * SS), 255)
    hd = ImageDraw.Draw(holes)
    for p in panels:
        _rr(hd, p, r, fill=0)
    img.putalpha(ImageChops.multiply(img.getchannel("A"), _down(holes, W, H)))

    lines = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for p in panels:
        _rr(ld, p, r, outline=br.c("rule"), width=2 * SS)
    if geo.get("face"):
        fx, fy, fw, fh = geo["face"]
        _rr(ld, (fx, fy, fw, fh), r, outline=br.c("accent", 110), width=2 * SS)
    img = Image.alpha_composite(img, _down(lines, W, H))

    d = ImageDraw.Draw(img)
    if geo.get("face") and name:
        fx, fy, fw, fh = geo["face"]
        s = geo["label_size"]
        # dark fade at the bottom of the facecam: hides the meeting app's own name tag
        scrim_h = int(fh * 0.34)
        grad = Image.linear_gradient("L").resize((fw, scrim_h)).point(lambda v: int(v * 0.86))
        scrim = Image.new("RGBA", (fw, scrim_h), br.c("bg"))
        scrim.putalpha(grad)
        smask = Image.new("L", (W * SS, H * SS), 0)
        _rr(ImageDraw.Draw(smask), geo["face"], br.b.get("radius", 22), fill=255)
        smask = _down(smask, W, H).crop((fx, fy + fh - scrim_h, fx + fw, fy + fh))
        scrim.putalpha(ImageChops.multiply(scrim.getchannel("A"), smask))
        img.alpha_composite(scrim, (fx, fy + fh - scrim_h))
        d = ImageDraw.Draw(img)
        f = br.font("body", s)
        tw = d.textlength(name, font=f)
        pad = s * 0.6
        # bottom-left, over the meeting app's own name label
        chip = (fx + 10, fy + fh - 10 - s * 1.9, max(tw + pad * 2 + s * 0.9, fw * 0.36), s * 1.9)
        cl = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cl)
        _rr(cd, chip, int(s * 0.95), fill=br.c("bg", 215), outline=br.c("rule"), width=SS)
        dot = s * 0.32
        cx, cy = chip[0] + pad + dot, chip[1] + chip[3] / 2
        cd.ellipse([(cx - dot) * SS, (cy - dot) * SS, (cx + dot) * SS, (cy + dot) * SS], fill=br.c("up"))
        img = Image.alpha_composite(img, _down(cl, W, H))
        d = ImageDraw.Draw(img)
        d.text((cx + dot + s * 0.5, cy), name, font=f, fill=br.c("fg"), anchor="lm")

    tb = geo.get("title_box")
    if tb and title:
        x, y, w = tb["x"], tb["y"], tb["w"]
        ls = tb["series_size"]
        d.text((x, y), spaced(series.upper(), 1), font=br.font("mono", ls), fill=br.c("accent"))
        ts = tb["title_size"]
        tf = br.font("display", ts)
        chars = max(8, int(w / (ts * 0.52)))
        ty = y + ls * 1.9
        for line in textwrap.wrap(title, chars)[:tb.get("max_lines", 3)]:
            d.text((x, ty), line, font=tf, fill=br.c("fg"))
            ty += ts * 1.12
        d.rectangle([x, ty + ts * 0.25, x + ts * 1.4, ty + ts * 0.25 + max(3, ts // 12)], fill=br.c("accent"))
    return img


# ---------------------------------------------------------------- B-roll cards

def _card_base(br, W, H, label, title):
    # Opaque RGB canvas so translucent fills blend instead of punching holes;
    # the rounded corners become alpha in _card_done.
    img = Image.new("RGB", (W * SS, H * SS), br.c("paper")[:3])
    d = ImageDraw.Draw(img, "RGBA")
    _rr(d, (40, 28, 44, 5), 2, fill=br.c("accent"))
    d.text((40 * SS, 40 * SS), spaced(label.upper(), 1), font=br.font("mono", 20 * SS), fill=br.c("accent"))
    d.text((40 * SS, 72 * SS), title, font=br.font("display", 54 * SS), fill=br.c("fg"))
    return img, d


def _card_done(br, img, W, H):
    d = ImageDraw.Draw(img, "RGBA")
    _rr(d, (0, 0, W, H), 28, outline=br.c("rule"), width=2 * SS)
    mask = Image.new("L", img.size, 0)
    _rr(ImageDraw.Draw(mask), (0, 0, W, H), 28, fill=255)
    out = img.convert("RGBA")
    out.putalpha(mask)
    return _down(out, W, H)


def card_panel(br, card, w, h):
    """Card centred on a rounded panel of exactly w x h (used where the card
    stands in for the facecam)."""
    r = br.b.get("radius", 22)
    panel = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    _rr(ImageDraw.Draw(panel), (0, 0, w, h), r, fill=br.c("paper"), outline=br.c("rule"), width=2 * SS)
    panel = _down(panel, w, h)
    k = min((w - 24) / card.width, (h - 24) / card.height)
    c = card.resize((int(card.width * k), int(card.height * k)), Image.LANCZOS)
    panel.alpha_composite(c, ((w - c.width) // 2, (h - c.height) // 2))
    return panel


def _chips(br, d, chips, W, H):
    f = br.font("mono", 21 * SS)
    x, y = 40 * SS, (H - 66) * SS
    for text in chips:
        tw = d.textlength(text, font=f)
        if x + tw + 36 * SS > (W - 30) * SS:
            break
        d.rounded_rectangle([x, y, x + tw + 32 * SS, y + 42 * SS], radius=21 * SS,
                            fill=br.c("code_bg"), outline=br.c("rule"), width=SS)
        d.text((x + 16 * SS, y + 21 * SS), text, font=f, fill=br.c("fg2"), anchor="lm")
        x += tw + 44 * SS


def _diagram_box(W, H):
    return (60, 150, W - 120, H - 150 - 96)


def _plot(points, box, pad=0.08):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bx, by, bw, bh = box
    sx = bw * (1 - 2 * pad) / max(x1 - x0, 1e-6)
    sy = bh * (1 - 2 * pad) / max(y1 - y0, 1e-6)
    return [((bx + bw * pad + (x - x0) * sx) * SS, (by + bh * pad + (y1 - y) * sy) * SS) for x, y in points]


def card_pattern(br, item, W=880, H=640):
    img, d = _card_base(br, W, H, item.get("label", "Harmonic pattern"), item["title"])
    names = item.get("names", ["X", "A", "B", "C", "D"])
    pts = _plot(item["points"], _diagram_box(W, H))
    soft = br.c("accent", 70)
    if len(pts) == 5:
        d.polygon([pts[0], pts[1], pts[2]], fill=soft)
        d.polygon([pts[2], pts[3], pts[4]], fill=soft)
        for a, b in ((0, 2), (2, 4), (1, 3), (0, 4)):
            _dashed(d, pts[a], pts[b], br.c("muted", 150), 2 * SS)
    d.line(pts, fill=br.c("accent"), width=6 * SS, joint="curve")
    f = br.font("display", 30 * SS)
    for (x, y), n, (_, vy) in zip(pts, names, item["points"]):
        d.ellipse([x - 9 * SS, y - 9 * SS, x + 9 * SS, y + 9 * SS], fill=br.c("fg"), outline=br.c("bg"), width=3 * SS)
        up = vy >= sorted(p[1] for p in item["points"])[len(item["points"]) // 2]
        d.text((x, y + (-34 if up else 34) * SS), n, font=f, fill=br.c("fg"), anchor="mm")
    _chips(br, d, item.get("chips", []), W, H)
    return _card_done(br, img, W, H)


def card_fib(br, item, W=880, H=640):
    img, d = _card_base(br, W, H, item.get("label", "Fibonacci"), item["title"])
    bx, by, bw, bh = _diagram_box(W, H)
    levels = item.get("levels", [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1])
    f = br.font("mono", 22 * SS)
    key = item.get("key", 0.618)
    for lv in levels:
        y = (by + bh * 0.06 + lv * bh * 0.88) * SS
        hot = abs(lv - key) < 1e-6
        col = br.c("highlight") if hot else br.c("muted", 200)
        d.line([((bx + 150) * SS, y), ((bx + bw) * SS, y)], fill=col, width=(5 if hot else 2) * SS)
        d.text(((bx + 130) * SS, y), f"{lv:.3f}".rstrip("0").rstrip(".") if lv not in (0, 1) else str(lv),
               font=f, fill=col, anchor="rm")
    swing = [((bx + 190) * SS, (by + bh * 0.94) * SS), ((bx + bw * 0.55) * SS, (by + bh * 0.06) * SS),
             ((bx + bw * 0.86) * SS, (by + bh * (0.06 + key * 0.88)) * SS)]
    d.line(swing, fill=br.c("accent"), width=6 * SS, joint="curve")
    _chips(br, d, item.get("chips", []), W, H)
    return _card_done(br, img, W, H)


def card_waves(br, item, W=880, H=640):
    img, d = _card_base(br, W, H, item.get("label", "Elliott wave"), item["title"])
    pts = _plot(item["points"], _diagram_box(W, H))
    cols = br.b["colors"]["degrees"]
    for k in range(len(pts) - 1):
        d.line([pts[k], pts[k + 1]], fill=rgba(cols[k % len(cols)]), width=6 * SS)
    f = br.font("display", 30 * SS)
    for k, ((x, y), n) in enumerate(zip(pts, item.get("names", []))):
        if not n:
            continue
        up = k % 2 == 1 if item.get("up_odd", True) else k % 2 == 0
        d.text((x, y + (-32 if up else 32) * SS), n, font=f, fill=br.c("fg"), anchor="mm")
    _chips(br, d, item.get("chips", []), W, H)
    return _card_done(br, img, W, H)


def _dashed(d, a, b, fill, width, dash=14 * SS):
    (x0, y0), (x1, y1) = a, b
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    n = max(1, int(L / dash))
    for i in range(0, n, 2):
        t0, t1 = i / n, min(1, (i + 1) / n)
        d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
               fill=fill, width=width)


CARD_KINDS = {"pattern": card_pattern, "fib": card_fib, "waves": card_waves}


def make_cards(brand_cfg, library, out_dir):
    br = Brand(brand_cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for item in library:
        kind = item.get("kind")
        if kind not in CARD_KINDS:
            continue
        path = out_dir / f"{item['id']}.png"
        CARD_KINDS[kind](br, item).save(path)
        made.append(path)
    return made
