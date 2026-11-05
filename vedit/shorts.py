"""Vertical short-form compositor (1080x1920 Reels / Shorts) for phone footage.

Per frame: background theme -> type behind him -> matted subject (edge-tightened, light-wrapped,
graded) -> stickers and hook in front -> vignette and film grain. Footage of a screen is
never matted (the chart would vanish); it gets the grade, a slow push-in and the same overlays."""
import math
import subprocess

import cv2
import numpy as np

from . import backgrounds, edge, fx, motion
from .plan import Timeline
from .util import log

W, H = 1080, 1920
BW, BH = 540, 960          # backgrounds are drawn at half size and upscaled: soft, like a lens


# ------------------------------------------------------------------ scenes

def classify_scenes(conform, alpha, info, fps):
    """Per source frame: True where he is on camera (matte covers a good share of the frame).
    Smoothed over about a second so a hand crossing the lens does not flip the scene."""
    w, h = info["width"], info["height"]
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(alpha), "-vf", f"scale={w // 8}:{h // 8}",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    sw, sh = w // 8, h // 8
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, sh, sw).astype(np.float32) / 255
    cover = a.mean(axis=(1, 2))
    top = a[:, : sh // 2, sw // 5: 4 * sw // 5].mean(axis=(1, 2))      # head and shoulders zone
    raw = (cover > 0.22) & (top > 0.18)
    k = max(3, int(fps) | 1)
    pad = np.pad(raw.astype(np.float32), (k // 2, k // 2), mode="edge")
    return (np.convolve(pad, np.ones(k) / k, mode="valid") > 0.5)


# ------------------------------------------------------------------ subject

def _soft(a, sigma, scale=8):
    """Gaussian blur of a one- or three-channel image, done at 1/scale size (cheap for big sigmas)."""
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / scale)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class Subject:
    """Turns a matted frame into a 1080x1920 layer, with zoom about the face, and sets him into the scene.

    `refined` layers come from the enhance pass (already sharp, matte snapped to the picture); the
    plain 480x848 originals take the older route (stretch, tighten the matte, sharpen). Either way
    the cut-out is finished the same way (see edge.py): a decisive matte, colour from inside, skin
    balance; then the integration: hair edge darkening, a little of the new background wrapped onto
    his edge, a rim from the key light, a soft shadow and a warm glow in the scene behind him."""

    def __init__(self, src_w, src_h, face=(0.5, 0.33), refined=False, cfg=None):
        self.sw, self.sh = src_w, src_h
        self.face = face
        self.refined = refined
        self.k = H / src_h                           # cover scale
        self.er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sc = (cfg or {}).get("short", {})
        self.ec = {"lo": 0.32, "hi": 0.88, "feather": 1.0, "wrap": 0.16, "wrap_hair": 0.07, "rim": 0.10, "rim_hair": 0.045,
                   "hair_dark": 0.15, "shadow": 0.30, "glow": 0.07, **sc.get("edge", {})}
        self.skin = {"enabled": True, "strength": 0.65, "lift": 6.0, "tame_yellow": 0.12, **sc.get("skin", {})}
        yy = np.arange(H, dtype=np.float32)
        self.head_w = (1 - edge.smooth(yy, 0.40 * H, 0.52 * H))[:, None]      # 1 around the head, 0 on the shoulders
        self.low_w = edge.smooth(yy, 0.40 * H, 0.62 * H)[:, None]

    def _scaled(self, img, zoom, cx, cy, up, cyfrac):
        k = self.k * zoom
        ow, oh = int(round(self.sw * k)), int(round(self.sh * k))
        x0 = int(np.clip(cx * ow - W / 2, 0, max(0, ow - W)))
        y0 = int(np.clip(cy * oh - H * cyfrac, 0, max(0, oh - H)))
        interp = cv2.INTER_AREA if k < 1 else up
        out = cv2.resize(img, (ow, oh), interpolation=interp)[y0:y0 + H, x0:x0 + W]
        if out.shape[0] != H or out.shape[1] != W:       # source narrower than the frame at zoom 1
            out = cv2.resize(out, (W, H))
        return out

    def layer(self, rgb, alpha, zoom, canvas, key=(-0.6, -0.8)):
        """Returns (colour, alpha) for compositing onto `canvas`; the canvas is darkened/lit in place for the
        shadow and glow behind him. `key` is the direction toward the key light (x right, y down)."""
        e = self.ec
        rgb_b = self._scaled(rgb, zoom, self.face[0], self.face[1], cv2.INTER_LANCZOS4, 0.36).astype(np.float32)
        a_b = self._scaled(alpha, zoom, self.face[0], self.face[1], cv2.INTER_CUBIC, 0.36).astype(np.float32) / 255
        rgb_b = fx.unsharp(rgb_b, 0.22 if self.refined else 0.6, 1.1 if self.refined else 1.3)
        rgb_b, a_b = edge.refine(rgb_b, a_b, e["lo"], e["hi"], e["feather"])
        if self.skin.get("enabled", True):
            edge.skin_balance(rgb_b, 0.16 * H, 0.56 * H, self.skin["strength"], lift=self.skin["lift"])
            edge.tame_yellow(rgb_b, 0.40 * H, self.skin["tame_yellow"])
        fx.grade(rgb_b, contrast=1.10, saturation=1.03, warmth=0.012)
        # the edge band: 0 inside, about 1 at the border
        eb = np.clip(2.4 * a_b * (1 - cv2.GaussianBlur(a_b, (0, 0), 4.0)), 0, 1)
        lum = rgb_b.mean(axis=2) / 255.0
        hair = (lum < 0.24).astype(np.float32) * self.head_w
        # 1. hair dissolves into the dark room instead of showing a grey rim
        rgb_b *= (1 - e["hair_dark"] * eb * hair)[..., None]
        # 2. light wrap: the room's own light on his edge (less on hair)
        wrap_src = _soft(canvas, 22, 4)
        rgb_b += wrap_src * (eb * (e["wrap"] * (1 - hair) + e["wrap_hair"] * hair))[..., None]
        # 3. rim from the key light (top left): where the edge faces the light
        g = cv2.GaussianBlur(a_b, (0, 0), 3.0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        facing = np.clip(-gx / mag * key[0] - gy / mag * key[1], 0, 1)
        rim = facing * eb * (e["rim"] * (1 - hair) + e["rim_hair"] * hair)
        rgb_b += rim[..., None] * np.array((154, 199, 232), np.float32)
        # 4. soft contact shadow, offset away from the key (weaker around the head: hair must stay lighter than the room)
        sh = cv2.warpAffine(_soft(a_b, 40), np.float32([[1, 0, 18], [0, 1, 24]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
        canvas *= (1 - e["shadow"] * sh * (0.45 + 0.55 * self.low_w))[..., None]
        # 5. warm glow behind the shoulders, never behind the head
        gl = _soft(a_b * self.low_w, 110, 8)
        canvas += gl[..., None] * np.array((70, 115, 170), np.float32) * (e["glow"] * 255 / 255.0) * 2.2
        return rgb_b, a_b[..., None]

    def footage(self, rgb, zoom, cx=0.5, cy=0.5):
        out = self._scaled(rgb, zoom, cx, cy, cv2.INTER_LANCZOS4, 0.5).astype(np.float32)
        out = fx.unsharp(out, 0.25 if self.refined else 0.5, 1.1 if self.refined else 1.3)
        return fx.grade(out, contrast=1.10, saturation=1.05, warmth=0.02)


# ------------------------------------------------------------------ renderer

class Shorts:
    def __init__(self, plan, cfg, wd, gfx, talk, face=(0.5, 0.33), caption_events=(), enh=None):
        self.plan, self.cfg, self.gfx = plan, cfg, gfx or {}
        info = plan["info"]
        self.fps = plan["fps"]
        self.tl = Timeline(plan["segments"], self.fps)
        self.duration = self.tl.duration
        self.talk = talk
        self.enh = enh                      # enhanced layers {rgb, alpha, w, h} or None (use the originals)
        if enh:
            self.subject = Subject(enh["w"], enh["h"], face, refined=True, cfg=cfg)
        else:
            self.subject = Subject(info["width"], info["height"], face, cfg=cfg)
        self.elements = motion.build(self.gfx, self.tl, self.duration)
        self.captions = motion.Captions(caption_events)
        self.sched = self.gfx.get("themes", [])
        self.default_theme = self.gfx.get("theme", "studio")
        self._bg, self._last = {}, (None, -9.0)
        sc = cfg.get("short", {})
        self.palette = str(sc.get("palette", "graphite"))
        self.grain = fx.Grain(W, H, amount=float(sc.get("grain", 3.6)))
        self.vig = fx.vignette(W, H, float(sc.get("vignette", 0.30)))
        self.fin = {"halation": 0.05, "saturation": 0.94, **sc.get("finish", {})}
        self.scrim = float(sc.get("caption_scrim", 0.42))
        yy = np.linspace(0, 1, H, dtype=np.float32)
        self.scrim_map = (fx.smooth01((yy - 0.60) / 0.40) ** 1.4)[:, None, None]      # darkens the bottom, where captions sit
        self.seg_start = self.tl.off
        self.seg_src = self.tl.src
        self.card_mask = np.zeros(len(talk), bool)
        for f0, f1 in plan.get("cards", []):
            self.card_mask[f0:f1] = True

    # --- scene helpers
    def theme_at(self, t):
        s = self.tl.to_src(t)
        for e in self.sched:
            if float(e["from"]) <= s < float(e["to"]):
                return e["theme"]
        return self.default_theme

    def theme_change(self, t, theme):
        """(previous theme, when this one began) if it began within the last 0.4 s, else (None, None).
        Worked out from the time alone, so any piece of the video can be rendered on its own."""
        for j in range(1, 9):
            tt = t - j * 0.05
            if tt < 0:
                return None, None
            prev = self.theme_at(tt)
            if prev != theme:
                return prev, tt + 0.05
        return None, None

    def bg(self, theme, t, k=0):
        if theme not in self._bg:
            self._bg[theme] = backgrounds.make(theme, BW, BH, palette=self.palette)
        return cv2.resize(self._bg[theme].frame(t, k), (W, H), interpolation=cv2.INTER_CUBIC)

    def cut_index(self, t):
        return max(0, int(np.searchsorted(self.seg_start, t, side="right")) - 1)

    def zoom_at(self, t, talk):
        """Alternating punch on every cut (hides the jump) plus a slow push-in inside each piece."""
        k = max(0, int(np.searchsorted(self.seg_start, t, side="right")) - 1)
        seg_len = self.seg_src[k][1] - self.seg_src[k][0]
        within = (t - self.seg_start[k]) / max(seg_len, 0.1)
        base = 1.0 if k % 2 == 0 else (1.085 if talk else 1.05)
        return base * (1.0 + 0.03 * within)

    def sfx_events(self):
        ev = []
        for e in self.elements:
            ev += [(kind, max(0.0, t)) for kind, t in e.sfx]
        prev = None
        for t in np.arange(0.0, self.duration, 0.5):
            th = self.theme_at(t)
            if prev is not None and th != prev and self.is_talk(t):
                ev.append(("whoosh", t))
            prev = th
        return sorted(ev, key=lambda x: x[1])

    def is_card(self, t):
        n = min(len(self.card_mask) - 1, max(0, int(round(self.tl.to_src(t) * self.fps))))
        return bool(self.card_mask[n])

    def is_talk(self, t):
        n = min(len(self.talk) - 1, max(0, int(round(self.tl.to_src(t) * self.fps))))
        return bool(self.talk[n])

    # --- one frame
    def frame(self, n, rgb, alpha):
        t = n / self.fps
        talk = self.is_talk(t)
        card = self.is_card(t)
        zoom = self.zoom_at(t, talk)
        k = self.cut_index(t)
        for e in self.elements:
            e.talk = talk or card
        if card:                                   # picture unusable: his words over the branded background
            canvas = self.bg(self.theme_at(t), t, k)
            for e in self.elements:
                if e.alive(t):
                    e.back(canvas, t)
            fx.finish(canvas, **self.fin)
        elif talk:
            theme = self.theme_at(t)
            canvas = self.bg(theme, t, k)
            prev, tc = self.theme_change(t, theme)
            if prev is not None:                                           # cross-fade between themes
                x = (t - tc) / 0.4
                canvas = canvas * x + self.bg(prev, t, k) * (1 - x)
            for e in self.elements:
                if e.alive(t):
                    e.back(canvas, t)
            sub, a = self.subject.layer(rgb, alpha, zoom, canvas)
            band = a[int(H * 0.20):int(H * 0.38), :, 0].max(axis=0) > 0.5     # where his head is, so stickers keep clear
            xs = np.flatnonzero(band)
            for e in self.elements:
                e.avoid = (int(xs[0]), int(xs[-1])) if len(xs) else None
            canvas = sub * a + canvas * (1 - a)
            fx.finish(canvas, **self.fin)
        else:
            canvas = self.subject.footage(rgb, zoom)
            for e in self.elements:
                if e.alive(t):
                    e.back(canvas, t)
            fx.finish(canvas, **{**self.fin, "halation": self.fin["halation"] * 0.5})
        canvas *= 1 - (self.scrim * (1.0 if (talk or card) else 0.7)) * self.scrim_map
        for e in self.elements:
            if e.alive(t):
                e.front(canvas, t)
        self.captions.draw(canvas, t)
        canvas *= self.vig
        self.grain(canvas, n)
        return np.clip(canvas, 0, 255).astype(np.uint8)


def sel_expr(frames):
    return "+".join(f"between(n,{a},{b - 1})" for a, b in frames)


def stream(path, frames, fps, w, h, pix):
    """Decoder process yielding raw frames of `path` for exactly the kept frame ranges."""
    sel = sel_expr(frames)
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{sel}',setpts=N/{fps}/TB", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", pix, "-"], stdout=subprocess.PIPE)


def chunk_ranges(frames, n0, n1):
    """Source frame ranges that make up edited frames [n0, n1)."""
    out, acc = [], 0
    for a, b in frames:
        s, e = acc, acc + (b - a)
        lo, hi = max(s, n0), min(e, n1)
        if hi > lo:
            out.append([a + (lo - s), a + (hi - s)])
        acc = e
    return out


def _encode_cmd(R, cfg, out_path, extra_in=(), extra_map=(), vf=""):
    """NVENC H.264 at reel quality; colour converted with the BT.709 matrix and tagged as such,
    so every player shows the colours the compositor made."""
    from .render import _encode_args
    o = dict(cfg["outputs"]["instagram"])
    o["fps"] = R.fps
    o.update(cfg.get("short", {}).get("encode", {}))
    conv = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    return ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(R.fps),
            "-i", "-", *extra_in, "-vf", (vf + "," if vf else "") + conv, *extra_map, *_encode_args(o, cfg),
            "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0"]


def sources(R, conform, alpha, n0, n1):
    """Decoder processes for edited frames [n0, n1): the enhanced layers (edit order) or the originals."""
    info = R.plan["info"]
    if R.enh:
        rng = [[n0, n1]]
        return (stream(R.enh["rgb"], rng, R.fps, R.enh["w"], R.enh["h"], "bgr24"),
                stream(R.enh["alpha"], rng, R.fps, R.enh["w"], R.enh["h"], "gray"), R.enh["w"], R.enh["h"])
    rng = chunk_ranges(R.plan["segments"], n0, n1)
    return (stream(conform, rng, R.fps, info["width"], info["height"], "bgr24"),
            stream(alpha, rng, R.fps, info["width"], info["height"], "gray"), info["width"], info["height"])


def render_range(R, conform, alpha, n0, n1, out_path, cfg):
    """Composite and encode edited frames [n0, n1) to a video-only file (one worker's share)."""
    vid, alp, sw, sh = sources(R, conform, alpha, n0, n1)
    cmd = _encode_cmd(R, cfg, out_path, extra_map=("-an",))
    enc = subprocess.Popen(cmd + [str(out_path)], stdin=subprocess.PIPE)
    vs, as_ = sw * sh * 3, sw * sh
    done = 0
    for n in range(n0, n1):
        vb, ab = vid.stdout.read(vs), alp.stdout.read(as_)
        if len(vb) < vs or len(ab) < as_:
            break
        enc.stdin.write(R.frame(n, np.frombuffer(vb, np.uint8).reshape(sh, sw, 3),
                                np.frombuffer(ab, np.uint8).reshape(sh, sw)).tobytes())
        done += 1
        if done % 100 == 0:
            print(f"  frames {n0}-{n1}: {done}/{n1 - n0}", flush=True)
    vid.kill(), alp.kill()
    enc.stdin.close()
    enc.wait()
    return done


def render_parallel(R, audio_wav, out_path, cfg, workers):
    """Split the edit into pieces, render each in its own process, join them and add the sound."""
    import os
    import sys
    from pathlib import Path
    from .util import REPO, run
    total = int(round(R.duration * R.fps))
    step = -(-total // workers)
    cdir = Path(audio_wav).parent / "chunks"
    cdir.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(REPO))
    procs = []
    for k in range(workers):
        n0, n1 = k * step, min(total, (k + 1) * step)
        if n0 >= n1:
            break
        out = cdir / f"chunk{k}.mp4"
        out.unlink(missing_ok=True)
        logf = open(cdir / f"chunk{k}.log", "w")
        procs.append((subprocess.Popen(
            [sys.executable, "-m", "vedit", "chunk", str(R.plan["source"]), str(n0), str(n1), str(out)],
            stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)), out, logf, (n0, n1)))
    log(f"compositing {total} frames at {W}x{H} in {len(procs)} parallel pieces ...")
    for p, out, logf, (n0, n1) in procs:
        p.wait()
        logf.close()
        if p.returncode or not out.exists():
            raise RuntimeError(f"piece {n0}-{n1} failed; see {cdir / out.with_suffix('.log').name}")
    lst = cdir / "list.txt"
    lst.write_text("".join(f"file '{out.name}'\n" for _, out, _, _ in procs), encoding="utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", audio_wav, "-map", "0:v",
         "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
         "-t", f"{R.duration:.3f}", out_path])
    return total


def render(R, conform, alpha, audio_wav, ass_path, out_path, cfg, fonts_dir, frames_limit=None):
    """Run the compositor over the whole edit and encode with NVENC (in parallel pieces when configured)."""
    workers = int(cfg.get("short", {}).get("workers", 1))
    if workers > 1 and not frames_limit and not ass_path and int(round(R.duration * R.fps)) > 200:
        return render_parallel(R, audio_wav, out_path, cfg, workers)
    total = int(round(R.duration * R.fps))
    n1 = min(total, frames_limit) if frames_limit else total
    vid, alp, sw, sh = sources(R, conform, alpha, 0, n1)
    vf = f"ass={ass_path.name}:fontsdir={fonts_dir.name}" if ass_path else ""
    cmd = _encode_cmd(R, cfg, out_path, extra_in=("-i", str(audio_wav)), extra_map=("-map", "0:v", "-map", "1:a"), vf=vf)
    enc = subprocess.Popen(cmd + ["-t", f"{R.duration:.3f}", str(out_path)], stdin=subprocess.PIPE,
                           cwd=ass_path.parent if ass_path else None)
    n, vs, as_ = 0, sw * sh * 3, sw * sh
    log(f"compositing {total} frames at {W}x{H} ...")
    while n < n1:
        vb, ab = vid.stdout.read(vs), alp.stdout.read(as_)
        if len(vb) < vs or len(ab) < as_:
            break
        enc.stdin.write(R.frame(n, np.frombuffer(vb, np.uint8).reshape(sh, sw, 3),
                                np.frombuffer(ab, np.uint8).reshape(sh, sw)).tobytes())
        n += 1
        if n % 250 == 0:
            log(f"  {n}/{total} frames")
    vid.kill(), alp.kill()
    enc.stdin.close()
    enc.wait()
    return n


def preview_frame(R, conform, alpha, t_edit, out_png, fps=None):
    """Render one edited-time frame to a PNG (for judging the look without a full render).
    Uses the cached enhanced layers when they exist; otherwise enhances just this frame on the GPU."""
    fps = R.fps
    n = int(round(t_edit * fps))
    # which source frame does edited frame n come from?
    k = max(0, int(np.searchsorted(R.seg_start, n / fps, side="right")) - 1)
    src_n = int(round(R.seg_src[k][0] * fps + (n / fps - R.seg_start[k]) * fps))
    info = R.plan["info"]
    sw, sh = info["width"], info["height"]

    def grab(path, idx, pix, w, h):
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", pix, "-"], capture_output=True)
        buf = np.frombuffer(p.stdout, np.uint8)
        return buf.reshape(h, w, 3) if pix == "bgr24" else buf.reshape(h, w)

    if R.enh:
        rgb = grab(R.enh["rgb"], n, "bgr24", R.enh["w"], R.enh["h"])
        a = grab(R.enh["alpha"], n, "gray", R.enh["w"], R.enh["h"])
    elif getattr(R, "ctx", None) and R.ctx.get("live"):
        from . import enhance
        c = R.ctx
        sc = R.cfg.get("short", {})
        enh = enhance.Enhancer(sw, sh, sc.get("enhance", {}), sc.get("matte", {}))
        src = grab(conform, src_n, "bgr24", sw, sh)[:, :, ::-1].copy()
        al = grab(alpha, src_n, "gray", sw, sh)
        corr = None
        if c["stab"] is not None and c["stab"][src_n]:
            cc = c["corr"][src_n]
            corr = (float(cc[0]), float(cc[1]), float(cc[2]), float(cc[3]), float(sc.get("enhance", {}).get("stab_zoom", 1.07)))
        o, ao = enh.process(src, al, bool(R.talk[src_n]) if src_n < len(R.talk) else False, corr)
        rgb, a = o[:, :, ::-1].copy(), ao
        R.subject = Subject(enh.tw, enh.th, R.subject.face, refined=True, cfg=R.cfg)
    else:
        rgb = grab(conform, src_n, "bgr24", sw, sh)
        a = grab(alpha, src_n, "gray", sw, sh)
    img = R.frame(n, rgb, a)
    cv2.imwrite(str(out_png), img)
    return img
