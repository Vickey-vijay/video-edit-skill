"""Vertical short-form compositor (1080x1920 Reels / Shorts) for phone footage.

Per frame: background theme -> type behind him -> matted subject (edge-tightened, light-wrapped,
graded) -> stickers and hook in front -> vignette and film grain. Footage of a screen is
never matted (the chart would vanish); it gets the grade, a slow push-in and the same overlays."""
import math
import subprocess

import cv2
import numpy as np

from . import backgrounds, fx, motion
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

class Subject:
    """Turns a 480x848 matted frame into a 1080x1920 layer, with zoom about the face."""

    def __init__(self, src_w, src_h, face=(0.5, 0.33)):
        self.sw, self.sh = src_w, src_h
        self.face = face
        self.k = H / src_h                           # cover scale
        self.er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def layer(self, rgb, alpha, zoom, bg_blur):
        k = self.k * zoom
        ow, oh = int(round(self.sw * k)), int(round(self.sh * k))
        cx = self.face[0] * ow
        cy = self.face[1] * oh
        x0 = int(np.clip(cx - W / 2, 0, max(0, ow - W)))
        y0 = int(np.clip(cy - H * 0.36, 0, max(0, oh - H)))
        rgb_b = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)[y0:y0 + H, x0:x0 + W].astype(np.float32)
        a_b = cv2.resize(alpha, (ow, oh), interpolation=cv2.INTER_CUBIC)[y0:y0 + H, x0:x0 + W].astype(np.float32) / 255
        if rgb_b.shape[0] != H or rgb_b.shape[1] != W:       # source narrower than the frame at zoom 1
            rgb_b = cv2.resize(rgb_b, (W, H))
            a_b = cv2.resize(a_b, (W, H))
        # tighten the matte: pull the soft edge in a hair so the old room never shows as a halo
        a_b = np.clip((a_b - 0.10) / 0.80, 0, 1)
        a_in = cv2.erode(a_b, self.er)
        a_b = cv2.GaussianBlur(a_in, (0, 0), 1.0)
        rgb_b = fx.unsharp(rgb_b, 0.6, 1.3)
        fx.grade(rgb_b, contrast=1.12, saturation=1.08, warmth=0.03)
        # light wrap: a little of the new background bleeds onto his edge, so he sits in the scene
        edge = np.clip(a_b - cv2.erode(a_b, np.ones((15, 15), np.uint8)), 0, 1)[..., None]
        rgb_b += bg_blur * edge * 0.55
        return rgb_b, a_b[..., None]

    def footage(self, rgb, zoom, cx=0.5, cy=0.5):
        k = self.k * zoom
        ow, oh = int(round(self.sw * k)), int(round(self.sh * k))
        x0 = int(np.clip(cx * ow - W / 2, 0, max(0, ow - W)))
        y0 = int(np.clip(cy * oh - H / 2, 0, max(0, oh - H)))
        out = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)[y0:y0 + H, x0:x0 + W].astype(np.float32)
        if out.shape[0] != H or out.shape[1] != W:
            out = cv2.resize(out, (W, H))
        out = fx.unsharp(out, 0.5, 1.3)
        return fx.grade(out, contrast=1.10, saturation=1.05, warmth=0.02)


# ------------------------------------------------------------------ renderer

class Shorts:
    def __init__(self, plan, cfg, wd, gfx, talk, face=(0.5, 0.33), caption_events=()):
        self.plan, self.cfg, self.gfx = plan, cfg, gfx or {}
        info = plan["info"]
        self.fps = plan["fps"]
        self.tl = Timeline(plan["segments"], self.fps)
        self.duration = self.tl.duration
        self.talk = talk
        self.subject = Subject(info["width"], info["height"], face)
        self.elements = motion.build(self.gfx, self.tl, self.duration)
        self.captions = motion.Captions(caption_events)
        self.sched = self.gfx.get("themes", [])
        self.default_theme = self.gfx.get("theme", "studio")
        self._bg, self._last = {}, (None, -9.0)
        self.grain = fx.Grain(W, H, amount=float(cfg.get("short", {}).get("grain", 6.0)))
        self.vig = fx.vignette(W, H, 0.34)
        self.seg_start = self.tl.off
        self.seg_src = self.tl.src

    # --- scene helpers
    def theme_at(self, t):
        s = self.tl.to_src(t)
        for e in self.sched:
            if float(e["from"]) <= s < float(e["to"]):
                return e["theme"]
        return self.default_theme

    def bg(self, theme, t):
        if theme not in self._bg:
            self._bg[theme] = backgrounds.make(theme, BW, BH)
        return cv2.resize(self._bg[theme].frame(t), (W, H), interpolation=cv2.INTER_CUBIC)

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

    def is_talk(self, t):
        n = min(len(self.talk) - 1, max(0, int(round(self.tl.to_src(t) * self.fps))))
        return bool(self.talk[n])

    # --- one frame
    def frame(self, n, rgb, alpha):
        t = n / self.fps
        talk = self.is_talk(t)
        zoom = self.zoom_at(t, talk)
        for e in self.elements:
            e.talk = talk
        if talk:
            theme = self.theme_at(t)
            canvas = self.bg(theme, t)
            prev, tc = self._last
            if prev is not None and prev != theme and t - tc < 0.4:        # cross-fade between themes
                k = (t - tc) / 0.4
                canvas = canvas * k + self.bg(prev, t) * (1 - k)
            elif prev != theme:
                self._last = (theme, t) if prev is not None else (theme, -9.0)
            for e in self.elements:
                if e.alive(t):
                    e.back(canvas, t)
            blur = cv2.GaussianBlur(canvas, (0, 0), 16)
            sub, a = self.subject.layer(rgb, alpha, zoom, blur)
            canvas = sub * a + canvas * (1 - a)
        else:
            canvas = self.subject.footage(rgb, zoom)
            for e in self.elements:
                if e.alive(t):
                    e.back(canvas, t)
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


def render(R, conform, alpha, audio_wav, ass_path, out_path, cfg, fonts_dir, frames_limit=None):
    """Run the compositor over the whole edit and encode with NVENC in one pass."""
    from .render import _encode_args
    info = R.plan["info"]
    sw, sh, fps = info["width"], info["height"], R.fps
    frames = R.plan["segments"]
    total = int(round(R.duration * fps))
    vid = stream(conform, frames, fps, sw, sh, "bgr24")
    alp = stream(alpha, frames, fps, sw, sh, "gray")
    o = dict(cfg["outputs"]["instagram"])
    o["fps"] = fps
    vf = f"ass={ass_path.name}:fontsdir={fonts_dir.name}," if ass_path else ""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", str(audio_wav), "-vf", vf + "format=yuv420p", "-map", "0:v", "-map", "1:a",
           *_encode_args(o, cfg), "-t", f"{R.duration:.3f}", str(out_path)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=ass_path.parent if ass_path else None)
    n, vs, as_ = 0, sw * sh * 3, sw * sh
    log(f"compositing {total} frames at {W}x{H} ...")
    while n < total and (not frames_limit or n < frames_limit):
        vb, ab = vid.stdout.read(vs), alp.stdout.read(as_)
        if len(vb) < vs or len(ab) < as_:
            break
        rgb = np.frombuffer(vb, np.uint8).reshape(sh, sw, 3)
        a = np.frombuffer(ab, np.uint8).reshape(sh, sw)
        enc.stdin.write(R.frame(n, rgb, a).tobytes())
        n += 1
        if n % 250 == 0:
            log(f"  {n}/{total} frames")
    vid.kill(), alp.kill()
    enc.stdin.close()
    enc.wait()
    return n


def preview_frame(R, conform, alpha, t_edit, out_png, fps=None):
    """Render one edited-time frame to a PNG (for judging the look without a full render)."""
    fps = R.fps
    n = int(round(t_edit * fps))
    # which source frame does edited frame n come from?
    k = max(0, int(np.searchsorted(R.seg_start, n / fps, side="right")) - 1)
    src_n = int(round(R.seg_src[k][0] * fps + (n / fps - R.seg_start[k]) * fps))
    info = R.plan["info"]
    sw, sh = info["width"], info["height"]

    def grab(path, pix, size):
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{src_n})", "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", pix, "-"], capture_output=True)
        return np.frombuffer(p.stdout, np.uint8)

    rgb = grab(conform, "bgr24", sw * sh * 3).reshape(sh, sw, 3)
    a = grab(alpha, "gray", sw * sh).reshape(sh, sw)
    img = R.frame(n, rgb, a)
    cv2.imwrite(str(out_png), img)
    return img
