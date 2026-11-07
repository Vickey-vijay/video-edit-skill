"""GPU pass over the phone footage, run once per edit and cached:

1. super-resolution (Real-ESRGAN "general v3", official weights) with a denoise blend tuned for
   WhatsApp-compressed video, a little plain upscale mixed back so skin does not turn waxy, and
   temporal smoothing of the fine detail so the sharpening does not shimmer between frames;
2. a steadier picture on hand-held footage shots (camera path smoothed, one warp);
3. a much cleaner cut-out of the speaker: the low-resolution matte is re-snapped to the
   enhanced picture with a guided filter, then the colour of the old room is removed from his
   edge pixels (Blur-Fusion foreground estimation), so hair meets the new background without
   a halo.

Output (per edited frame, in edit order): subj_rgb.mp4 (enhanced picture, foreground colours
for talk shots) and subj_alpha.mkv (lossless matte). The compositor reads these instead of the
480x848 original."""
import hashlib
import json
import math
import subprocess
import time
import urllib.request

import numpy as np

from .util import HOME, log

WEIGHTS = {
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "wdn": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
}
SHA256 = {           # the exact official weights the reference reels were made with; anything else is refused
    "general": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    "wdn": "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
}
VERSION = 4


# ------------------------------------------------------------------ availability

def available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _weights(name):
    d = HOME / "models" / "realesrgan"
    d.mkdir(parents=True, exist_ok=True)
    p = d / WEIGHTS[name].rsplit("/", 1)[1]
    if not p.exists():
        log(f"downloading {p.name} (official Real-ESRGAN release, ~5 MB) ...")
        urllib.request.urlretrieve(WEIGHTS[name], p)
    want = SHA256.get(name)
    if want:
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            p.unlink()
            raise RuntimeError(f"{p.name}: checksum mismatch, file removed")
    return p


# ------------------------------------------------------------------ small tensor tools

def _box1(torch, x, r, dim):
    n = x.shape[dim]
    cs = torch.cat([torch.zeros_like(x.narrow(dim, 0, 1)), x.cumsum(dim)], dim)
    idx = torch.arange(n, device=x.device)
    hi = (idx + r + 1).clamp(max=n)
    lo = (idx - r).clamp(min=0)
    shape = [1] * x.dim()
    shape[dim] = n
    return (cs.index_select(dim, hi) - cs.index_select(dim, lo)) / (hi - lo).to(x.dtype).view(shape)


def box(torch, x, r):
    """Mean over a (2r+1) square window, edges handled by shrinking the window. O(1) per pixel."""
    return _box1(torch, _box1(torch, x, r, -1), r, -2)


def guided(torch, guide, p, r, eps):
    """He et al. guided filter, gray guide: snaps p (the matte) onto the edges of the picture."""
    mi, mp = box(torch, guide, r), box(torch, p, r)
    cov = box(torch, guide * p, r) - mi * mp
    var = box(torch, guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(torch, a, r) * guide + box(torch, b, r)


def blur_fusion(torch, img, a, r1, r2):
    """Approximate foreground colour (Germer et al., "Fast multi-level foreground estimation"
    and the two-pass Blur-Fusion variant): pulls colour out of the old room from edge pixels."""
    def step(f, b, r):
        ba = box(torch, a, r)
        bf = box(torch, f * a, r) / (ba + 1e-5)
        bb = box(torch, b * (1 - a), r) / ((1 - ba) + 1e-5)
        return (bf + a * (img - a * bf - (1 - a) * bb)).clamp(0, 1), bb
    f, bb = step(img, img, r1)
    f, _ = step(f, bb, r2)
    return f


def gauss_blur(torch, x, sigma):
    r = max(1, int(round(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    f = torch.nn.functional
    x = f.conv2d(f.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    return f.conv2d(f.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)


# ------------------------------------------------------------------ the network

def load_net(torch, denoise):
    nn = torch.nn
    F = torch.nn.functional

    class SRVGGNetCompact(nn.Module):
        def __init__(self, nf=64, nc=32, up=4):
            super().__init__()
            self.up = up
            body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
            for _ in range(nc):
                body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
            body.append(nn.Conv2d(nf, 3 * up * up, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(up)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + F.interpolate(x, scale_factor=self.up, mode="nearest")

    g = torch.load(_weights("general"), map_location="cpu", weights_only=True)["params"]
    if denoise < 0.999:
        w = torch.load(_weights("wdn"), map_location="cpu", weights_only=True)["params"]
        g = {k: denoise * g[k] + (1 - denoise) * w[k] for k in g}
    net = SRVGGNetCompact()
    net.load_state_dict(g, strict=True)
    return net.eval().cuda().half().to(memory_format=torch.channels_last)


# ------------------------------------------------------------------ geometry

def affine_theta(dx, dy, rot, dls, zoom, w, h):
    """grid_sample matrix for: correct the frame by (dx, dy, rot, dls) about its centre and zoom in.
    Pixels are those of the source frame (w x h); output and input share normalised coordinates."""
    s = math.exp(dls) * zoom
    c, sn = math.cos(rot), math.sin(rot)
    # output pixel offset p -> input offset R^-1 (p - t) / s
    lin = np.array([[c, sn], [-sn, c]]) / s
    t = -(lin @ np.array([dx, dy]))
    nin = np.array([[1 / (w / 2), 0], [0, 1 / (h / 2)]])
    nout = np.array([[w / 2, 0], [0, h / 2]])
    A = nin @ lin @ nout
    tt = nin @ t
    return np.array([[A[0, 0], A[0, 1], tt[0]], [A[1, 0], A[1, 1], tt[1]]], np.float32)


# ------------------------------------------------------------------ the enhancer

class Enhancer:
    def __init__(self, src_w, src_h, ec, mc):
        import torch
        self.torch = torch
        torch.backends.cudnn.benchmark = True
        self.sw, self.sh = src_w, src_h
        self.scale = float(ec.get("scale", 2.7))
        self.tw, self.th = int(round(src_w * self.scale)) // 2 * 2, int(round(src_h * self.scale)) // 2 * 2
        self.lam = float(ec.get("texture", 0.22))
        self.ema = float(ec.get("temporal", 0.55))
        self.mc = mc
        self.net = load_net(torch, float(ec.get("denoise", 0.5)))
        self.prev_x = None
        self.prev_d = None
        self.last = None

    def reset(self):
        self.prev_x = self.prev_d = None

    # --- picture
    def picture(self, rgb):
        torch = self.torch
        F = torch.nn.functional
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            sr = self.net(x.half().contiguous(memory_format=torch.channels_last)).float().clamp(0, 1)
            sr = F.interpolate(sr, size=(self.th, self.tw), mode="bilinear", antialias=True, align_corners=False)
            plain = F.interpolate(x, size=(self.th, self.tw), mode="bicubic", align_corners=False).clamp(0, 1)
            base = gauss_blur(torch, sr, 2.0)
            detail = sr - base
            if self.prev_d is not None and self.prev_x is not None:
                m = (x - self.prev_x).abs().mean(1, keepdim=True)
                m = F.interpolate(gauss_blur(torch, m, 2.0), size=(self.th, self.tw), mode="bilinear")
                beta = self.ema * (1 - (m / 0.05).clamp(0, 1))
                detail = beta * self.prev_d + (1 - beta) * detail
            self.prev_d, self.prev_x = detail, x
            out = ((1 - self.lam) * (base + detail) + self.lam * plain).clamp(0, 1)
        return out

    def warp(self, img, theta):
        torch = self.torch
        F = torch.nn.functional
        th = torch.from_numpy(theta)[None].cuda()
        grid = F.affine_grid(th, list(img.shape), align_corners=False)
        return F.grid_sample(img, grid, mode="bicubic", padding_mode="border", align_corners=False)

    # --- matte
    def matte(self, img, alpha):
        torch = self.torch
        F = torch.nn.functional
        mc = self.mc
        with torch.inference_mode():
            a0 = torch.from_numpy(alpha).cuda()[None, None].float() / 255
            a = F.interpolate(a0, size=(self.th, self.tw), mode="bicubic", align_corners=False).clamp(0, 1)
            gray = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
            a = guided(torch, gray, a, int(mc.get("guide_radius", 8)), float(mc.get("guide_eps", 3e-4))).clamp(0, 1)
            lo, hi = float(mc.get("lo", 0.10)), float(mc.get("hi", 0.92))
            a = ((a - lo) / (hi - lo)).clamp(0, 1)
            a = a * self._largest(a)
            a = gauss_blur(torch, a, float(mc.get("feather", 0.8)))
            fg = blur_fusion(torch, img, a, int(mc.get("fusion_radius", 90)), 6)
        return fg, a

    def _largest(self, a):
        """Keep the speaker (the large connected shapes), drop stray blobs of the old room."""
        import cv2
        torch = self.torch
        F = torch.nn.functional
        small = F.interpolate(a, scale_factor=1 / 8, mode="area")[0, 0].cpu().numpy()
        bw = (small > 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if n <= 2:
            return torch.ones_like(a)
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.isin(lab, 1 + np.flatnonzero(areas >= 0.2 * areas.max())).astype(np.uint8)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        m = torch.from_numpy(keep.astype(np.float32)).cuda()[None, None]
        m = F.interpolate(m, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return gauss_blur(torch, m, 6.0).clamp(0, 1)

    # --- one frame
    def process(self, rgb, alpha, talk, corr=None, hold=None):
        """rgb uint8 HxWx3, alpha uint8 HxW. Returns (uint8 rgb, uint8 alpha) at the enhanced size.
        corr: stabilising correction (dx, dy, rot, dlogscale, zoom) or None.
        hold: (k) push-in amount 0..1 when this frame repeats the last good one."""
        torch = self.torch
        with torch.inference_mode():
            if hold is not None and self.last is not None:
                img, a = self.last
                theta = affine_theta(0, 0, 0, 0, 1.0 + 0.035 * hold, self.sw, self.sh)
                img = self.warp(img, theta)
                a = self.warp(a, theta) if a is not None else None
            else:
                img = self.picture(rgb)
                if corr is not None:
                    img = self.warp(img, affine_theta(*corr[:4], corr[4], self.sw, self.sh))
                a = None
                if talk:
                    img, a = self.matte(img, alpha)
                self.last = (img, a)
            out = (img[0].permute(1, 2, 0) * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            if a is None:
                ao = np.zeros((self.th, self.tw), np.uint8)
            else:
                ao = (a[0, 0] * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
        return out, ao


# ------------------------------------------------------------------ driver

def _sel(frames):
    return "+".join(f"between(n,{a},{b - 1})" for a, b in frames)


def settings_key(plan, cfg, conform, alpha, corr):
    sc = cfg.get("short", {})
    blob = json.dumps([VERSION, plan["segments"], plan.get("holds", []), sc.get("enhance"), sc.get("matte"),
                       conform.stat().st_size, alpha.stat().st_size,
                       hashlib.md5(np.ascontiguousarray(corr).tobytes()).hexdigest() if corr is not None else ""],
                      sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def run(plan, cfg, wd, conform, alpha, talk, corr, stab, limit=None, build=True):
    """Enhance every kept frame. Returns {rgb, alpha, w, h} (cached while nothing it depends on changes)."""
    sc = cfg.get("short", {})
    ec, mc = sc.get("enhance", {}), sc.get("matte", {})
    info = plan["info"]
    sw, sh, fps = info["width"], info["height"], plan["fps"]
    rgb_p, a_p, key_p = wd / "subj_rgb.mp4", wd / "subj_alpha.mkv", wd / "subj.key"
    key = settings_key(plan, cfg, conform, alpha, corr)
    tw, th = int(round(sw * float(ec.get("scale", 2.7)))) // 2 * 2, int(round(sh * float(ec.get("scale", 2.7)))) // 2 * 2
    meta = {"rgb": rgb_p, "alpha": a_p, "w": tw, "h": th}
    if not limit and rgb_p.exists() and a_p.exists() and key_p.exists() and key_p.read_text() == key:
        log("enhanced footage cached")
        return meta
    if not build:
        return None
    frames = plan["segments"]
    holds, lead = {}, {}
    for h in plan.get("holds", []):
        f0, f1, ld = h[0], h[1], (h[2] if len(h) > 2 else 0)
        for f in range(f0, f1):
            holds[f] = (f - f0) / max(1, f1 - f0 - 1)
        if ld:
            lead[f0] = f1                     # frames f0..f1-1 show frame f1 (the first sharp one)
    enh = Enhancer(sw, sh, ec, mc)
    stab_zoom = float(ec.get("stab_zoom", 1.07))
    total = sum(b - a for a, b in frames)
    if limit:
        total = min(total, limit)
    dec_v = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(conform), "-vf",
                              f"select='{_sel(frames)}',setpts=N/{fps}/TB", "-vsync", "0", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    dec_a = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(alpha), "-vf",
                              f"select='{_sel(frames)}',setpts=N/{fps}/TB", "-vsync", "0", "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    enc_v = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}",
                              "-r", str(fps), "-i", "-", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "constqp",
                              "-qp", str(int(ec.get("qp", 10))), "-pix_fmt", "yuv420p", str(rgb_p)],
                             stdin=subprocess.PIPE)
    enc_a = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{tw}x{th}",
                              "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", "-slices", "4",
                              "-pix_fmt", "gray", str(a_p)], stdin=subprocess.PIPE)
    vs, as_ = sw * sh * 3, sw * sh
    t0, n = time.time(), 0
    log(f"enhancing {total} frames to {tw}x{th} (super-resolution + steadier picture + cleaner cut-out) ...")
    seg_starts = {a for a, _ in frames}
    card = np.zeros(max(1, max((b for _, b in frames), default=1)), bool)
    for f0, f1 in plan.get("cards", []):
        card[f0:f1] = True
    blank_rgb = np.zeros((th, tw, 3), np.uint8).tobytes()
    blank_a = np.zeros((th, tw), np.uint8).tobytes()
    seq = [f for a0, b0 in frames for f in range(a0, b0)]

    def read():
        vb, ab = dec_v.stdout.read(vs), dec_a.stdout.read(as_)
        return (vb, ab) if len(vb) == vs and len(ab) == as_ else None

    def emit(out, ao):
        nonlocal n
        enc_v.stdin.write(out.tobytes() if hasattr(out, "tobytes") else out)
        enc_a.stdin.write(ao.tobytes() if hasattr(ao, "tobytes") else ao)
        n += 1
        if n % 100 == 0:
            rate = n / (time.time() - t0)
            log(f"  enhanced {n}/{total} frames ({rate:.1f}/s, about {(total - n) / rate / 60:.1f} min left)")

    def args_for(f):
        c = None
        if stab is not None and f < len(stab) and stab[f]:
            c = (float(corr[f, 0]), float(corr[f, 1]), float(corr[f, 2]), float(corr[f, 3]), stab_zoom)
        return (bool(talk[f]) if f < len(talk) else False), c

    i = 0
    while i < len(seq) and n < total:
        f = seq[i]
        got = read()
        if got is None:
            break
        vb, ab = got
        i += 1
        if f in seg_starts:
            enh.reset()
        if card[f]:                              # shown as words on the background: no picture needed
            emit(blank_rgb, blank_a)
            continue
        is_talk, c = args_for(f)
        if f in lead:
            # soft frames at the start of a piece: show the first sharp frame after them instead
            k = lead[f] - f
            ahead = [got]
            for _ in range(k):
                nxt = read()
                if nxt is None:
                    break
                ahead.append(nxt)
            if len(ahead) == k + 1:
                a_talk, a_c = args_for(lead[f])
                out_a, ao_a = enh.process(np.frombuffer(ahead[-1][0], np.uint8).reshape(sh, sw, 3),
                                          np.frombuffer(ahead[-1][1], np.uint8).reshape(sh, sw), a_talk, a_c, None)
                for j in range(k):
                    out, ao = enh.process(None, None, a_talk, None, 1.0 - j / max(1, k - 1))
                    emit(out, ao)
                emit(out_a, ao_a)
                i += k
                continue
        rgb = np.frombuffer(vb, np.uint8).reshape(sh, sw, 3)
        al = np.frombuffer(ab, np.uint8).reshape(sh, sw)
        out, ao = enh.process(rgb, al, is_talk, c, holds.get(f))
        emit(out, ao)
    for p in (dec_v, dec_a):
        p.kill()
    for p in (enc_v, enc_a):
        p.stdin.close()
        p.wait()
    if not limit:
        key_p.write_text(key)
    log(f"enhanced {n} frames in {(time.time() - t0) / 60:.1f} min")
    return meta
