"""Render the plan: cut the source, normalise audio, then compose each output
(YouTube 16:9, Instagram 9:16) with ffmpeg."""
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .analyze import face_crop
from .captions import write_ass, write_srt
from .graphics import Brand, frame_overlay
from .util import REPO, has_nvenc, log, run

SR = 48000


# ---------------------------------------------------------------- cut

def cut_media(src, plan, wd, cfg):
    fps, frames, info = plan["fps"], plan["segments"], plan["info"]
    key = json.dumps(frames)
    stamp = wd / "cut.key"
    vid, aud = wd / "cut_video.mp4", wd / "cut_audio.wav"
    if stamp.exists() and stamp.read_text() == key and vid.exists() and aud.exists():
        log("cut media cached")
        return vid, aud
    off = int(round((info["audio_start"] - info["video_start"]) * fps))
    terms = "+".join(f"between(n,{a + off},{b - 1 + off})" for a, b in frames)
    (wd / "cut_filter.txt").write_text(
        f"[0:v]fps={fps},select='{terms}',setpts=N/{fps}/TB,format=yuv420p[v]", encoding="utf-8")
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "16"]
           if _nvenc(cfg) else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "15"])
    log(f"cutting video: {len(frames)} pieces")
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-filter_complex_script", "cut_filter.txt",
         "-map", "[v]", "-an", *enc, "-r", fps, vid.name], cwd=wd)

    log("cutting audio")
    if info["has_audio"]:
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-ac", "2",
                            "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
        raw = np.frombuffer(p.stdout, np.int16).reshape(-1, 2)
    else:
        raw = np.zeros((int(info["duration"] * SR), 2), np.int16)
    spf = SR // fps
    fade = max(1, int(SR * cfg["cuts"]["fade_ms"] / 1000))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    pieces = []
    for a, b in frames:
        s0, n = a * spf, (b - a) * spf
        seg = raw[s0:s0 + n].astype(np.float32)
        if len(seg) < n:
            seg = np.vstack([seg, np.zeros((n - len(seg), 2), np.float32)])
        if n > 2 * fade:
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        pieces.append(seg)
    out = np.clip(np.concatenate(pieces), -32768, 32767).astype(np.int16)
    with wave.open(str(aud), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(out.tobytes())
    stamp.write_text(key)
    return vid, aud


def master_audio(aud, wd, cfg):
    a = cfg["audio"]
    out = wd / "audio_master.wav"
    chain = [f"highpass=f={a['highpass']}"]
    if a.get("denoise"):
        chain.append("afftdn=nf=-25")
    target = f"I={a['loudness']}:TP={a['true_peak']}:LRA=11"
    p = run(["ffmpeg", "-hide_banner", "-i", aud, "-af", ",".join(chain + [f"loudnorm={target}:print_format=json"]),
             "-f", "null", "-"], quiet=True)
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr).group(0))
    second = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
              f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
              f":offset={m['target_offset']}:linear=true")
    run(["ffmpeg", "-y", "-v", "error", "-i", aud, "-af", ",".join(chain + [second]),
         "-ar", SR, "-c:a", "pcm_s16le", out])
    log(f"audio: {m['input_i']} LUFS -> {a['loudness']} LUFS")
    return out


# ---------------------------------------------------------------- geometry

def _fit(aspect, box):
    x, y, w, h = box
    if w / h > aspect:
        nw, nh = h * aspect, h
    else:
        nw, nh = w, w / aspect
    nw, nh = int(nw) // 2 * 2, int(nh) // 2 * 2
    return [int(x + (w - nw) / 2), int(y + (h - nh) / 2), nw, nh]


def _contain(sw, sh, box):
    x, y, w, h = box
    k = min(w / sw, h / sh)
    nw, nh = int(sw * k) // 2 * 2, int(sh * k) // 2 * 2
    return [int(x + (w - nw) / 2), int(y + (h - nh) / 2), nw, nh]


def geometry(kind, W, H, layout):
    g = {"W": W, "H": H}
    mode = layout["mode"]
    if mode == "full":
        sw, sh = layout["size"]
        aspect = W / H
        if sw / sh > aspect:
            cw, ch = int(sh * aspect) // 2 * 2, sh // 2 * 2
            fx = (layout.get("face_focus") or [0.5, 0.5])[0] * sw
            cx = int(min(max(fx - cw / 2, 0), sw - cw)) // 2 * 2
            g["src_content"] = [cx, 0, cw, ch]
        else:
            cw, ch = sw // 2 * 2, int(sw / aspect) // 2 * 2
            g["src_content"] = [0, int((sh - ch) / 2) // 2 * 2, cw, ch]
        g["content"] = [0, 0, W, H]
        g["frame"] = False
        g["captions_on_video"] = True
        if kind == "youtube":
            g.update(cap_size=64, cap_margin_v=110, cap_margin_x=160, broll=[W - 40 - 520, 40, 520, 380])
        else:
            g.update(cap_size=70, cap_margin_v=560, cap_margin_x=90, broll=[90, 300, 900, 660])
        g["cap_margin_l"] = g["cap_margin_r"] = g["cap_margin_x"]
        return g

    content_src = layout["content"]
    aspect = content_src[2] / content_src[3]
    face_tile = layout.get("face_tile") if mode == "screen_face" else None
    g["frame"] = True
    g["src_content"] = content_src
    if kind == "youtube":
        m, gap, col = 40, 28, 440
        band = 190
        face = [W - m - col, m, col, round(col * 9 / 16) // 2 * 2]
        area = [m, m, W - m - col - gap - m, H - m - band]
        c = _fit(aspect, area)
        g["content"] = [c[0], m, c[2], c[3]]  # top-aligned with the face card
        g["face"] = face if face_tile else None
        top = face[1] + face[3] + gap if face_tile else m
        g["title_box"] = {"x": face[0], "y": top + 16, "w": col, "series_size": 20,
                          "title_size": 46, "max_lines": 2}
        btop = top + 180  # below a two-line title
        g["broll"] = [face[0], btop, col, max(330, g["content"][1] + g["content"][3] - btop)]
        g["label_size"] = 20
        cb = g["content"][1] + g["content"][3]
        g.update(cap_size=60, cap_margin_v=max(30, int((H - cb) / 2 - 36)), cap_margin_x=40,
                 cap_margin_l=g["content"][0], cap_margin_r=W - g["content"][0] - g["content"][2])
    else:
        m = 60
        g["title_box"] = {"x": m, "y": 200, "w": W - 2 * m, "series_size": 26,
                          "title_size": 58, "max_lines": 2}
        fw = W - 2 * m
        face = [m, 440, fw, round(fw * 9 / 16) // 2 * 2]
        g["face"] = face if face_tile else None
        top = face[1] + face[3] + 30 if face_tile else 440
        g["content"] = _fit(aspect, [m, top, fw, 560 if face_tile else 900])
        g["broll"] = face if face_tile else [m, top + g["content"][3] + 20, fw, 400]
        g["broll_panel"] = bool(face_tile)  # the card stands in for the facecam
        g["label_size"] = 28
        cb = g["content"][1] + g["content"][3]
        g.update(cap_size=66, cap_margin_v=max(260, int(H - cb - 190)), cap_margin_x=70,
                 cap_margin_l=70, cap_margin_r=70)
    if face_tile:
        g["src_face"] = face_crop(face_tile, layout.get("face_center"), g["face"][2] / g["face"][3])
    return g


# ---------------------------------------------------------------- zoom

def _ss(p):
    return f"({p})*({p})*(3-2*({p}))"


def zoom_filter(events, fps, base_focus):
    if not events:
        return None
    T = f"(in/{fps})"
    amps, fxs, fys = [], [], []
    bx, by = base_focus
    for ev in events:
        s, e, L = ev["start"], ev["end"], ev["level"] - 1
        if ev["type"] == "punch":
            pin = f"clip(({T}-{s})/{ev['ease_in']},0,1)"
            pout = f"clip(({e}-{T})/{ev['ease_out']},0,1)"
            body = f"{L:.4f}*{_ss(pin)}*{_ss(pout)}"
        elif ev["type"] == "reveal":
            body = f"{L:.4f}*(1-{_ss(f'clip(({T}-{s})/{e - s:.3f},0,1)')})"
        else:
            body = f"{L:.4f}"
        cond = f"between({T},{s},{e})"
        amps.append(f"if({cond},{body},0)")
        if ev["type"] != "hold":
            fxs.append(f"if({cond},{ev['fx'] - bx:.3f},0)")
            fys.append(f"if({cond},{ev['fy'] - by:.3f},0)")
    Z = "(1+" + "+".join(amps) + ")"
    FX = f"({bx}+" + ("+".join(fxs) or "0") + ")"
    FY = f"({by}+" + ("+".join(fys) or "0") + ")"
    X0 = f"({FX}*W*(1-1/{Z}))"
    Y0 = f"({FY}*H*(1-1/{Z}))"
    X1 = f"({X0}+W/{Z})"
    Y2 = f"({Y0}+H/{Z})"
    enable = "+".join(f"between(t,{ev['start']},{ev['end']})" for ev in events)
    return (f"perspective=x0='{X0}':y0='{Y0}':x1='{X1}':y1='{Y0}':x2='{X0}':y2='{Y2}'"
            f":x3='{X1}':y3='{Y2}':interpolation=cubic:sense=source:eval=frame:enable='{enable}'")


# ---------------------------------------------------------------- compose

def _nvenc(cfg):
    if cfg.get("encoder") == "x264":
        return False
    if cfg.get("encoder") == "nvenc":
        return True
    if not hasattr(_nvenc, "ok"):
        _nvenc.ok = has_nvenc()
    return _nvenc.ok


def _encode_args(o, cfg):
    if _nvenc(cfg):
        v = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(o["cq"]),
             "-b:v", "0", "-maxrate", o["maxrate"], "-bufsize", o["maxrate"].replace("M", "") + "M",
             "-profile:v", "high", "-bf", "3", "-g", str(o["fps"] * 2)]
    else:
        v = ["-c:v", "libx264", "-preset", "medium", "-crf", str(o["cq"] - 2), "-maxrate", o["maxrate"],
             "-bufsize", o["maxrate"], "-profile:v", "high", "-g", str(o["fps"] * 2)]
    return v + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
                "-movflags", "+faststart"]


def _broll_source(item_file):
    p = Path(item_file)
    return p if p.is_absolute() else REPO / "assets" / "broll" / p


def render_output(kind, plan, cfg, wd, video, audio, out_path):
    o = cfg["outputs"][kind]
    W, H, fps, dur = o["width"], o["height"], plan["fps"], plan["duration"]
    br = Brand(cfg["brand"])
    geo = geometry(kind, W, H, plan["layout"])
    title = plan.get("title") or ""

    fonts = wd / "fonts"
    if not fonts.exists():
        shutil.copytree(REPO / "assets" / "fonts", fonts)

    inputs = [["-i", video], ["-i", audio]]
    graph = []
    bgc = cfg["brand"]["colors"]["bg"].lstrip("#")[:6]
    graph.append(f"color=c=0x{bgc}:s={W}x{H}:r={fps}:d={dur:.3f}[base]")
    cx, cy, cw, ch = geo["src_content"]
    px, py, pw, ph = geo["content"]
    base_focus = (plan["layout"].get("face_focus") or [0.5, 0.45]) if plan["layout"]["mode"] == "full" else [0.5, 0.5]
    zf = zoom_filter(plan["zooms"], fps, base_focus) if cfg["zoom"].get("enabled") else None
    split = "split=2[vc][vf];[vc]" if geo.get("src_face") else ""
    graph.append(f"[0:v]{split}crop={cw}:{ch}:{cx}:{cy}" + (f",{zf}" if zf else "")
                 + f",scale={pw}:{ph}:flags=lanczos[content]")
    graph.append(f"[base][content]overlay={px}:{py}[l0]")
    last = "l0"
    if geo.get("src_face"):
        fx, fy, fw, fh = geo["src_face"]
        qx, qy, qw, qh = geo["face"]
        graph.append(f"[vf]crop={fw}:{fh}:{fx}:{fy},scale={qw}:{qh}:flags=lanczos[face]")
        graph.append(f"[{last}][face]overlay={qx}:{qy}[l1]")
        last = "l1"
    if geo.get("frame"):
        frame_png = wd / f"frame_{kind}.png"
        frame_overlay(W, H, br, geo, title, cfg["brand"].get("series", ""), cfg["brand"].get("name", "")).save(frame_png)
        inputs.append(["-loop", "1", "-framerate", str(fps), "-i", frame_png.name])
        graph.append(f"[{last}][{len(inputs) - 1}:v]overlay=0:0[l2]")
        last = "l2"

    for k, ev in enumerate(plan["broll"]):
        src = _broll_source(ev["file"])
        if not src.exists():
            log(f"b-roll missing, skipped: {src}")
            continue
        d = ev["end"] - ev["start"]
        if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            from PIL import Image
            img = Image.open(src).convert("RGBA")
            if geo.get("broll_panel"):
                from .graphics import card_panel
                bx, by, bw, bh = geo["broll"]
                img = card_panel(br, img, bw, bh)
            src = wd / f"broll_{kind}_{k}.png"
            img.save(src)
            sw, sh = img.size
            inputs.append(["-loop", "1", "-framerate", str(fps), "-t", f"{d:.3f}", "-i", src.name])
        else:
            sw, sh = _video_size(src)
            inputs.append(["-t", f"{d:.3f}", "-i", src])
        bx, by, bw, bh = _contain(sw, sh, geo["broll"])
        idx = len(inputs) - 1
        s = ev["start"]
        graph.append(f"[{idx}:v]format=rgba,scale={bw}:{bh}:flags=lanczos,fps={fps},"
                     f"fade=t=in:st=0:d=0.3:alpha=1,fade=t=out:st={max(0, d - 0.3):.3f}:d=0.3:alpha=1,"
                     f"setpts=PTS-STARTPTS+{s}/TB[b{k}]")
        graph.append(f"[{last}][b{k}]overlay=x='{bx}+40*(1-min(1,(t-{s})/0.3))':y={by}"
                     f":eof_action=pass:enable='between(t,{s},{ev['end']})'[r{k}]")
        last = f"r{k}"

    if cfg["captions"].get("enabled"):
        write_ass(plan, cfg, geo, wd / f"captions_{kind}.ass")
        graph.append(f"[{last}]ass=captions_{kind}.ass:fontsdir=fonts[cap]")
        last = "cap"
    accent = cfg["brand"]["colors"]["accent"].lstrip("#")[:6]
    pb = 6 if kind == "youtube" else 8
    graph.append(f"color=c=0x{accent}:s={W}x{pb}:r={fps}[pb]")
    graph.append(f"[{last}][pb]overlay=x='-w+w*t/{dur:.3f}':y={H - pb}:shortest=1,format=yuv420p[vout]")
    (wd / f"graph_{kind}.txt").write_text(";\n".join(graph), encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-v", "error", "-stats"]
    for i in inputs:
        cmd += [str(x) for x in i]
    cmd += ["-filter_complex_script", f"graph_{kind}.txt", "-map", "[vout]", "-map", "1:a",
            *_encode_args(o, cfg), "-t", f"{dur:.3f}", str(out_path)]
    log(f"rendering {kind} {W}x{H}, {dur:.1f}s ...")
    run(cmd, cwd=wd)
    return out_path


def _video_size(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", path], quiet=True)
    w, h = p.stdout.strip().split(",")[:2]
    return int(w), int(h)


def split_parts(full, plan, max_s, cfg, out_dir, stem):
    """Split an over-long Instagram render at caption boundaries."""
    dur = plan["duration"]
    if dur <= max_s:
        return [full]
    words = plan["words"]
    ends = [g["end"] for g in plan["captions"]]
    # prefer the end of a sentence; otherwise any caption boundary
    sentence = [g["end"] for g in plan["captions"] if words[g["words"][-1]]["text"].endswith((".", "?", "!"))]
    n = int(-(-dur // max_s))  # equal-length parts, not one long part and a scrap
    cuts, start = [], 0.0
    for k in range(1, n):
        target = dur * k / n
        options = [e for e in sentence if abs(e - target) < 12 and e - start <= max_s] or \
                  [e for e in ends if start < e <= start + max_s]
        end = min(options, key=lambda e: abs(e - target)) if options else min(start + max_s, target)
        cuts.append((start, end))
        start = end
    cuts.append((start, dur))
    parts = []
    o = cfg["outputs"]["instagram"]
    for n, (a, b) in enumerate(cuts, 1):
        out = out_dir / f"{stem}_instagram_part{n}.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", full,
             *_encode_args(o, cfg), out])
        parts.append(out)
    return parts


def contact_sheet(mp4, dur, out_jpg, cols=3, rows=2):
    n = cols * rows
    run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vf",
         f"fps={n}/{dur:.3f},scale=640:-2,tile={cols}x{rows}:padding=6:color=0x0E1828",
         "-frames:v", "1", out_jpg])
    return out_jpg
