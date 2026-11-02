"""Vertical Reel / Short for phone footage: matte, composite, English captions, sound design, encode."""
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import captions_en, shorts, sfx
from .matte import conform, matte
from .render import cut_media, master_audio
from .util import REPO, load_json, log


def is_reel(plan):
    info = plan["info"]
    return plan["layout"]["mode"] == "full" and info["height"] > info["width"]


def load_graphics(wd):
    p = wd / "graphics.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def prepare(plan, cfg, wd):
    """Everything the compositor needs, cached in the work folder."""
    video = Path(plan["source"])
    info = plan["info"]
    conf = conform(video, wd, info)
    alpha = matte(conf, wd, info["width"], info["height"], info["fps_out"], ratio=0.5)
    talk_path = wd / "talk.npy"
    if talk_path.exists() and talk_path.stat().st_mtime > alpha.stat().st_mtime:
        talk = np.load(talk_path)
    else:
        talk = shorts.classify_scenes(conf, alpha, info, plan["fps"])
        np.save(talk_path, talk)
    face = plan["layout"].get("face_focus") or (0.5, 0.33)
    en = english(wd)
    ev = captions_en.events(plan["cues"], en, max_chars=24, total=plan["duration"]) if en else []
    R = shorts.Shorts(plan, cfg, wd, load_graphics(wd), talk, tuple(face), ev)
    log(f"scenes: {int(talk.sum())}/{len(talk)} source frames on camera; "
        f"{len(R.elements)} graphic elements")
    return R, conf, alpha


def english(wd):
    p = wd / "captions_en.json"
    return load_json(p) if p.exists() else {}


def render_reel(plan, cfg, wd, out_dir):
    R, conf, alpha = prepare(plan, cfg, wd)
    video = Path(plan["source"])
    _, voice = cut_media(video, plan, wd, cfg)
    mixed = sfx.mix(voice, R.sfx_events(), wd / "audio_sfx.wav")
    master = master_audio(mixed, wd, cfg)
    en = english(wd)
    if en:
        captions_en.write_srt(plan["cues"], en, out_dir / f"{video.stem}_en.srt", plan["duration"])
        log(f"English captions: {len(R.captions.events)} caption events")
    else:
        log("no English captions yet (run: cues, then translate): rendering without captions")
    out = out_dir / f"{video.stem}_reel.mp4"
    n = shorts.render(R, conf, alpha, master, None, out, cfg, wd / "fonts")
    log(f"reel: {n} frames -> {out}")
    return out


def preview(plan, cfg, wd, t_edit, out_png):
    R, conf, alpha = prepare(plan, cfg, wd)
    shorts.preview_frame(R, conf, alpha, t_edit, out_png)
    return out_png
