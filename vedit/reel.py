"""Vertical Reel / Short for phone footage: matte, composite, English captions, sound design, encode."""
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import captions_en, enhance, footage, shorts, sfx
from .matte import conform, matte
from .render import cut_media, master_audio
from .util import REPO, load_json, log


def is_reel(plan):
    info = plan["info"]
    return plan["layout"]["mode"] == "full" and info["height"] > info["width"]


def load_graphics(wd):
    p = wd / "graphics.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def scenes(video, info, cfg, wd, fps):
    """Conformed footage, its matte and, per source frame, whether he is on camera."""
    sc = cfg.get("short", {})
    conf = conform(video, wd, info)
    alpha = matte(conf, wd, info["width"], info["height"], info["fps_out"],
                  ratio=float(sc.get("matte", {}).get("ratio", 0.75)))
    talk_path = wd / "talk.npy"
    if talk_path.exists() and talk_path.stat().st_mtime > alpha.stat().st_mtime:
        talk = np.load(talk_path)
    else:
        talk = shorts.classify_scenes(conf, alpha, info, fps)
        np.save(talk_path, talk)
    return conf, alpha, talk


def prepare(plan, cfg, wd, build=True):
    """Everything the compositor needs, cached in the work folder.
    build=False (previews) uses the enhanced footage only if it already exists."""
    video = Path(plan["source"])
    info = plan["info"]
    sc = cfg.get("short", {})
    conf, alpha, talk = scenes(video, info, cfg, wd, plan["fps"])
    ec = sc.get("enhance", {})
    use = str(ec.get("enabled", "auto")).lower() != "off" and enhance.available()
    corr = stab = enh = None
    if use:
        fc = {**sc.get("footage", {}), "stab_sigma": ec.get("stab_sigma", 15), "stab_limit": ec.get("stab_limit", 0.03)}
        m = footage.analyze(conf, info, wd)
        n = len(m["tx"])
        foot = ~np.pad(talk, (0, max(0, n - len(talk))), constant_values=False)[:n]
        if ec.get("stabilize", True):
            corr, stab = footage.corrections(m, foot, footage.bad_frames(m, plan["fps"], fc) > 0, info["width"], fc)
        else:
            corr, stab = np.zeros((n, 4), np.float32), np.zeros(n, bool)
        enh = enhance.run(plan, cfg, wd, conf, alpha, talk, corr, stab, build=build)
    elif str(ec.get("enabled", "auto")).lower() != "off":
        log("no CUDA GPU / torch found: using the plain upscale (see setup.ps1 to enable the enhancer)")
    face = plan["layout"].get("face_focus") or (0.5, 0.33)
    en = english(wd)
    ev = captions_en.events(plan["cues"], en, max_chars=24, total=plan["duration"]) if en else []
    R = shorts.Shorts(plan, cfg, wd, load_graphics(wd), talk, tuple(face), ev, enh=enh)
    R.ctx = {"corr": corr, "stab": stab, "live": use and enh is None}
    log(f"scenes: {int(talk.sum())}/{len(talk)} source frames on camera; "
        f"{len(R.elements)} graphic elements; layers: {'enhanced' if enh else 'original'}")
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
    R, conf, alpha = prepare(plan, cfg, wd, build=False)
    shorts.preview_frame(R, conf, alpha, t_edit, out_png)
    return out_png
