"""Load default settings, then layer sidecar / --config files and --set overrides."""
import copy
from pathlib import Path

import yaml

from .util import REPO


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def parse_set(items):
    """--set zoom.punch_in=1.2 --set cuts.soft_fillers=off -> nested dict."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw) if raw else None
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def load_config(video=None, config_files=(), overrides=None):
    cfg = _read(REPO / "config" / "default.yaml")
    cfg["brand"] = _read(REPO / "config" / "brand.yaml")
    cfg["fillers"] = _read(REPO / "config" / "fillers.yaml")
    cfg["keywords"] = _read(REPO / "config" / "keywords.yaml").get("keywords", [])
    cfg["corrections"] = _read(REPO / "config" / "corrections.yaml").get("words", {})
    if video is not None:
        sidecar = Path(video).with_suffix(".yaml")
        if sidecar.exists():
            cfg = deep_merge(cfg, _read(sidecar))
    for f in config_files or ():
        p = Path(f)
        if not p.exists() and (REPO / p).exists():  # e.g. presets/meet-edge-share.yaml from anywhere
            p = REPO / p
        cfg = deep_merge(cfg, _read(p))
    return deep_merge(cfg, overrides or {})
