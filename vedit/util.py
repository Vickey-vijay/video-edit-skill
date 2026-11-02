"""Shared helpers: paths, logging, subprocess, ffprobe, JSON."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # Tamil text in logs on a cp1252 Windows console
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
# Kept outside OneDrive so models and intermediate renders don't sync.
HOME = Path(os.environ.get("VEDIT_HOME", Path.home() / ".video-edit-skill"))
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def log(msg):
    print(f"[vedit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        log("$ " + " ".join(cmd)[:400])
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({p.returncode}):\n{p.stderr[-5000:]}")
    return p


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
               "-of", "json", path], quiet=True).stdout
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise ValueError(f"{path} has no video stream")
    num, den = (v.get("avg_frame_rate") or v["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        rot = int(sd.get("rotation", rot) or 0)
    rot = int(v.get("tags", {}).get("rotate", rot) or rot)
    if abs(rot) % 180 == 90:               # phone video: stored sideways, shown upright
        w, h = h, w
    # keep the cadence of the footage (phones record 24-30 fps, often variable)
    fps_out = min((24, 25, 30), key=lambda f: abs(f - fps)) if 23.0 <= fps <= 31.0 else 30
    return {
        "duration": float(data["format"]["duration"]),
        "width": w,
        "height": h,
        "rotation": rot,
        "fps": fps,
        "fps_out": fps_out,
        "video_start": float(v.get("start_time", 0) or 0),
        "audio_start": float(a.get("start_time", 0) or 0) if a else 0.0,
        "has_audio": a is not None,
    }


def file_key(path):
    st = Path(path).stat()
    h = hashlib.sha1(f"{Path(path).name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    return h[:8]


def work_dir(video):
    d = HOME / "work" / f"{Path(video).stem[:40]}-{file_key(video)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_nvenc():
    try:
        run(["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
             "color=s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"], quiet=True)
        return True
    except RuntimeError:
        return False
