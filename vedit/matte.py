"""Person matting with Robust Video Matting (ONNX), so animated backgrounds can sit
behind him. Runs over the whole uncut clip: the model carries state from frame to
frame and would ghost across a jump cut if it only ever saw the edited footage."""
import subprocess

import numpy as np

from .util import HOME, log, run

MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"


def model_path():
    p = HOME / "models" / "rvm" / "rvm_mobilenetv3_fp32.onnx"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing. Download it from {MODEL_URL}")
    return p


def conform(src, wd, info):
    """Upright, constant frame rate, visually lossless: the footage every later step reads."""
    out = wd / "conform.mp4"
    stamp = wd / "conform.key"
    key = f"{info['fps_out']}|{info['width']}x{info['height']}|{src.stat().st_size}"
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return out
    log(f"conforming footage to {info['fps_out']} fps, {info['width']}x{info['height']} upright")
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", f"fps={info['fps_out']}", "-an",
         "-c:v", "libx264", "-crf", "8", "-preset", "veryfast", "-pix_fmt", "yuv420p", out])
    stamp.write_text(key)
    return out


def matte(conformed, wd, w, h, fps, ratio=1.0, max_frames=None):
    """Write alpha.mkv (lossless grayscale, same frames as the conformed video)."""
    out = wd / "alpha.mkv"
    stamp = wd / "alpha.key"
    key = f"{conformed.stat().st_size}|{ratio}"
    if out.exists() and stamp.exists() and stamp.read_text() == key and not max_frames:
        log("matte cached")
        return out
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(model_path()), so, providers=["CPUExecutionProvider"])
    states = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
    ds = np.array([ratio], np.float32)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(conformed), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                            "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray", str(out)],
                           stdin=subprocess.PIPE)
    n, size = 0, w * h * 3
    log(f"matting {w}x{h} (CPU) ...")
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size or (max_frames and n >= max_frames):
            break
        x = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        fgr, pha, *states = sess.run(None, {"src": x, "r1i": states[0], "r2i": states[1],
                                            "r3i": states[2], "r4i": states[3], "downsample_ratio": ds})
        enc.stdin.write((np.clip(pha[0, 0], 0, 1) * 255).astype(np.uint8).tobytes())
        n += 1
        if n % 300 == 0:
            log(f"  matted {n} frames")
    dec.kill()
    enc.stdin.close()
    enc.wait()
    if not max_frames:
        stamp.write_text(key)
    log(f"matte: {n} frames")
    return out
