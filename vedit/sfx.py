"""Sound design, synthesised: whooshes, pops and impacts, mixed under his voice.

No sample library is needed. Whooshes are noise swept through a moving band, pops are a
fast pitch-dropping sine with a click, impacts are a sub thump plus a short noise burst
with a small reverb tail. They stay quiet: they punctuate the graphics, they never compete."""
import wave

import numpy as np

SR = 48000


def _stft_filter(noise, centers, width_oct=1.1, n_fft=2048, hop=512):
    """Shape noise by a spectral bump whose centre frequency follows `centers` (Hz per frame)."""
    win = np.hanning(n_fft)
    out = np.zeros(len(noise) + n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    n_frames = max(1, (len(noise) - n_fft) // hop + 1)
    for i in range(n_frames):
        frame = noise[i * hop:i * hop + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.fft.rfft(frame * win)
        c = centers[min(i, len(centers) - 1)]
        gain = np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20) / c)) / width_oct) ** 2)
        out[i * hop:i * hop + n_fft] += np.fft.irfft(spec * gain) * win
    return out[:len(noise)]


def whoosh(dur=0.6, rising=True, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    frames = max(1, (n - 2048) // 512 + 1)
    sweep = np.geomspace(250, 5200, frames) if rising else np.geomspace(5200, 250, frames)
    y = _stft_filter(noise, sweep)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.6
    y = y * env
    return y / (np.abs(y).max() + 1e-9) * 0.55


def pop(seed=2):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = 190 + 900 * np.exp(-t * 55)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 24)
    click = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-t * 400) * 0.35
    y = y + click
    return y / (np.abs(y).max() + 1e-9) * 0.6


def impact(seed=3):
    n = int(1.3 * SR)
    t = np.arange(n) / SR
    f = 44 + 70 * np.exp(-t * 11)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.6)
    burst = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-t * 26)
    frames = max(1, (n - 2048) // 512 + 1)
    burst = _stft_filter(burst, np.geomspace(2600, 700, frames), 1.6)
    y = y * 0.9 + burst * 0.45
    tail = np.zeros(n)
    for d, g in ((0.043, 0.30), (0.071, 0.22), (0.113, 0.15), (0.17, 0.09)):      # tiny room
        k = int(d * SR)
        tail[k:] += y[:-k] * g
    y = y + tail
    return y / (np.abs(y).max() + 1e-9) * 0.8


BANK = {"whoosh": lambda: whoosh(0.62), "pop": pop, "impact": impact}
LEVEL_DB = {"whoosh": -17.0, "pop": -15.0, "impact": -11.0}      # relative to a full-scale voice peak


def mix(voice_wav, events, out_wav):
    """Lay the effects under the voice. `events` = [(kind, time_in_seconds)]."""
    with wave.open(str(voice_wav)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    voice = raw.reshape(-1, ch)
    if ch == 1:
        voice = np.repeat(voice, 2, axis=1)
    out = voice.copy()
    cache = {}
    for kind, t in events:
        if kind not in BANK:
            continue
        if kind not in cache:
            s = BANK[kind]()
            cache[kind] = (s * 10 ** (LEVEL_DB[kind] / 20)).astype(np.float32)
        s = cache[kind]
        i = int(max(0.0, t) * sr)
        j = min(len(out), i + len(s))
        if j <= i:
            continue
        seg = s[:j - i]
        out[i:j, 0] += seg
        out[i:j, 1] += seg * 0.92
    out = np.clip(out, -1, 1)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype(np.int16).tobytes())
    return out_wav
