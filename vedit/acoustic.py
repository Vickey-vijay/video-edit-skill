"""Listening to the audio itself: speech phrases from the energy, and whether two
phrases open with the same sound (a repeated attempt), independent of what
Whisper wrote down or when it says the words were spoken."""
import wave

import numpy as np

HOP = 0.01
SR = 16000


def load_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def _mel_filterbank(n_fft=512, n_mels=40, fmin=60, fmax=7600):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    return fb


def mfcc(y, n_mfcc=13):
    """12 cepstral coefficients (no loudness term) every 10 ms."""
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    n_fft, hop = 400, 160
    fr = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * np.hamming(n_fft)
    spec = np.abs(np.fft.rfft(fr, 512, axis=1)) ** 2
    logmel = np.log(spec @ _mel_filterbank().T + 1e-8)
    n = logmel.shape[1]
    dct = np.cos(np.pi / n * (np.arange(n) + 0.5)[None, :] * np.arange(n_mfcc)[:, None])
    return (logmel @ dct.T)[:, 1:]


class Phrases:
    """Speech phrases: stretches of sound separated by a real pause."""

    def __init__(self, y, env, floor, min_pause=0.22, min_len=0.25):
        self.feat = mfcc(y)
        self.env, self.floor = env, floor
        dur = len(env) * HOP
        voiced = env > floor
        runs, i, n = [], 0, len(voiced)
        while i < n:                       # silent stretches long enough to separate phrases
            if not voiced[i]:
                j = i
                while j < n and not voiced[j]:
                    j += 1
                if (j - i) * HOP >= min_pause:
                    runs.append((i * HOP, j * HOP))
                i = j
            else:
                i += 1
        self.silences = runs
        out, t = [], 0.0
        for a, b in runs + [(dur, dur)]:
            if a - t >= min_len and voiced[int(t / HOP):int(a / HOP)].mean() > 0.35:
                out.append({"on": round(t, 3), "off": round(a, 3)})
            t = b
        self.items = out

    def onset_near(self, t, back=0.8):
        """Start of the sound burst that contains or immediately precedes time t."""
        for p in self.items:
            if p["on"] - 0.05 <= t <= p["off"] + 0.05:
                return p["on"]
        prior = [p["on"] for p in self.items if p["on"] <= t]
        return prior[-1] if prior and t - prior[-1] <= back else t

    def _window(self, a, b):
        f = self.feat[int(a / HOP):int(b / HOP)]
        return f - f.mean(axis=0) if len(f) else f

    def opening_cost(self, A, B):
        """How alike the first moments of phrase A and phrase B sound (lower = more alike).
        Unrelated pairs score about 50 (median); a repeated attempt is usually 27 to 40.
        B may begin with a connector word, so A is matched against any stretch inside
        the first moments of B rather than only its very start."""
        if A["off"] - A["on"] < 0.4:
            return 99.0
        L = max(0.45, min(0.9, 0.7 * (A["off"] - A["on"])))
        fa = self._window(A["on"], min(A["off"], A["on"] + L))
        fb = self._window(B["on"], min(B["off"], B["on"] + 0.5 + L * 1.6))
        if len(fa) < 40 or len(fb) < 40:      # too short to compare fairly
            return 99.0
        d = np.linalg.norm(fa[:, None, :] - fb[None, :, :], axis=2)
        n, m = d.shape
        free = int(0.5 / HOP)                  # where in B the match may begin
        D = np.full((n + 1, m + 1), np.inf)
        D[0, :free] = 0.0
        steps = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cands = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
                k = int(np.argmin(cands))
                pi, pj = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
                D[i, j] = d[i - 1, j - 1] + D[pi, pj]
                steps[i, j] = steps[pi, pj] + 1
        return float((D[n, 1:] / np.maximum(steps[n, 1:], 1)).min())
