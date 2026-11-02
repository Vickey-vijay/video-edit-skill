"""English captions for Tamil speech: simple, large, readable. One caption per phrase he
speaks, wrapped to two short lines, shown just as he says it. Written to ASS for burning
in, and to SRT for uploading to YouTube."""
import textwrap

from .captions import ts

HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Montserrat,{size},&H00FFFFFF,&H000000FF,&H00101010,&H90000000,-1,0,0,0,100,100,0.3,0,1,{outline},{shadow},2,{mx},{mx},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

READ_CPS = 16.5       # characters per second a viewer can comfortably read


def clean(text):
    t = " ".join(text.split())
    return t.replace("{", "(").replace("}", ")")


def events(cues, en, max_chars=28, total=None):
    """[(start, end, text with \\N line breaks)], timed from the phrase, never overlapping the next."""
    cues = [c for c in cues if clean(en.get(str(c["id"]), ""))]
    out = []
    for i, c in enumerate(cues):
        text = clean(en[str(c["id"])])
        nxt = cues[i + 1]["t0"] if i + 1 < len(cues) else (total or c["t1"] + 3)
        t0 = max(0.0, c["t0"] - 0.05)
        need = max(1.0, len(text) / READ_CPS)
        t1 = max(c["t1"] + 0.25, min(t0 + need, nxt - 0.03))
        t1 = min(t1, nxt - 0.03) if nxt - 0.03 > t0 + 0.4 else t1
        lines = textwrap.wrap(text, max_chars, break_long_words=False)
        groups = [lines[k:k + 2] for k in range(0, len(lines), 2)]
        weights = [sum(len(l) for l in g) for g in groups]
        span, cur = t1 - t0, t0
        for g, wgt in zip(groups, weights):
            d = span * wgt / sum(weights)
            out.append((round(cur, 3), round(cur + d, 3), "\\N".join(g)))
            cur += d
    return out


def write_ass(cues, en, out_path, W=1080, H=1920, size=66, margin_v=400, total=None):
    head = HEAD.format(W=W, H=H, size=size, outline=5, shadow=2, mx=90, mv=margin_v)
    rows = []
    for a, b, text in events(cues, en, total=total):
        rows.append(f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{{\\fad(90,60)}}{text}\n")
    out_path.write_text(head + "".join(rows), encoding="utf-8-sig")
    return len(rows)


def write_srt(cues, en, out_path, total=None):
    def st(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    rows = []
    for a, b, text in events(cues, en, 36, total):
        rows.append(f"{len(rows) + 1}\n{st(a)} --> {st(b)}\n{text.replace(chr(92) + 'N', chr(10))}\n")
    out_path.write_text("\n".join(rows), encoding="utf-8")
