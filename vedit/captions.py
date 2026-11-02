"""Burned-in captions as an ASS script: the word being spoken lights up in the
accent colour, keywords stay in the highlight colour, Tamil words switch font."""
import re

from .plan import TAMIL

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{fg},&H000000FF,{outline},{shadow},-1,0,0,0,100,100,0,0,1,{bord},{shad},2,{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_color(hex_rgb, alpha=0):
    h = hex_rgb.lstrip("#")[:6]
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ts(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def display(text, upper):
    t = re.sub(r"[,;:]+$", "", text.strip())
    t = re.sub(r"(?<!\d)\.+$", "", t)
    return t.upper() if upper and not TAMIL.search(t) else t


def esc(t):
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_ass(plan, cfg, geo, out_path):
    c, brand = cfg["captions"], cfg["brand"]
    col = brand["colors"]
    fonts = brand["fonts"]
    on_video = geo.get("captions_on_video", False)
    W, H = geo["W"], geo["H"]
    head = ASS_HEADER.format(
        W=W, H=H, font=fonts["display"]["family"], size=geo["cap_size"],
        fg=ass_color(col["fg"]), outline=ass_color(col["bg"], 0x20 if on_video else 0x00),
        shadow=ass_color(col["bg"], 0x60), bord=geo["cap_size"] // 14 if on_video else 0,
        shad=3 if on_video else 2, ml=geo["cap_margin_x"], mr=geo["cap_margin_x"], mv=geo["cap_margin_v"])
    fg, accent, hi = ass_color(col["fg"]), ass_color(col["accent"]), ass_color(col["highlight"])
    latin, tamil = fonts["display"]["family"], fonts["tamil"]["family"]
    words = plan["words"]
    lines = [head]
    for g in plan["captions"]:
        ws = [words[i] for i in g["words"] if not words[i].get("junk")]
        if not ws:
            continue
        texts = [display(w["text"], c.get("uppercase")) for w in ws]
        for k, w in enumerate(ws):
            s = g["start"] if k == 0 else w["ns"]
            e = ws[k + 1]["ns"] if k + 1 < len(ws) else g["end"]
            if e - s < 0.02:
                continue
            parts = []
            for j, (ww, tx) in enumerate(zip(ws, texts)):
                colr = accent if j == k else (hi if (ww.get("kw") or ww.get("emph")) else fg)
                fn = tamil if TAMIL.search(tx) else latin
                tag = f"{{\\fn{fn}\\c{colr}"
                if j == k:
                    tag += "\\fscx106\\fscy106"
                parts.append(tag + "}" + esc(tx) + "{\\fscx100\\fscy100}")
            intro = ""
            if k == 0 and c.get("pop", True):
                intro = "{\\fad(70,0)\\fscx88\\fscy88\\t(0,110,\\fscx100\\fscy100)}"
            lines.append(f"Dialogue: 0,{ts(s)},{ts(e)},Cap,,0,0,0,,{intro}{' '.join(parts)}\n")
    out_path.write_text("".join(lines), encoding="utf-8-sig")
    return out_path


def write_srt(plan, out_path):
    words = plan["words"]

    def st(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    rows = []
    for g in plan["captions"]:
        text = " ".join(display(words[i]["text"], False) for i in g["words"] if not words[i].get("junk"))
        if text:
            rows.append(f"{len(rows) + 1}\n{st(g['start'])} --> {st(g['end'])}\n{text}\n")
    out_path.write_text("\n".join(rows), encoding="utf-8")
    return out_path
