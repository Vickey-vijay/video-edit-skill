"""Command line entry point: python -m vedit <command> ..."""
import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

from .config import load_config, parse_set
from .util import REPO, VIDEO_EXTS, load_json, log, probe, save_json, work_dir

OUT = REPO / "output"


def _library(cfg):
    path = REPO / cfg["broll"]["library"]
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("items", [])
    cards = REPO / "assets" / "broll" / "cards"
    for it in items:
        if it.get("kind") in ("pattern", "fib", "waves"):
            it["file"] = f"cards/{it['id']}.png"
    missing = [it for it in items if it.get("kind") in ("pattern", "fib", "waves")
               and not (cards / f"{it['id']}.png").exists()]
    if missing:
        from .graphics import make_cards
        make_cards(cfg["brand"], missing, cards)
    return items


def resolve_layout(video, info, cfg, wd):
    from .analyze import detect_layout, face_center, layout_preview, sample_frames
    lc = cfg["layout"]
    frames = sample_frames(video, info["duration"])
    det = detect_layout(frames)
    mode = det["mode"] if lc["mode"] == "auto" else lc["mode"]
    screen = lc.get("screen") or det["screen"]
    tiles = det["tiles"]
    face_tile = lc.get("face") or (tiles[lc.get("face_tile", 0)] if len(tiles) > lc.get("face_tile", 0) else None)
    if mode == "screen_face" and not face_tile:
        mode = "screen"
    if mode in ("screen", "screen_face") and not screen:
        mode = "full"
    layout = {"mode": mode, "size": [info["width"], info["height"]], "screen": screen, "tiles": tiles,
              "face_tile": face_tile if mode == "screen_face" else None}
    if screen:
        x, y, w, h = screen
        trim = lc.get("screen_trim")
        if trim == "auto":
            from .analyze import auto_trim
            trim = auto_trim(frames, screen)
        layout["screen_trim"] = trim
        t, b, l, r = trim or [0, 0, 0, 0]
        layout["content"] = [int(x + l * w) // 2 * 2, int(y + t * h) // 2 * 2,
                             int(w * (1 - l - r)) // 2 * 2, int(h * (1 - t - b)) // 2 * 2]
    if mode == "screen_face":
        layout["face_center"] = face_center(frames, face_tile)
    if mode == "full":
        fc = face_center(frames)
        layout["face_focus"] = [fc["x"], fc["y"]] if fc else [0.5, 0.45]
    preview = dict(layout)
    if mode == "screen_face":
        from .analyze import face_crop
        preview["face"] = face_crop(face_tile, layout["face_center"], 16 / 9)
    layout_preview(frames[len(frames) // 2], preview, wd / "layout_preview.png")
    log(f"layout: {mode}  screen={screen}  content={layout.get('content')}  face_tile={layout['face_tile']}")
    return layout


def build_plan(video, cfg, wd):
    from .plan import make_plan
    from .transcribe import extract_audio_16k, transcribe
    info = probe(video)
    a16 = extract_audio_16k(video, wd / "audio16k.wav")
    tr = transcribe(a16, cfg["whisper"], wd / "transcript.json")
    layout = resolve_layout(video, info, cfg, wd)
    motion = None
    if cfg["zoom"].get("enabled") and cfg["zoom"].get("focus") == "auto" and layout.get("content"):
        from .analyze import Motion
        motion = Motion(video, layout["content"])
    title = cfg.get("title") or Path(video).stem.replace("_", " ").replace("-", " ").title()
    plan = make_plan(tr, a16, info, cfg, layout, motion, _library(cfg), title)
    plan["source"] = str(video)
    save_json(wd / "plan.json", plan)
    s = plan["stats"]
    log(f"plan: {s['source_duration']}s -> {s['edited_duration']}s (-{s['removed_pct']}%), "
        f"{s['cuts']} cuts, removed words {s['removed_words']}, {s['zooms']} zooms, {s['broll']} b-roll")
    return plan


def render_plan(plan, cfg, wd, only=None):
    from .render import contact_sheet, cut_media, master_audio, render_output, split_parts
    from .report import write_report
    from .captions import write_srt
    video = Path(plan["source"])
    stem = video.stem
    out_dir = OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cut_v, cut_a = cut_media(video, plan, wd, cfg)
    audio = master_audio(cut_a, wd, cfg)
    results = {}
    for kind in ("youtube", "instagram"):
        if not cfg["outputs"][kind].get("enabled") or (only and kind != only):
            continue
        out = out_dir / f"{stem}_{kind}.mp4"
        render_output(kind, plan, cfg, wd, cut_v, audio, out)
        results[kind] = [out]
        if kind == "instagram":
            results[kind] = split_parts(out, plan, cfg["outputs"]["instagram"]["max_seconds"], cfg, out_dir, stem)
        contact_sheet(out, plan["duration"], out_dir / f"{stem}_{kind}_preview.jpg",
                      *((3, 2) if kind == "youtube" else (6, 1)))
    if cfg["captions"].get("write_srt"):
        write_srt(plan, out_dir / f"{stem}.srt")
    save_json(out_dir / "plan.json", plan)
    write_report(plan, cfg, results, out_dir / "report.html")
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return results


def cmd_run(a):
    video = Path(a.video).resolve()
    cfg = load_config(video, a.config, parse_set(a.set))
    if a.title:
        cfg["title"] = a.title
    wd = work_dir(video)
    log(f"work dir: {wd}")
    plan = build_plan(video, cfg, wd)
    if a.stop_after == "plan":
        log(f"plan written: {wd / 'plan.json'}  (edit it, then: python -m vedit render \"{video}\")")
        return
    render_plan(plan, cfg, wd, a.only)


def cmd_render(a):
    video = Path(a.video).resolve()
    cfg = load_config(video, a.config, parse_set(a.set))
    wd = work_dir(video)
    plan = load_json(wd / "plan.json")
    if a.title:
        plan["title"] = a.title
    render_plan(plan, cfg, wd, a.only)


def cmd_layout(a):
    video = Path(a.video).resolve()
    cfg = load_config(video, a.config, parse_set(a.set))
    wd = work_dir(video)
    resolve_layout(video, probe(video), cfg, wd)
    print(wd / "layout_preview.png")


def cmd_review(a):
    """Print the plan compactly: numbered caption words, cuts, zooms, B-roll."""
    video = Path(a.video).resolve()
    plan = load_json(work_dir(video) / "plan.json")
    s, words = plan["stats"], plan["words"]
    print(f"TITLE: {plan.get('title')}   LANGUAGE: {plan.get('language')}")
    print(f"{s['source_duration']}s -> {s['edited_duration']}s (-{s['removed_pct']}%), {s['cuts']} cuts, "
          f"removed words {s['removed_words']}, {s['zooms']} zooms, {s['broll']} b-roll")
    print(f"LAYOUT: {plan['layout']['mode']} content={plan['layout'].get('content')} "
          f"preview: {work_dir(video) / 'layout_preview.png'}")
    print("\nCAPTIONS  ([index]word, *=highlighted, ~=heard differently and corrected)")
    for g in plan["captions"]:
        ws = []
        for i in g["words"]:
            w = words[i]
            if w.get("junk"):
                continue
            mark = ("*" if w.get("kw") or w.get("emph") else "") + ("~" if w.get("heard") else "")
            ws.append(f"[{i}]{mark}{w['text']}")
        print(f"{g['start']:7.2f}  {' '.join(ws)}")
    print("\nREMOVED WORDS")
    for i, w in enumerate(words):
        if w.get("removed") or w.get("junk"):
            print(f"  [{i}] {w['start']:7.2f}s {w.get('removed') or 'hidden from captions'}: {w['text']}")
    print("\nCUTS (source time)")
    for c in plan["cuts"]:
        print(f"  {c['start']:7.2f}s  {c['dur']:5.2f}s  {','.join(c['reasons'])}  {c['text'][:50]}")
    print("\nZOOMS")
    for z in plan["zooms"]:
        print(f"  {z['start']:7.2f}-{z['end']:7.2f}  {z['type']:6s} x{z['level']}  focus=({z['fx']},{z['fy']})  {z.get('word', '')}")
    print("\nB-ROLL")
    for k, b in enumerate(plan["broll"]):
        print(f"  #{k} {b['start']:7.2f}-{b['end']:7.2f}  {b['id']:16s} on \"{b['word']}\"")


def cmd_fix(a):
    """Correct caption words, hide garbled ones, drop B-roll, set the title."""
    video = Path(a.video).resolve()
    path = work_dir(video) / "plan.json"
    plan = load_json(path)
    words = plan["words"]
    for item in a.words:
        idx, _, text = item.partition("=")
        w = words[int(idx)]
        w.setdefault("heard", w["text"])
        if text.strip():
            w["text"] = text.strip()
            w.pop("junk", None)
        else:
            w["junk"] = True
    for k in sorted(a.drop_broll, reverse=True):
        plan["broll"].pop(k)
    if a.title:
        plan["title"] = a.title
    save_json(path, plan)
    log(f"updated {path}")


def cmd_cards(a):
    from .graphics import make_cards
    cfg = load_config()
    items = (yaml.safe_load((REPO / cfg["broll"]["library"]).read_text(encoding="utf-8")) or {}).get("items", [])
    for p in make_cards(cfg["brand"], items, REPO / "assets" / "broll" / "cards"):
        print(p)


def cmd_watch(a):
    inbox = Path(a.inbox).resolve()
    done, failed = inbox / "done", inbox / "failed"
    for d in (inbox, done, failed):
        d.mkdir(parents=True, exist_ok=True)
    sizes = {}
    log(f"watching {inbox} every {a.interval}s (Ctrl+C to stop)")
    while True:
        for f in sorted(inbox.iterdir()):
            if f.suffix.lower() not in VIDEO_EXTS or not f.is_file():
                continue
            size = f.stat().st_size
            if sizes.get(f) != size:  # still uploading / syncing
                sizes[f] = size
                continue
            log(f"new upload: {f.name}")
            try:
                cfg = load_config(f, a.config, parse_set(a.set))
                wd = work_dir(f)
                render_plan(build_plan(f, cfg, wd), cfg, wd)
                dest = done
            except Exception:
                traceback.print_exc()
                dest = failed
            for p in (f, f.with_suffix(".yaml")):
                if p.exists():
                    shutil.move(str(p), str(dest / p.name))
            sizes.pop(f, None)
        time.sleep(a.interval)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vedit", description="Nagarajan's video editing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", action="append", default=[], help="extra YAML settings file")
        sp.add_argument("--set", action="append", default=[], help="override, e.g. zoom.punch_in=1.2")

    r = sub.add_parser("run", help="transcribe, plan and render one video")
    r.add_argument("video")
    r.add_argument("--title")
    r.add_argument("--only", choices=["youtube", "instagram"])
    r.add_argument("--stop-after", choices=["plan"])
    common(r)
    r.set_defaults(fn=cmd_run)

    rr = sub.add_parser("render", help="render again from the saved (possibly edited) plan.json")
    rr.add_argument("video")
    rr.add_argument("--title")
    rr.add_argument("--only", choices=["youtube", "instagram"])
    common(rr)
    rr.set_defaults(fn=cmd_render)

    lo = sub.add_parser("layout", help="detect the screen/face layout and write layout_preview.png")
    lo.add_argument("video")
    common(lo)
    lo.set_defaults(fn=cmd_layout)

    rv = sub.add_parser("review", help="print the plan: numbered caption words, cuts, zooms, B-roll")
    rv.add_argument("video")
    rv.set_defaults(fn=cmd_review)

    fx = sub.add_parser("fix", help="edit the saved plan: fix INDEX=TEXT (empty TEXT hides the word)")
    fx.add_argument("video")
    fx.add_argument("words", nargs="*", help='e.g. 57=patterns 80=Gartley 91=')
    fx.add_argument("--drop-broll", type=int, action="append", default=[], help="B-roll number to remove")
    fx.add_argument("--title")
    fx.set_defaults(fn=cmd_fix)

    c = sub.add_parser("cards", help="redraw the B-roll cards from assets/broll/library.yaml")
    c.set_defaults(fn=cmd_cards)

    w = sub.add_parser("watch", help="process every video dropped into an inbox folder")
    w.add_argument("--inbox", default=str(REPO / "inbox"))
    w.add_argument("--interval", type=int, default=15)
    common(w)
    w.set_defaults(fn=cmd_watch)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
