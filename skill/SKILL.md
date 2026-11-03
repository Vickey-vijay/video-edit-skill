---
name: nagarajan-video-edit
description: Edit Naga Rajan's (Nagarajan's) Tamil trading videos into finished MP4s. Phone talking-head / screen-pointing clips (vertical) become a 1080x1920 Reel/Short with his false starts cut, English captions, a cut-out of him over a topic-themed animated background, a hook, stat stickers and sound design. Meet recordings (16:9) become YouTube 16:9 plus Instagram 9:16. Use whenever Vickey or Nagarajan uploads or points at a class/market video, says "edit this video", "make a reel / short from this", "process the inbox", or asks to re-render or tweak an edit (cuts, captions, graphics, theme, hook).
---

# Nagarajan video edit

Repo: `{{REPO}}`. Python venv, Whisper and matting models, and per-video work folders live in
`%USERPROFILE%\.video-edit-skill` (outside OneDrive). Run from the repo folder in PowerShell:
`.\vedit.cmd <command> ...`. In Git Bash, `vedit.cmd` mangles paths with spaces (WhatsApp files have
them): call the venv python directly instead: `PYTHONPATH=<repo> <venv>\Scripts\python.exe -m vedit ...`.
Outputs land in `output/<video name>/`.

**How he speaks (why the cutter works the way it does).** He is Tamil with English finance words. His
real problem is not "um": it is false starts. He begins a sentence, stops (sometimes mid-word), pauses
0.4-4 s and says it again, sometimes three attempts in a row; only the last is complete. Whisper often
never writes the aborted attempt and its timestamps drift, so cuts are decided from the audio as well as
the text. Tamil particles such as ஆ / ஏ are grammar, not fillers.

## Phone footage (vertical, 9:16): the Reel pipeline

1. **Plan**: `run "<video>" --title "<topic>" --stop-after plan`. First run downloads Whisper large-v3
   (~3 GB). Transcribing 2.5 min takes ~4 min on the RTX 3050.
2. **Review the cuts**: `restarts "<video>"` prints every speech phrase (P#, times, text, `*` = cut) and
   every false-start candidate with its evidence (text coverage, audio similarity cost: repeats are
   usually 27-40, unrelated phrases ~50; TRUNCATED = cut off mid-word). Read ALL phrases, not just the
   candidates. Decide, then record by time (numbers shift between plans):
   `decide "<video>" --cut 3 --keep 5 --cut-at 89.2 --keep-at 40.1 --cut-range 106.1-107.7`
   and re-run step 1. Judge by meaning: keep the last, complete attempt; keep a sentence that is
   genuinely different even if a few words match; a very low audio cost (<30) beats a text mismatch,
   because Whisper may not have written the attempt down.
3. **Listen to the edit**: `qa "<video>"` re-transcribes the edited audio and looks for surviving false
   starts and long pauses. Fix what it finds (step 2) until it is clean. Never skip this: reading the
   transcript is not the same as hearing the result.
4. **English captions** (he speaks Tamil; viewers outside Tamil Nadu need English; no Tamil on screen):
   `cues "<video>"` prints each kept phrase with the Tamil and a second decode. Translate BY MEANING,
   cross-checking context and the picture (Whisper writes English words phonetically in Tamil script,
   e.g. "திருமணமான விஸினிஸ்" = "established business"). Write `id|English` lines to a file and run
   `translate "<video>" file.txt`. Cue ids are positional: if a re-plan adds or removes a phrase, remap the
   ids. Never invent what he did not say; flag unclear numbers to the user.
5. **Graphics spec**: write `graphics.yaml` in the video's work folder (path printed by `restarts`;
   worked examples in `examples/`). Times are SOURCE seconds. Choose: `theme`/`themes` (gold for gold,
   golden ratio and money; market for price action; studio for calm), a `hook` (big word behind him +
   two kicker lines), `stats` (sticker with a figure), `tags` (short statements) and a `cta`. Every figure
   must be one he said and one you can verify (on screen or in the transcript).
6. **Look before rendering**: `preview "<video>" 0.8 8.5 40` renders those edited-time seconds to PNG in
   the output folder. Open them and judge them honestly (hook legible? sticker off his face? caption
   readable? matte clean?). Fix and repeat.
7. **Render**: `render "<video>"` (matting is cached; first matte takes ~10 min CPU per clip; the
   composite runs ~1.5 frames/s, so a 90 s reel takes ~25 min: run it in the background). Then verify:
   duration, a few extracted frames, audio level (-14 LUFS), and send the MP4 plus the `_en.srt`.

Footage of him pointing at a screen is never cut out (the chart would vanish); it gets the grade, a slow
push-in, the hook with a scrim and the stickers. Talking-head frames get the matte and background.

## Meet / 16:9 recordings (older pipeline)

`run "<video>" --title "<topic>" --config presets/meet-edge-share.yaml --stop-after plan`, then
`review` / `fix` for Tamil captions, then `render`. Produces YouTube 16:9 and Instagram 9:16.

## Automatic mode

`watch.cmd` processes whatever is dropped into `inbox/` with defaults: no review, no English captions,
no graphics spec. Fine for speed, not for quality.

## Things to keep in mind

- This is a coding-agent pipeline, not a human editor. State honestly what was verified (frames, audio
  check) and what was not (the Tamil translation has not been checked by a Tamil speaker).
- No stock footage, music or AI image generation is available; backgrounds are procedural, sound effects
  are synthesised. Put own clips in `assets/broll/` and music is not mixed in.
- Patching files from a shell heredoc: avoid backslashes (a `\n` becomes a real newline); use the file
  tools for anything containing them.
- Pillow here has no Tamil shaping: on-screen text is English only.
- Colours/fonts: gold on near-black, Anton + Montserrat ExtraBold + Playfair italic. The earlier navy/cyan
  look was rejected as "AI-looking"; do not bring it back.
