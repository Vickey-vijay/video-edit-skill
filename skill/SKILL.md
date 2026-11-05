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
   (~3 GB). Transcribing 2.5 min takes ~4 min on the RTX 3050. The plan step also checks the hand-held
   footage (see "Hand-held footage" below): the log line `hand-held footage: ...` lists the swings, blur and
   camera interludes it found; `footage "<video>"` prints them with what the edit does about each.
1b. **Let a model that can listen judge the content** (`understand`). The sound-based cutter below only
   finds repeats and pauses; it cannot tell that a sentence stalled, that the same point is made twice, or
   that a stretch is a filler or an aside, and the Whisper Tamil text is too poor for me to judge meaning.
   Google Gemini takes the raw video with its Tamil audio and judges every speech phrase (what he really
   says, complete thought / false start / stalled / filler / repeat of another phrase / aside, keep or cut,
   why). The cuts are still made on the audio-detected phrase windows, never on Gemini's own timestamps.
   `understand "<video>" --prompt-only` writes the prompt (no upload); with a key (environment variable
   `GEMINI_API_KEY` or `Gemini_API`, or the key alone in `%USERPROFILE%\.video-edit-skill\gemini.key`)
   `understand "<video>"` uploads the clip, asks, and deletes the upload again. A FREE Google key has no
   quota for the Pro models (429, limit 0): the command then falls back to `understand.fallback_model`
   (Gemini 3.6 Flash, verified on clip 1) and a paid key gets `gemini-pro-latest`. Gemini's raw reply is
   kept in `understand_reply_raw.txt` so a parsing problem never costs another upload. When applying,
   hand trims made earlier inside phrases that Gemini keeps are preserved. Either way the answer is stored in `understanding.json` and printed
   as a table. READ IT: check each `[CHECK: ...]` flag, the repeats (is the dropped take really worse?),
   the cut phrases, and read only the kept English in order as one script to see whether it flows.
   `--refine` adds a text-only second pass (best takes, dangling sentences). When it is right,
   `understand "<video>" --apply --replan` writes cut_at / keep_at / trims into `decisions.json` (the old
   file is kept) and rebuilds the plan; later `understand "<video>" --captions` fills English captions from
   the analysis. Only send a client's raw clip when the client has said yes (the user approved this for
   Nagarajan's clips on 2026-11-05).
   Lessons from clip 2: (a) the Flash models are sometimes overloaded (503); `--model gemini-2.5-flash` or
   another from `--list-models` retries, and every upload is deleted afterwards. (b) A model can answer
   validly but on the wrong phrases (3.6 Flash gave P07 the sentence of P37, invented "EV/EBITDA" and a
   line for a one-word phrase): put its English next to Whisper's Tamil for every phrase where two runs
   disagree and trust the run whose English fits the Tamil. (c) A short phrase that Whisper heard no word in
   (empty text) is dropped by the planner even when Gemini keeps it: add the word to `transcript.json`
   (keep a backup) and run `run "<video>" --stop-after plan` again; the cue then exists too.
   (d) `--captions` writes the whole phrase's English even where a `trim` removed its tail: edit
   `captions_en.json` by hand and read it against the final cues.
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
   worked examples in `examples/`). Times are SOURCE seconds. Choose: `theme`/`themes` (they swap ONE
   mid-layer of the same "After Hours" room: `market` = frosted glass with defocused price levels and one
   price path, for price action and charts; `gold` = a brass banknote rosette, for money, valuation and
   golden ratio; `studio` = light and haze only, for calm/mindset), a `hook` (`behind` = the small brass
   kicker such as the ticker, `lines` = the two-line headline, `accent` = which line is brass), `stats`
   (label: kicker, big figure, optional `sub`), `tags` (short statement on a label, above his head by
   default), `cards` (words on the background while the picture is unusable: one per `card` the footage
   check reports) and a `cta` (an italic serif question, a brass line, a quiet kicker). Every figure must be
   one he said and one you can verify (on screen or in the transcript). Do not put detail where his head is.
6. **Look before rendering**: `preview "<video>" 0.8 8.5 40` renders those edited-time seconds to PNG in
   the output folder. Open them and judge them honestly (hook legible? sticker off his face? caption
   readable? matte clean?). Fix and repeat.
7. **Render**: `render "<video>"`. Order of work (all cached in the work folder): conform to constant
   frame rate -> matte on the GPU (about 2 min per clip) -> **enhance** (super-resolution, steadier picture,
   cleaner cut-out; about 5 frames/s on the RTX 3050, so ~7-9 min for a 90 s reel) -> composite in 4
   parallel pieces. Run it in the background. Then verify from the real MP4: duration, frames (zoom into
   his hair and the sleeve edge), audio level (-14 LUFS), the `footage` report (nothing bad left in the
   edit), and send the MP4 plus the `_en.srt`. Send the full-quality file, not a re-encoded phone copy.

Footage of him pointing at a screen is never cut out (the chart would vanish); it gets the grade, a slow
push-in, the hook with a scrim and the stickers. Talking-head frames get the matte and background.

## Hand-held footage and picture quality (what the render does for you)

He films himself by hand. Four things follow, and each is handled; check each in the output.

- **Swings, blur, exposure jolts** (`footage.py`): the camera is tracked over the whole clip. A swing is a
  step of more than ~14 px a frame at 480 wide (a slow deliberate pan is up to ~10 and is left alone), or
  fast and smeared; blur is sharpness falling below ~55% of its recent level for 4+ frames. Stretches with
  no speech are cut. With speech: a **card** (his words over the animated background, for swings and for
  camera interludes) or a **hold** (last good frame, slowly pushed in, for blur in the middle of a shot).
  If a soft stretch starts a piece there is nothing to hold, so the soft frames stay (cutting would lose
  words): listed as `kept soft`. A short stretch with nobody in frame between two talking shots is a camera
  interlude (he moved or put down the phone), not a cutaway.
- **Shake on screen footage**: camera path smoothed per shot (sigma 15 frames, at most 3% of the width), one
  warp, slight crop-in (`short.enhance.stab_zoom`). Talking shots are not stabilised: the camera is steady
  there and his own movement must not be fought.
- **Soft picture**: Real-ESRGAN "general v3" (official weights, fp16) at 2.7x the 480x848 source, denoise
  blend for WhatsApp compression, a little plain upscale mixed back so skin is not waxy, temporal smoothing
  of fine detail so the sharpening does not shimmer. Needs torch + a CUDA GPU (`setup.ps1` installs it);
  without them the render falls back to the plain upscale and says so.
- **Hair, ears, halo of the old room**: RVM runs on the original footage; its low-resolution matte is
  re-snapped to the enhanced picture with a guided filter, the colour of the old room is removed from his
  edge pixels (Blur-Fusion foreground estimate), and stray blobs are dropped. Tune in `short.matte`.
- Output is 1080x1920 at ~16 Mbps (BT.709-tagged). A 4K file would only be bigger pixels: the source is
  480 px wide and Instagram / Shorts serve 1080.

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
- THE LOOK ("After Hours", rebuilt 2026-11-06 after the client said the old background was "like shit" for a
  1 lakh edit). Rejected earlier, do not bring back: (1) the navy/cyan/grid/rounded-translucent-card look
  ("AI-looking", round 1), (2) gold dust / sparkles / bokeh discs / light rays / scrolling candles on near-black
  ("a simple animation"). A frosted-glass-card and navy version of this rebuild was built and thrown away for
  being too close to (1). What ships: one calm room in warm graphite (#0A0A0B to #1E2328, no blue) with an amber
  light low right, a faint slate fill left, a soft key streak from the top left, very fine slow haze (2.6%);
  brass (#C9A15B) as the single accent, ivory (#EEF6FA) type. One world, one light, one slow camera: the
  camera drifts and steps a little at every edit cut (like a multi-camera shoot). Every motion is slower than
  1% of the width per second; nothing bounces or springs; type fades and slides. Overlays: letter-spaced
  Montserrat kickers, ivory headlines (accent line in brass) in the headroom above his head, stat cards and
  tags as solid near-black printed-looking labels with a brass hairline (sharp 8 px corners, no blur, no
  pills), plain white captions with a soft shadow over a darkened bottom, an italic Playfair question for the
  end card. No outlines, no tilt, no hard shadows. `short.palette: ink` switches to the blue-black variant;
  `fx.Panel(frost=True)` is the frosted variant. Edit the look in `vedit/backgrounds.py`, `vedit/motion.py`,
  `vedit/fx.py` (Panel, with_halo, finish). Themes swap one mid-layer: market (glass pane with defocused price
  levels and one price path), gold (a brass banknote rosette), studio (light and haze only).
- Setting him into the scene (`vedit/edge.py`, `Subject.layer` in `shorts.py`): the matte is made decisive
  (lo 0.32 / hi 0.88, then 1 px feather) because the phone clip's matte trails a moving head as a translucent
  band that shows the old room (a beige wooden wall!); edge pixels take their colour from inside him; skin-like
  pixels are pulled toward a healthy brown (the phone lights one cheek blue-grey) while hair, whites of the
  eyes and the hoodie are protected; hair edges are darkened; a little room light is wrapped onto the edge;
  a rim light from the key; a soft shadow and a warm glow behind the shoulders; then a finishing grade, a
  caption scrim, vignette and grain over everything. How to QA the edge: composite the cached layers over flat
  magenta and mid-grey (`diag/matte_diag.py` style) at 1:1 on the frames where the head moves fastest, and look
  at consecutive frames for flicker. Keep detail (price lines, rosette) away from his head: it fades out in an
  ellipse around the head, otherwise a line "grows out of his hair".
