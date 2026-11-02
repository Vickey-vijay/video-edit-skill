---
name: nagarajan-video-edit
description: Edit Naga Rajan's (Nagarajan's) trading-class recordings into finished MP4s for YouTube (16:9) and Instagram Reels (9:16). Cuts silences, Tamil/English fillers, repeats and stutters using a word-timestamped transcript, adds zoom-ins/zoom-outs, brand-styled captions with highlighted keywords, and B-roll concept cards. Use whenever Vickey or Nagarajan uploads or points at a new class video, says "edit this video", "make a reel / YouTube video from this class", "process the inbox", or asks to re-render or tweak an edit (captions, zoom, B-roll, title).
---

# Nagarajan video edit

Repo: `{{REPO}}`
Python: `%USERPROFILE%\.video-edit-skill\venv\Scripts\python.exe` (models, work files, venv live in
`%USERPROFILE%\.video-edit-skill`, outside OneDrive). Run commands from the repo folder:

```
.\vedit.cmd <command> ...
```

Outputs land in `output/<video name>/`: `<name>_youtube.mp4`, `<name>_instagram.mp4` (plus
`_instagram_partN.mp4` when the edit is over 3 minutes), `<name>.srt`, preview contact sheets,
`report.html`, `plan.json`.

## Procedure for each new video

1. **Plan** (transcribe + decide the edit, no rendering yet):
   ```
   .\vedit.cmd run "<video>" --title "<topic>" --stop-after plan [--config presets/meet-edge-share.yaml]
   ```
   - Pick the title from what the class is about (ask if unclear). It shows on screen.
   - Use `presets/meet-edge-share.yaml` for Google Meet recordings where he shares his whole
     screen with Edge open (TradingView / PDF): it hides the browser bars and the taskbar.
   - First run on a machine downloads Whisper large-v3 (~3 GB). Transcription of a 3-minute
     clip takes ~4 minutes on the RTX 3050; a 20-minute class roughly 20-25 minutes.

2. **Check the layout**: open `layout_preview.png` (path printed by `review`). The orange box
   is what fills the main panel, the purple box is the facecam crop. If wrong, set
   `layout.screen` / `layout.face` / `layout.screen_trim` in a sidecar YAML next to the video
   (`<video>.yaml`) and re-run step 1.

3. **Review the plan**:
   ```
   .\vedit.cmd review "<video>"
   ```
   Read every caption line. Whisper is weak on Tamil-English speech, so fix it:
   - Wrong word, clear from context (English trading term spelled in Tamil script, misheard
     ratio): `.\vedit.cmd fix "<video>" 57=patterns 80=Gartley`
   - Garbled and not recoverable from context: hide it, `.\vedit.cmd fix "<video>" 91=`
   - **Never invent words he did not say.** If unsure, hide rather than guess.
   - A mishearing that will recur in future videos: add it to `config/corrections.yaml`
     (only if the Tamil spelling always means that English word).
   - Check the removed-words list. Tamil particles like ஆ / ஏ after a number are grammar,
     not fillers; if something meaningful was cut, remove it from `config/fillers.yaml` and
     re-run step 1.
   - Drop an irrelevant B-roll card: `.\vedit.cmd fix "<video>" --drop-broll 2`

4. **Render**:
   ```
   .\vedit.cmd render "<video>"
   ```
   About 1.5x real time per output on this PC (NVENC). Renders YouTube and Instagram.

5. **Verify before reporting**: look at `output/<name>/<name>_youtube_preview.jpg` and
   `_instagram_preview.jpg`; grab a frame or two where B-roll / zooms happen
   (`ffmpeg -ss <t> -i <mp4> -frames:v 1 x.png`) and check them. Open `report.html` for the
   cut list. Then send the MP4s and the report to the user (SendUserFile), saying what was
   checked and what was not (e.g. caption accuracy in Tamil is reviewed but not perfect).

## Automatic mode (no review)

`.\vedit.cmd watch` (or double-click `watch.cmd`) processes every video dropped into `inbox/`
with the default settings and moves the source to `inbox/done/`. Same pipeline without step 3,
so captions keep Whisper's mistakes. Use it for speed; use the procedure above for quality.

## Knobs (config/default.yaml, override per video with `--set key=value` or a sidecar YAML)

| What | Setting |
|---|---|
| How tight the cutting is | `cuts.max_pause` (0.45 s), `cuts.pad_before/after` |
| Soft fillers (so, basically, அது வந்து...) | `cuts.soft_fillers: off / isolated / all` |
| Zoom strength / frequency | `zoom.punch_in`, `zoom.min_interval`, `zoom.max_per_minute`, `zoom.enabled` |
| Zoom-out reveals | `zoom.zoom_out_reveal`, `zoom.reveal_from` |
| Caption size of each line | `captions.max_words`, `captions.max_chars` |
| B-roll frequency | `broll.min_interval`, `broll.max_per_minute`, `broll.enabled` |
| Brand colours / fonts / name | `config/brand.yaml` |
| Highlighted keywords | `config/keywords.yaml` |
| B-roll cards and their keywords | `assets/broll/library.yaml`, then `.\vedit.cmd cards` |

## Things to keep in mind

- Harmonic ratios on the B-roll cards come from his HARMONIC PATTERN.pdf as seen in the
  recordings; if he teaches different values, edit `assets/broll/library.yaml`.
- The style tokens come from his Elliott Wave guide (dark navy, cyan accent, Bricolage
  Grotesque / Atkinson Hyperlegible / JetBrains Mono). If a Claude Design project for him
  exists, update `config/brand.yaml` from it.
- Other speakers in the call (questions from students) are kept; only dead air and
  disfluencies are cut.
