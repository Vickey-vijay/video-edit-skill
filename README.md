# video-edit-skill

Turns Naga Rajan's class recordings into finished videos for YouTube (1920×1080) and
Instagram Reels (1080×1920), the same way every time:

1. **Transcribe** with Whisper large-v3 (Tamil, keeping English trading terms in English),
   word timestamps, then a repair pass that retries weak sentences and fills speech Whisper
   skipped.
2. **Cut** dead air (pauses over 0.45 s), hesitation fillers (um, uh, ம்ம், அ...), repeated
   words ("இந்த இந்த", "the the", "so we so we") and stutters ("pa- pattern"). Audio cuts get
   6 ms fades so they don't click; speech Whisper missed is never cut.
3. **Zoom**: eased punch-ins on keywords and emphasised words, aimed at where he is drawing
   or pointing, plus zoom-out reveals at the start and after topic breaks.
4. **Captions**: 3–4 words at a time, the spoken word lit in cyan, keywords and numbers in
   orange, Tamil words in Noto Sans Tamil, English in Bricolage Grotesque. Also an `.srt`.
5. **B-roll**: when he names a pattern (Gartley, Crab, Cypher, Fibonacci, impulse wave...),
   a concept card in the brand style slides in beside the screen share, never over the chart.
6. **Layout**: finds the shared screen and his facecam tile in a Meet recording and rebuilds
   the frame in the style of his Elliott Wave guide (navy grid, cyan accent).
7. **Render** with NVENC, loudness normalised to −14 LUFS; Instagram edits over 3 minutes are
   split into equal parts at sentence ends.

## Setup (once per PC)

Needs Windows, Python 3.10+, ffmpeg (full build) on PATH, and an NVIDIA GPU (falls back to
CPU, much slower).

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This creates the Python environment in `%USERPROFILE%\.video-edit-skill` (outside OneDrive),
installs the Claude Code skill `nagarajan-video-edit`, and draws the B-roll cards.

## Use

```
.\vedit.cmd run "C:\path\class.mp4" --title "Harmonic Patterns"                  # everything
.\vedit.cmd run "C:\path\class.mp4" --title "..." --stop-after plan               # plan only
.\vedit.cmd review "C:\path\class.mp4"                                            # read the plan
.\vedit.cmd fix "C:\path\class.mp4" 57=patterns 91=                               # fix / hide caption words
.\vedit.cmd render "C:\path\class.mp4"                                            # render the saved plan
.\vedit.cmd layout "C:\path\class.mp4"                                            # check crop boxes
.\vedit.cmd watch                                                                 # auto-edit everything put in inbox\
```

In Claude Code, just say "edit this class video: <path>" and the skill runs the
plan → review → render procedure, fixing Tamil captions before rendering.

Settings: `config/default.yaml` (every knob is commented), `config/brand.yaml` (colours,
fonts, name), `config/keywords.yaml`, `config/fillers.yaml`, `config/corrections.yaml`,
`assets/broll/library.yaml`. Override per video with a sidecar `<video>.yaml` next to the
video, `--config file.yaml`, or `--set zoom.punch_in=1.2`.

## Layout of the code

| File | Does |
|---|---|
| `vedit/transcribe.py` | Whisper with word timestamps, loop removal, best-of retries, gap filling |
| `vedit/plan.py` | fillers / repeats / stutters, silence cuts, timeline remap, captions, zooms, B-roll |
| `vedit/analyze.py` | Meet layout detection, auto-trim of static bars, face position, motion focus |
| `vedit/render.py` | frame-exact cut, audio splice + loudness, layout geometry, ffmpeg compose |
| `vedit/captions.py` | ASS captions with active-word / keyword colours, SRT |
| `vedit/graphics.py` | background frame, panel holes, name chip, B-roll cards (Pillow) |
| `vedit/cli.py` | commands above, watch folder |
