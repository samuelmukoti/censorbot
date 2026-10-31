# CLAUDE.md

Guidance for Claude Code working in this repository.

## What it is

CensorBot removes profanity from the dialogue of a video file and writes a new
file with a censored audio track (first, default) plus the untouched original.
Package: `censorbot/`. `censor.py` is a thin backward-compatible entry point.

## Pipeline (censorbot/cli.py: `process`)

1. **Probe** (`media.probe`): audio layout, subtitle tracks.
2. **Analysis audio** (`media.extract_analysis_audio`): 16 kHz mono of the
   *centre channel* when the layout has one (dialogue lives there), else a downmix.
3. **Transcribe** (`transcribe.py`): MLX Whisper on Apple Silicon, faster-whisper
   on CUDA/CPU, word timestamps, `condition_on_previous_text=False` (stops
   long-film hallucination loops). Cached in `~/.cache/censorbot/` keyed by file
   size, mtime, stream, model and language.
4. **Match** (`words.py`): whole-token matching with globs, plurals, phrases,
   allowlist and three strictness tiers. Never substring matching: that muted
   "hello", "class", "Michelle".
5. **Place** (`detect.py` + `align.py`): each hit is re-timed by CTC forced
   alignment (MMS_FA by default; `--aligner base` = wav2vec2-base, worse on film audio) on a window of the hit plus 3 words either side, then the
   edges are grown to the real voice onset/offset without crossing neighbour
   words. Falls back to Whisper times (+/-120 ms) if alignment is implausible.
6. **Subtitles** (optional, `-s` or embedded text track): cues with hits that
   the transcript did not catch are aligned to the audio to place the word.
   Image subtitles (PGS) are skipped.
7. **Render** (`render.py`): streams the original track at native rate and
   layout through NumPy. Raised-cosine fades (20 ms). Strategies:
   `centre` (cut FC; Demucs removes dialogue spill/reverb from the FL/FR and
   surround pairs; without Demucs, subtract the correlated FC copy from FL/FR),
   `separate` (stereo: Demucs vocals on a 1 s-margin window, subtract),
   `all` (mono or `--scope all`). Chunk boundaries never fall inside a cut.
8. **Mux** (`media.mux`): video and original audio stream-copied; censored track
   is a:0 and default, original is a:1. Censored track is AAC at 2x source
   bitrate (a 288k re-encode measured -24 dB waveform error, 640k -38 dB).

## Commands

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev,separation]"
.venv/bin/python -m pytest              # fast unit tests
.venv/bin/python -m pytest -m slow      # end-to-end: macOS say + Whisper
.venv/bin/censorbot -i in.mp4 -o out.mp4 --report r.json
.venv/bin/python tools/verify.py out.mp4 r.json --reel reel.m4a   # leak / collateral / sync check
.venv/bin/black censorbot censor.py tests && .venv/bin/flake8 censorbot censor.py tests --max-line-length=120 --extend-ignore=E203,W503
```

## Gotchas

- On Sam's Mac `~/.cache/huggingface` is a symlink to an external drive the
  terminal may not be allowed to read. Set `HF_HOME` to a local folder
  (`.cache/hf` in the repo is gitignored) if model loading fails with
  "Operation not permitted".
- Test media lives on the Plex NUC (`ssh sam@192.168.86.10`, `/media/disk1/media/movies`);
  copy into `test_videos/` (gitignored).
- Never re-encode or alter the original audio track; never clip samples outside cuts.
- Judging leaks: Whisper fills a censored gap from context (it "hears" the word
  even when the cut is digital silence), so test the cut region on its own, as
  `tools/verify.py` does. Window-level ASR gives false leaks.
