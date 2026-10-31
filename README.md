# CensorBot

[![CI/CD](https://github.com/samuelmukoti/censorbot/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelmukoti/censorbot/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/censorbot.svg)](https://pypi.org/project/censorbot/)
[![Docker Pulls](https://img.shields.io/docker/pulls/samuelmukoti/censorbot.svg)](https://hub.docker.com/r/samuelmukoti/censorbot)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://www.buymeacoffee.com/smukoti)

CensorBot removes profanity from the dialogue of your movies and leaves everything else alone. The word disappears; the score, the explosions and the next line of dialogue carry on as if nothing happened. The output file gets a clean audio track (selected by default) alongside the untouched original, so you can switch back at any time.

## Legal Disclaimer

**CensorBot is intended for personal, educational, and lawful use only.**

- **Media Ownership Required**: You must legally own or have proper authorization to use any media processed with this tool. CensorBot is designed for personal media libraries, legally purchased content, and authorized educational materials.
- **No Endorsement of Piracy**: This project and its author(s) **do not endorse, promote, or support piracy** or copyright infringement in any form.
- **User Responsibility**: Users are solely responsible for ensuring they have the legal right to modify and use any media files processed with CensorBot. This includes compliance with copyright laws, terms of service, and licensing agreements.
- **No Warranty**: This software is provided "as is" without warranty of any kind. The author(s) are not liable for any misuse, legal consequences, or damages arising from the use of this tool.

By using CensorBot, you acknowledge that you understand and agree to these terms, and that you will use the software responsibly and in accordance with applicable laws.

## Why it sounds clean

Most censoring tools mute the whole soundtrack for a second or more around each word, or worse, the whole subtitle line. You hear the music drop out and you know exactly what was said. CensorBot 0.1 was rebuilt around four ideas:

1. **Only the voice is removed.** In a 5.1 or 7.1 mix the dialogue lives in the centre speaker, so CensorBot cuts the centre and uses vocal separation (Demucs) to remove the dialogue spill and reverb from the other speakers. Music and effects keep playing. Stereo sources use the same separation on the whole mix.
2. **Word-accurate timing.** Whisper finds the words; a forced aligner (Meta's MMS) then re-times each censored word to about 20 ms, and the edges are snapped to where the voice actually starts and stops. Typical cut: 0.4 s, not 2 s.
3. **No clicks.** Every cut fades in and out over 20 ms, and the censored track is rendered from the original audio (same sample rate, same channels) and encoded at twice the source bitrate, so the second encode stays transparent.
4. **Whole-word matching.** "Hell" is censored, "hello" is not. "Ass" is censored, "class", "glass" and "assistant" are not. Phrases, plurals, globs and an allow list are supported.

## Installation

```bash
pip install "censorbot[separation]"     # or: pipx install "censorbot[separation]"
```

FFmpeg is required (`brew install ffmpeg`, `sudo apt-get install ffmpeg`, or ffmpeg.org on Windows).

The `separation` extra installs Demucs. Without it CensorBot still works: 5.1 sources get a centre-only cut and stereo sources fall back to silencing the whole mix for the length of the word.

Hardware is detected automatically: Apple Silicon uses MLX (Metal), NVIDIA GPUs use CUDA through faster-whisper, everything else runs on the CPU.

### Docker

```bash
docker build -t censorbot .
docker run -v "$(pwd):/app" censorbot -i input.mp4 -o output.mp4

# NVIDIA
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t censorbot:cuda .
docker run --gpus all -v "$(pwd):/app" censorbot:cuda -i input.mp4 -o output.mp4
```

## Usage

```bash
censorbot -i movie.mp4 -o movie.clean.mp4            # the usual case
censorbot -i movie.mp4 --dry-run                     # list what would be cut, write nothing
censorbot -i movie.mkv -o movie.clean.mkv --mode beep
censorbot -i movie.mp4 -o out.mp4 -s movie.srt       # subtitles catch anything the transcript missed
censorbot -i movie.mp4 -o out.mp4 --strictness low   # strongest words only
```

The output container follows the extension of `-o`. Video, chapters and the original audio are copied without re-encoding.

### Options

| Option | What it does | Default |
|---|---|---|
| `-i, --input` | Input video | required |
| `-o, --output` | Output video | required unless `--dry-run` |
| `--strictness` | `low`: strongest words and slurs. `medium`: adds insults (ass, bitch, dick, ...). `high`: adds damn, hell, crap | `high` |
| `-w, --wordlist` | Extra words file (see below) | none |
| `-s, --subtitles` | SRT used to catch words the transcript missed. Embedded text subtitles are used automatically | embedded, if any |
| `--mode` | `mute` or `beep` | `mute` |
| `--scope` | `voice` removes only the dialogue; `all` silences every channel | `voice` |
| `--model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` | `turbo` |
| `--padding` | Extra seconds around each word | 0.03 before, 0.05 after |
| `--fade` | Fade length in seconds | 0.02 |
| `--single-audio` | Drop the original track | off |
| `--audio-stream` | Which audio track to censor | 0 |
| `--audio-codec`, `--audio-bitrate` | Censored track encoding (`flac` for lossless in MKV) | AAC, 2x source bitrate, up to 640k |
| `--no-separation` | 5.1+: centre-only cut (faster, some spill remains) | off |
| `--aligner` | Forced-alignment model: `mms` or `base` (smaller, less accurate on film audio) | `mms` |
| `--no-align` | Skip forced alignment (faster, less precise) | off |
| `--report` | Write a JSON list of every cut | off |
| `--export-srt` | Write a censored SRT built from the transcript | off |
| `--stats` | Print a word count summary | off |
| `--beep-file` | Your own beep sound | 1 kHz tone |
| `--config` | YAML file of option defaults (keys are option names) | none |

### Word lists

```text
# extra.txt
frick*          # glob: fricking, fricked, ...
jesus christ    # a phrase
!hell           # never censor this word
```

"Jesus Christ" is not censored by default, because plenty of films use it reverently. Add it to a word list if you want it.

### Which audio track plays?

The censored track is the first audio track and is marked default, so Plex, Jellyfin, VLC and most TVs pick it automatically. The original is the second track, named after the source.

## Measured results

Tested on two 1080p Blu-ray rips with 5.1 audio on an Apple M4 Max:

| | Deadpool 2 (2:14) | Guardians of the Galaxy (2:01) |
|---|---|---|
| Words censored | 195 in 184 cuts | 31 in 31 cuts |
| Total dialogue removed | 90 s | 14 s |
| Median cut length | 0.41 s | 0.38 s |
| First run (transcription included) | 9.5 min | 8.5 min |
| Re-run (transcript cached) | 3.1 min | 2.1 min |

Transcripts are cached in `~/.cache/censorbot/`, so changing the word list, strictness or mode and running again skips the slow step.

`tools/verify.py` checks an output the hard way: for every cut it re-transcribes the censored audio on its own to see whether the word is still recognisable, checks the clean words either side survived, and confirms the censored track is sample-aligned with the original. It can also write a listening reel (original, then censored, for every cut).

## Troubleshooting

- **"ffmpeg not found"**: install FFmpeg and make sure it is on your PATH.
- **Model download fails with "Operation not permitted"**: your Hugging Face cache (`~/.cache/huggingface`) points somewhere your terminal cannot read. Set `HF_HOME` to a local folder.
- **A word was missed**: run with `--dry-run --stats` to see what was found. Supplying subtitles (`-s`) catches words Whisper misheard. A bigger model (`--model large-v3`) helps with heavy accents.
- **Image subtitles (PGS) are ignored**: only text subtitles can be read. Extract or download an SRT and pass it with `-s`.
- **Out of memory on CPU**: use a smaller model, for example `--model small`.

## FAQ

**Does it still download subtitles from OpenSubtitles?** No. The provider integration was unreliable and silently returned empty files. Detection now comes from the audio itself; subtitles are an optional cross-check.

**How do I switch back to the original audio?** Pick the second audio track in your player.

**Will it work on stereo files?** Yes. With the `separation` extra the voice is separated from the music for each cut; without it the whole mix is silenced for the length of the word.

**Other languages?** Pass `--language` and a word list for that language. The default word list is English.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

## Support the Project

If you find this tool useful, consider buying me a coffee.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/smukoti)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper), [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- [MMS forced alignment](https://pytorch.org/audio/stable/pipelines.html) and wav2vec 2.0 via torchaudio
- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation
- [FFmpeg](https://ffmpeg.org/) for everything audio and video
- [pysrt](https://github.com/byroot/pysrt) for subtitles
- [CleanVid](https://github.com/clean-vid) for the original subtitle-based idea

## Contributing

Pull requests are welcome. Run `pytest` (and `pytest -m slow` on a Mac) before submitting, and format with `black`.
