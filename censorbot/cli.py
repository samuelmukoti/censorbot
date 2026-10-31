"""Command line entry point and the end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from . import __version__
from .media import (
    MediaError,
    default_bitrate,
    extract_analysis_audio,
    extract_text_subtitles,
    mux,
    probe,
    read_wav_mono,
    require_ffmpeg,
)
from .render import Cut, Renderer, merge_cuts
from .transcribe import MODEL_CHOICES, pick_backend, transcribe
from .words import LEVELS, Matcher, normalize

log = logging.getLogger("censorbot")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="censorbot",
        description="Remove profanity from the dialogue of a video file, cleanly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  censorbot -i movie.mp4 -o movie.clean.mp4
  censorbot -i movie.mkv -o movie.clean.mkv --mode beep
  censorbot -i movie.mp4 --dry-run
  censorbot -i movie.mp4 -o out.mp4 -s movie.srt --strictness medium
""",
    )
    p.add_argument("-i", "--input", help="input video file")
    p.add_argument("-o", "--output", help="output video file (container follows the extension)")
    p.add_argument(
        "-w",
        "--wordlist",
        help="extra words file: one per line, globs allowed, '!word' never censors",
    )
    p.add_argument(
        "-s", "--subtitles", help="subtitle file (SRT) used to catch words the transcript missed"
    )
    p.add_argument(
        "--strictness",
        choices=sorted(LEVELS),
        default="high",
        help="low: strongest words only; medium: adds insults; high (default): adds damn/hell/crap",
    )
    p.add_argument(
        "--model", default="turbo", choices=MODEL_CHOICES, help="Whisper model (default: turbo)"
    )
    p.add_argument("--language", default="en", help="spoken language (default: en)")
    p.add_argument("--mode", choices=["mute", "beep"], default="mute")
    p.add_argument(
        "--scope",
        choices=["voice", "all"],
        default="voice",
        help="voice (default): remove only the dialogue; all: silence every channel",
    )
    p.add_argument(
        "--padding",
        type=float,
        default=None,
        help="extra seconds before and after each word (default: 0.03 before, 0.05 after)",
    )
    p.add_argument(
        "--fade", type=float, default=0.02, help="fade length in seconds (default: 0.02)"
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=0.2,
        help="join cuts closer than this many seconds (default: 0.2)",
    )
    p.add_argument(
        "--audio-stream", type=int, default=0, help="which audio track to censor (default: 0)"
    )
    p.add_argument(
        "--audio-codec", default="aac", help="codec for the censored track (default: aac)"
    )
    p.add_argument(
        "--audio-bitrate",
        default=None,
        help="bitrate for the censored track (default: match source)",
    )
    p.add_argument("--single-audio", action="store_true", help="drop the original audio track")
    p.add_argument(
        "--no-separation",
        action="store_true",
        help="5.1+: cut only the centre channel (faster; dialogue spill in other speakers remains)",
    )
    p.add_argument(
        "--aligner",
        choices=["mms", "base"],
        default="mms",
        help="forced-alignment model: mms (default, robust on film audio) or base (smaller)",
    )
    p.add_argument(
        "--no-align", action="store_true", help="skip forced alignment (faster, less precise)"
    )
    p.add_argument("--no-cache", action="store_true", help="ignore the transcript cache")
    p.add_argument("--force-cpu", action="store_true", help="never use MLX/CUDA")
    p.add_argument("--dry-run", action="store_true", help="list what would be cut; write nothing")
    p.add_argument("--report", help="write a JSON report of every cut to this path")
    p.add_argument("--export-srt", help="write a censored SRT built from the transcript")
    p.add_argument("--beep-file", help="custom beep sound (any audio file)")
    p.add_argument("--stats", action="store_true", help="print word statistics")
    p.add_argument("--config", help="YAML file of option defaults")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"censorbot {__version__}")
    # Accepted for backward compatibility; no longer used.
    p.add_argument("--no-download", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu", action="store_true", help=argparse.SUPPRESS)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        conf = {k.replace("-", "_"): v for k, v in conf.items()}
        if "model_size" in conf and "model" not in conf:
            conf["model"] = conf.pop("model_size")
        parser.set_defaults(**conf)
        args = parser.parse_args(argv)  # command line still wins
    return args


def _load_beep(path: str, sr: int):
    import numpy as np

    res = __import__("subprocess").run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    )
    wave = np.frombuffer(res.stdout, dtype="<f4")
    peak = float(np.abs(wave).max()) or 1.0
    return (wave / peak).astype(np.float32)


def process(args: argparse.Namespace) -> int:
    require_ffmpeg()
    src = args.input
    info = probe(src)
    if not info.audio:
        raise MediaError("input has no audio track")
    if args.audio_stream >= len(info.audio):
        raise MediaError(f"audio stream {args.audio_stream} not found ({len(info.audio)} present)")
    stream = info.audio[args.audio_stream]
    log.info(
        "Input: %s (%.0f min), audio: %s %s %d Hz",
        Path(src).name,
        info.duration / 60,
        stream.codec,
        stream.layout,
        stream.sample_rate,
    )

    matcher = Matcher.from_file(args.wordlist, args.strictness)
    pre, post = (args.padding, args.padding) if args.padding is not None else (0.03, 0.05)
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix="censorbot-") as tmp:
        wav = os.path.join(tmp, "voice.wav")
        extract_analysis_audio(src, stream, wav)
        voice = read_wav_mono(wav)

        backend = pick_backend(args.force_cpu)
        words, segs = transcribe(
            wav, args.model, backend, args.language, None if args.no_cache else src, stream.order
        )

        aligner = None
        if not args.no_align:
            from .align import Aligner

            aligner = Aligner(args.aligner)

        from .detect import Detector, censored_transcript_srt, load_srt, write_censored_srt

        det = Detector(matcher, voice, aligner, pre, post)
        cuts = det.from_words(words)

        srt = args.subtitles or extract_text_subtitles(src, info, os.path.join(tmp, "embedded.srt"))
        if srt:
            cuts += det.from_subtitles(load_srt(srt), cuts)

        cuts = merge_cuts(cuts, max(args.merge_gap, 2 * args.fade))
        cuts = [
            Cut(max(0.0, c.start), min(info.duration or c.end, c.end), c.words, c.source)
            for c in cuts
        ]
        log.info("Detection: %s -> %d cuts", det.stats, len(cuts))

        counts = Counter(normalize(w) for c in cuts for w in c.words)
        report = {
            "input": src,
            "version": __version__,
            "model": args.model,
            "backend": backend,
            "strictness": args.strictness,
            "mode": args.mode,
            "scope": args.scope,
            "words_transcribed": len(words),
            "detection": det.stats,
            "cuts": [
                {
                    "start": round(c.start, 3),
                    "end": round(c.end, 3),
                    "words": c.words,
                    "source": c.source,
                }
                for c in cuts
            ],
        }

        if args.stats or args.dry_run:
            total = sum(len(c.words) for c in cuts)
            print(f"\nWords transcribed: {len(words)}   censored: {total}   cuts: {len(cuts)}")
            print(f"Seconds of dialogue removed: {sum(c.end - c.start for c in cuts):.1f}")
            for w, n in counts.most_common(10):
                print(f"  {n:4d}  {w}")
        if args.dry_run:
            for c in cuts:
                print(f"  [{_ts(c.start)} - {_ts(c.end)}] {' / '.join(c.words)}  ({c.source})")
        if args.export_srt:
            censored_transcript_srt(words, segs, matcher, args.export_srt)
            log.info("Wrote %s", args.export_srt)
        if args.dry_run:
            _write_report(args.report, report)
            return 0

        if not cuts:
            log.warning("No profanity found; output keeps the original audio unchanged")

        beep = _load_beep(args.beep_file, stream.sample_rate) if args.beep_file else None
        renderer = Renderer(
            stream, cuts, args.mode, args.scope, args.fade, beep, separate=not args.no_separation
        )
        audio_ext = ".m4a" if args.audio_codec in ("aac", "alac") else ".mka"
        out_audio = os.path.join(tmp, "censored" + audio_ext)
        bitrate = args.audio_bitrate or default_bitrate(stream)
        log.info("Rendering censored track (%s %s)", args.audio_codec, bitrate)
        bar = _progress(info.duration)
        rstats = renderer.run(src, out_audio, args.audio_codec, bitrate, bar)
        if bar:
            bar(None)
        report["render"] = rstats | {"strategy": renderer.strategy, "bitrate": bitrate}

        censored_srt = None
        if srt:
            censored_srt = os.path.join(tmp, "censored.srt")
            write_censored_srt(srt, censored_srt, matcher)

        out = args.output
        part = str(Path(out).with_name(Path(out).stem + ".partial" + Path(out).suffix))
        mux(src, info, stream, out_audio, part, not args.single_audio, censored_srt)
        shutil.move(part, out)

    report["seconds"] = round(time.time() - t_start, 1)
    _write_report(args.report, report)
    log.info("Done in %.1f min: %s", (time.time() - t_start) / 60, out)
    return 0


def _ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def _write_report(path: Optional[str], report: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(report, indent=2))
        log.info("Report: %s", path)


def _progress(total: float):
    if not sys.stderr.isatty() or not total:
        return None
    from tqdm import tqdm

    bar = tqdm(total=int(total), unit="s", desc="Rendering", leave=False)

    def update(pos):
        if pos is None:
            bar.close()
        else:
            bar.n = min(int(pos), bar.total)
            bar.refresh()

    return update


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "huggingface_hub", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.input:
        sys.exit("censorbot: --input is required")
    if not os.path.exists(args.input):
        sys.exit(f"censorbot: input not found: {args.input}")
    if not args.output and not args.dry_run:
        sys.exit("censorbot: --output is required unless --dry-run")
    if args.output and os.path.abspath(args.output) == os.path.abspath(args.input):
        sys.exit("censorbot: output must differ from input")
    if args.beep_file and not os.path.exists(args.beep_file):
        sys.exit(f"censorbot: beep file not found: {args.beep_file}")
    if args.no_download:
        log.info("--no-download is no longer needed: subtitles are never downloaded")
    try:
        sys.exit(process(args))
    except MediaError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
