"""Speech recognition with word timestamps (MLX on Apple Silicon, faster-whisper elsewhere)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("censorbot")

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
FW_NAMES = {"large": "large-v3", "turbo": "large-v3-turbo"}
MODEL_CHOICES = sorted(MLX_REPOS)


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str


def mlx_available() -> bool:
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        return False
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def pick_backend(force_cpu: bool) -> str:
    if not force_cpu and mlx_available():
        return "mlx"
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "No speech backend: install mlx-whisper (Apple Silicon) or faster-whisper"
        )
    if not force_cpu:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _cache_key(media: str, stream: int, model: str, language: str) -> str:
    st = os.stat(media)
    raw = f"v2|{Path(media).resolve()}|{st.st_size}|{int(st.st_mtime)}|{stream}|{model}|{language}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_dir() -> Path:
    base = os.environ.get("CENSORBOT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "censorbot"
    )
    return Path(base)


def transcribe(
    wav_path: str,
    model: str,
    backend: str,
    language: str = "en",
    media_for_cache: Optional[str] = None,
    stream_for_cache: int = 0,
) -> Tuple[List[Word], List[Segment]]:
    cache_file = None
    if media_for_cache:
        cache_file = (
            cache_dir() / f"{_cache_key(media_for_cache, stream_for_cache, model, language)}.json"
        )
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            log.info("Using cached transcript (%s)", cache_file)
            return [Word(**w) for w in data["words"]], [Segment(**s) for s in data["segments"]]

    log.info(
        "Transcribing with %s backend, model %s (this is the slow step)", backend.upper(), model
    )
    if backend == "mlx":
        words, segs = _mlx(wav_path, model, language)
    else:
        words, segs = _faster_whisper(wav_path, model, backend, language)
    log.info("Transcribed %d words", len(words))

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"words": [asdict(w) for w in words], "segments": [asdict(s) for s in segs]})
        )
    return words, segs


def _mlx(wav: str, model: str, language: str):
    import mlx_whisper

    result = mlx_whisper.transcribe(
        wav,
        path_or_hf_repo=MLX_REPOS.get(model, model),
        word_timestamps=True,
        language=language,
        # Carrying text between windows is what makes Whisper loop and
        # hallucinate on long films.
        condition_on_previous_text=False,
        verbose=None,
    )
    words, segs = [], []
    for seg in result.get("segments", []):
        segs.append(Segment(float(seg["start"]), float(seg["end"]), seg["text"].strip()))
        for w in seg.get("words", []):
            words.append(
                Word(
                    float(w["start"]),
                    float(w["end"]),
                    w["word"].strip(),
                    float(w.get("probability", 1.0)),
                )
            )
    return words, segs


def _faster_whisper(wav: str, model: str, backend: str, language: str):
    from faster_whisper import WhisperModel

    name = FW_NAMES.get(model, model)
    if backend == "cuda":
        m = WhisperModel(name, device="cuda", compute_type="float16")
    else:
        m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    segments, _ = m.transcribe(
        wav,
        word_timestamps=True,
        language=language,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    words, segs = [], []
    for seg in segments:
        segs.append(Segment(seg.start, seg.end, seg.text.strip()))
        for w in seg.words or []:
            words.append(Word(w.start, w.end, w.word.strip(), w.probability))
    return words, segs
