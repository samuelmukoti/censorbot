"""Full pipeline on a synthetic 5.1 clip with known word timings.

Needs macOS `say`, FFmpeg and a Whisper backend; downloads models on first run.
Run with: pytest -m slow
"""

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SR = 48000
SCRIPT = [
    ("Hello", False),
    ("class,", False),
    ("pass", False),
    ("me", False),
    ("that", False),
    ("glass.", False),
    ("Oh", False),
    ("shit,", True),
    ("that", False),
    ("fucking", True),
    ("hurt.", False),
    ("What", False),
    ("the", False),
    ("hell,", True),
    ("Michelle?", False),
    ("You", False),
    ("dumbass.", True),
    ("Thank", False),
    ("you,", False),
    ("assistant.", False),
]


def _say(word: str, path: Path) -> np.ndarray:
    subprocess.run(["say", "-v", "Samantha", "-o", str(path), word.strip(",.?")], check=True)
    out = path.with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), str(out)],
        check=True,
    )
    with wave.open(str(out)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    idx = np.nonzero(np.abs(a) > 0.01)[0]
    return a[idx[0] : idx[-1] + 1] if len(idx) else a


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("needs macOS say and ffmpeg")
    d = tmp_path_factory.mktemp("e2e")
    gap = int(0.14 * SR)
    pieces, truth, pos = [np.zeros(SR, np.float32)], [], SR
    for i, (word, dirty) in enumerate(SCRIPT):
        a = _say(word, d / f"w{i}.aiff")
        truth.append((pos / SR, (pos + len(a)) / SR, word, dirty))
        pieces += [a, np.zeros(gap, np.float32)]
        pos += len(a) + gap
    pieces.append(np.zeros(SR, np.float32))
    voice = np.concatenate(pieces) * 0.7
    n = len(voice)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    music = 0.05 * (np.sin(2 * np.pi * 110 * t) + np.sin(2 * np.pi * 164.8 * t))
    x = np.zeros((n, 6), np.float32)
    x[:, 0] = music + 0.2 * voice
    x[:, 1] = music + 0.2 * voice
    x[:, 2] = voice
    x[:, 3] = 0.05 * np.sin(2 * np.pi * 40 * t)
    x[:, 4:] = 0.01 * rng.standard_normal((n, 2))
    raw = d / "a.f32"
    raw.write_bytes(x.astype("<f4").tobytes())
    src = d / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={n / SR}",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "-ch_layout",
            "5.1",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-b:a",
            "384k",
            "-shortest",
            str(src),
        ],
        check=True,
    )
    return d, src, truth


def _decode(path: Path, track: int) -> np.ndarray:
    res = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:a:{track}",
            "-f",
            "f32le",
            "-ac",
            "6",
            "-ar",
            str(SR),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(res.stdout, dtype="<f4").reshape(-1, 6)


def test_pipeline_cuts_exactly_the_bad_words(clip):
    d, src, truth = clip
    out, report = d / "clean.mp4", d / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "censorbot",
            "-i",
            str(src),
            "-o",
            str(out),
            "--report",
            str(report),
            "--no-cache",
        ],
        check=True,
    )
    cuts = json.loads(report.read_text())["cuts"]

    def overlap(a0, a1):
        return sum(max(0.0, min(a1, c["end"]) - max(a0, c["start"])) for c in cuts)

    for s, e, word, dirty in truth:
        cov = overlap(s, e) / (e - s)
        if dirty:
            assert cov > 0.95, f"{word!r} only {cov:.0%} covered"
        else:
            assert overlap(s, e) < 0.03, f"clean word {word!r} clipped by {overlap(s, e):.3f}s"

    orig, clean = _decode(out, 1), _decode(out, 0)
    n = min(len(orig), len(clean))
    for s, e, word, dirty in truth:
        a, b = int(s * SR) + 480, int(e * SR) - 480
        c_rms = np.sqrt(np.mean(clean[a:b, 2] ** 2))
        o_rms = np.sqrt(np.mean(orig[a:b, 2] ** 2))
        if dirty:
            assert c_rms < 0.03 * o_rms, f"{word!r} still audible in centre"
        else:
            assert c_rms > 0.9 * o_rms, f"clean word {word!r} lost level"
    # The censored track stays sample-aligned with the original.
    seg = slice(int(0.2 * SR), int(0.9 * SR))
    lag = np.argmax(np.correlate(clean[seg, 0], orig[seg, 0], "full")) - (seg.stop - seg.start - 1)
    assert lag == 0 and n > 0


def test_subtitles_are_censored_and_embedded(clip):
    d, src, truth = clip
    srt = d / "clip.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello class, pass me that glass.\n\n"
        "2\n00:00:04,000 --> 00:00:08,000\nOh shit, that fucking hurt.\n"
    )
    out = d / "subs.mkv"
    subprocess.run(
        [sys.executable, "-m", "censorbot", "-i", str(src), "-o", str(out), "-s", str(srt)],
        check=True,
    )
    text = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-map", "0:s:0", "-f", "srt", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Hello class, pass me that glass." in text
    assert "shit" not in text and "fucking" not in text and "****," in text
