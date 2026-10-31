"""FFmpeg/FFprobe helpers: probing, analysis audio, streaming PCM, muxing."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("censorbot")

# Channel order FFmpeg uses for the common layouts.
LAYOUT_CHANNELS = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
}
DEFAULT_LAYOUT = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "quad",
    5: "5.0",
    6: "5.1",
    7: "6.1",
    8: "7.1",
}
TEXT_SUB_CODECS = {"subrip", "srt", "mov_text", "ass", "ssa", "webvtt", "text"}


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise MediaError(f"{tool} not found on PATH. Install FFmpeg first.")


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise MediaError(f"{cmd[0]} failed: {res.stderr.strip()[-2000:]}")
    return res


@dataclass
class AudioStream:
    index: int  # absolute stream index
    order: int  # position among audio streams (for -map 0:a:N)
    codec: str
    channels: int
    layout: str
    sample_rate: int
    bit_rate: Optional[int]
    language: str
    title: str

    @property
    def channel_names(self) -> List[str]:
        return (
            LAYOUT_CHANNELS.get(self.layout)
            or LAYOUT_CHANNELS[DEFAULT_LAYOUT.get(self.channels, "stereo")][: self.channels]
        )


@dataclass
class SubStream:
    index: int
    order: int
    codec: str
    language: str
    title: str
    forced: bool

    @property
    def is_text(self) -> bool:
        return self.codec in TEXT_SUB_CODECS


@dataclass
class MediaInfo:
    path: str
    duration: float
    audio: List[AudioStream]
    subs: List[SubStream]
    has_video: bool


def probe(path: str) -> MediaInfo:
    out = run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
    ).stdout
    data = json.loads(out)
    audio, subs, has_video = [], [], False
    a_order = s_order = 0
    for s in data.get("streams", []):
        tags = s.get("tags", {}) or {}
        kind = s.get("codec_type")
        if kind == "video" and not (s.get("disposition", {}) or {}).get("attached_pic"):
            has_video = True
        elif kind == "audio":
            ch = int(s.get("channels") or 2)
            layout = s.get("channel_layout") or DEFAULT_LAYOUT.get(ch, "stereo")
            br = s.get("bit_rate") or tags.get("BPS") or tags.get("BPS-eng")
            audio.append(
                AudioStream(
                    index=int(s["index"]),
                    order=a_order,
                    codec=s.get("codec_name", "?"),
                    channels=ch,
                    layout=layout,
                    sample_rate=int(s.get("sample_rate") or 48000),
                    bit_rate=int(br) if br and str(br).isdigit() else None,
                    language=tags.get("language", "und"),
                    title=tags.get("title", ""),
                )
            )
            a_order += 1
        elif kind == "subtitle":
            disp = s.get("disposition", {}) or {}
            title = tags.get("title", "")
            subs.append(
                SubStream(
                    index=int(s["index"]),
                    order=s_order,
                    codec=s.get("codec_name", "?"),
                    language=tags.get("language", "und"),
                    title=title,
                    forced=bool(disp.get("forced")) or "forced" in title.lower(),
                )
            )
            s_order += 1
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return MediaInfo(path, duration, audio, subs, has_video)


def voice_channel(stream: AudioStream) -> Optional[int]:
    """Index of the dialogue (centre) channel, if the layout has one."""
    names = stream.channel_names
    if stream.channels >= 3 and "FC" in names:
        return names.index("FC")
    return None


def extract_analysis_audio(path: str, stream: AudioStream, out_wav: str) -> None:
    """16 kHz mono WAV for speech recognition.

    Uses only the centre channel when there is one: that is where film
    dialogue lives, so Whisper hears far less music and effects.
    """
    fc = voice_channel(stream)
    af = f"pan=mono|c0=c{fc}" if fc is not None else "aresample=16000"
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            path,
            "-map",
            f"0:a:{stream.order}",
            "-vn",
            "-af",
            af,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            out_wav,
        ]
    )


def read_wav_mono(path: str) -> np.ndarray:
    """Read a 16-bit mono WAV written by extract_analysis_audio as float32."""
    import wave

    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class PCMReader:
    """Stream an audio track as float32 frames at its native rate and layout."""

    def __init__(self, path: str, stream: AudioStream):
        self.channels = stream.channels
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                f"0:a:{stream.order}",
                "-vn",
                "-f",
                "f32le",
                "-c:a",
                "pcm_f32le",
                "-ac",
                str(stream.channels),
                "-ar",
                str(stream.sample_rate),
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self, frames: int) -> np.ndarray:
        want = frames * self.channels * 4
        buf = bytearray()
        while len(buf) < want:
            chunk = self.proc.stdout.read(want - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        usable = len(buf) - len(buf) % (self.channels * 4)
        return np.frombuffer(bytes(buf[:usable]), dtype="<f4").reshape(-1, self.channels)

    def close(self) -> None:
        self.proc.stdout.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise MediaError(f"ffmpeg decode failed: {err[-2000:]}")


def pick_encoder(codec: str) -> str:
    if codec != "aac":
        return codec
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"]).stdout
    # AudioToolbox AAC (macOS) sounds better than FFmpeg's native encoder.
    return "aac_at" if " aac_at " in encoders else "aac"


class PCMWriter:
    """Encode float32 frames to an audio file."""

    def __init__(self, out_path: str, stream: AudioStream, codec: str, bitrate: str):
        enc = pick_encoder(codec)
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(stream.sample_rate),
            "-ch_layout",
            (
                stream.layout
                if stream.layout in LAYOUT_CHANNELS
                else DEFAULT_LAYOUT.get(stream.channels, "stereo")
            ),
            "-i",
            "-",
            "-c:a",
            enc,
        ]
        if enc not in ("flac", "pcm_s16le", "pcm_s24le"):
            cmd += ["-b:a", bitrate]
        cmd.append(out_path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frames, dtype="<f4").tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise MediaError(f"ffmpeg encode failed: {err[-2000:]}")


def default_bitrate(stream: AudioStream) -> str:
    """Bitrate for the re-encoded censored track.

    It is a second lossy generation, so give it headroom: twice the source,
    at least 112 kbps per channel, capped at 640 kbps (AAC's practical limit
    for 5.1). Measured on a 5.1 film: 288k re-encode = -24 dB waveform error,
    640k = -38 dB. Use --audio-codec flac (MKV) for lossless.
    """
    br = max(2 * (stream.bit_rate or 0), 112_000 * stream.channels)
    cap = 640_000 if stream.channels > 2 else 320_000
    return f"{min(br, cap) // 1000}k"


def extract_text_subtitles(path: str, info: MediaInfo, out_srt: str) -> Optional[str]:
    """Extract the best embedded text subtitle track as SRT, if any."""
    text = [s for s in info.subs if s.is_text]
    if not text:
        if info.subs:
            log.info(
                "Embedded subtitles are image-based (%s); skipping them",
                ", ".join(sorted({s.codec for s in info.subs})),
            )
        return None

    def rank(s: SubStream):
        return (s.forced, s.language not in ("eng", "en", "und"), "sdh" in s.title.lower())

    best = sorted(text, key=rank)[0]
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                f"0:s:{best.order}",
                "-f",
                "srt",
                out_srt,
            ]
        )
    except MediaError as e:
        log.warning("Could not extract embedded subtitles: %s", e)
        return None
    if Path(out_srt).stat().st_size == 0:
        return None
    log.info("Using embedded subtitle track %d (%s, %s)", best.order, best.codec, best.language)
    return out_srt


def mux(
    src: str,
    info: MediaInfo,
    stream: AudioStream,
    censored_audio: str,
    out_path: str,
    keep_original: bool,
    censored_srt: Optional[str],
) -> None:
    """Write the output: video copied, censored audio first and default.

    The original audio is stream-copied (not re-encoded) when kept.
    """
    ext = Path(out_path).suffix.lower()
    is_mp4 = ext in (".mp4", ".m4v", ".mov")
    lang = stream.language if stream.language != "und" else "eng"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-i", censored_audio]
    if censored_srt:
        cmd += ["-i", censored_srt]
    cmd += ["-map", "0:v?", "-map", "1:a:0"]
    if keep_original:
        cmd += ["-map", f"0:a:{stream.order}"]
    copy_subs = [s for s in info.subs if not is_mp4 or s.codec == "mov_text"]
    for s in copy_subs:
        cmd += ["-map", f"0:s:{s.order}"]
    if censored_srt:
        cmd += ["-map", "2:0"]
    cmd += ["-map_metadata", "0", "-c", "copy"]
    if censored_srt:
        n = len(copy_subs)
        cmd += [
            f"-c:s:{n}",
            "mov_text" if is_mp4 else "srt",
            f"-metadata:s:s:{n}",
            "title=Censored",
            f"-metadata:s:s:{n}",
            f"language={lang}",
        ]
    # MP4 players read track names from handler_name; MKV uses title.
    cmd += ["-metadata:s:a:0", "title=Censored", "-metadata:s:a:0", "handler_name=Censored"]
    cmd += ["-metadata:s:a:0", f"language={lang}", "-disposition:a:0", "default"]
    if keep_original:
        name = stream.title or "Original"
        cmd += ["-metadata:s:a:1", f"title={name}", "-metadata:s:a:1", f"handler_name={name}"]
        cmd += ["-disposition:a:1", "0"]
    cmd.append(out_path)
    run(cmd)
