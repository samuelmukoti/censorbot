"""Apply the cuts to the full-quality audio.

Design points (each fixes a defect in the old renderer):

* Works on the original track at its native sample rate and channel layout,
  streamed through NumPy, so the censored track is identical to the original
  everywhere except the cuts. (The old one was the 16 kHz mono Whisper copy.)
* Raised-cosine fades instead of hard on/off steps: no clicks.
* Only the voice is removed. With a centre channel the centre is cut and any
  copy of the voice panned into the front left/right is subtracted, so music
  and effects play on. Stereo uses vocal separation (Demucs) for the window
  around each cut; mono falls back to cutting everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .media import AudioStream, PCMReader, PCMWriter, voice_channel

log = logging.getLogger("censorbot")

SEP_MARGIN = 1.0  # seconds of context given to the separator on each side


@dataclass
class Cut:
    start: float
    end: float
    words: List[str] = field(default_factory=list)
    source: str = "asr"


def merge_cuts(cuts: List[Cut], gap: float) -> List[Cut]:
    out: List[Cut] = []
    for c in sorted(cuts, key=lambda c: c.start):
        if out and c.start - out[-1].end <= gap:
            out[-1].end = max(out[-1].end, c.end)
            out[-1].words += c.words
            if c.source not in out[-1].source:
                out[-1].source += "+" + c.source
        else:
            out.append(Cut(c.start, c.end, list(c.words), c.source))
    return out


def mute_curve(t: np.ndarray, cut: Cut, fade: float) -> np.ndarray:
    """1 inside the cut, 0 outside, raised-cosine ramps of `fade` seconds outside the cut."""
    m = np.zeros_like(t)
    inside = (t >= cut.start) & (t <= cut.end)
    m[inside] = 1.0
    if fade > 0:
        up = (t >= cut.start - fade) & (t < cut.start)
        m[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - (cut.start - fade)) / fade)
        down = (t > cut.end) & (t <= cut.end + fade)
        m[down] = 0.5 + 0.5 * np.cos(np.pi * (t[down] - cut.end) / fade)
    return m


class Separator:
    """Demucs vocal separation on short windows (stereo sources only)."""

    def __init__(self):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.model = get_model("htdemucs").eval()
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.vocals = self.model.sources.index("vocals")

    def vocals_of(self, x: np.ndarray, sr: int) -> np.ndarray:
        """x: (frames, 2) at sr. Returns the vocal estimate at sr, same shape."""
        import torchaudio.functional as AF
        from demucs.apply import apply_model

        torch = self.torch
        msr = self.model.samplerate
        wav = torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32))
        if sr != msr:
            wav = AF.resample(wav, sr, msr)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            stems = apply_model(
                self.model,
                ((wav - mean) / std)[None],
                device=self.device,
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )
        voc = stems[0, self.vocals] * std
        if sr != msr:
            voc = AF.resample(voc, msr, sr)
        voc = voc.cpu().numpy().T
        if len(voc) < len(x):
            voc = np.pad(voc, ((0, len(x) - len(voc)), (0, 0)))
        return voc[: len(x)]


class Renderer:
    def __init__(
        self,
        stream: AudioStream,
        cuts: List[Cut],
        mode: str = "mute",
        scope: str = "voice",
        fade: float = 0.02,
        beep: Optional[np.ndarray] = None,
        beep_freq: float = 1000.0,
        separate: bool = True,
    ):
        self.s = stream
        self.cuts = sorted(cuts, key=lambda c: c.start)
        self.mode = mode
        self.fade = fade
        self.beep_wave = beep
        self.beep_freq = beep_freq
        self.fc = voice_channel(stream)
        names = stream.channel_names
        self.fronts = [names.index(n) for n in ("FL", "FR") if n in names]
        # Speaker pairs that can carry dialogue spill and reverb besides the centre.
        self.pairs = [
            [names.index(a), names.index(b)]
            for a, b in (("FL", "FR"), ("SL", "SR"), ("BL", "BR"))
            if a in names and b in names
        ]

        if scope == "all" or stream.channels == 1:
            self.strategy = "all"
        elif self.fc is not None:
            self.strategy = "centre"
        elif stream.channels == 2:
            self.strategy = "separate"
        else:
            self.strategy = "all"
        self.separator = None
        if cuts and (self.strategy == "separate" or (self.strategy == "centre" and separate)):
            try:
                self.separator = Separator()
            except Exception as e:  # missing demucs or model download failure
                if self.strategy == "separate":
                    log.warning("Vocal separation unavailable (%s); cutting the whole mix", e)
                    self.strategy = "all"
                else:
                    log.warning("Vocal separation unavailable (%s); centre cut only", e)
        self.margin = SEP_MARGIN if self.separator is not None else 0.0
        log.info(
            "Render strategy: %s (%s, %d ch @ %d Hz)",
            self.strategy,
            stream.layout,
            stream.channels,
            stream.sample_rate,
        )
        self.stats = {"cuts": len(self.cuts), "front_subtracted": 0, "pairs_separated": 0}

    # -- chunking -------------------------------------------------------
    def _guard(self) -> List[tuple]:
        """Time ranges a chunk boundary must not fall inside."""
        pad = self.fade + max(self.margin, 0.05) + 0.05
        zones = []
        for c in self.cuts:
            a, b = c.start - pad, c.end + pad
            if zones and a <= zones[-1][1]:
                zones[-1] = (zones[-1][0], max(zones[-1][1], b))
            else:
                zones.append((a, b))
        return zones

    def _chunk_sizes(self, nominal: float = 20.0):
        sr = self.s.sample_rate
        zones = self._guard()
        zi = 0
        pos = 0.0
        while True:
            end = pos + nominal
            while zi < len(zones) and zones[zi][1] < end:
                zi += 1
            if zi < len(zones) and zones[zi][0] < end <= zones[zi][1]:
                end = zones[zi][1] + 0.01
            n = int(round(end * sr)) - int(round(pos * sr))
            yield n
            pos = int(round(end * sr)) / sr

    # -- processing -----------------------------------------------------
    def _beep(self, n: int, sr: int, t0: float) -> np.ndarray:
        if self.beep_wave is not None:
            reps = int(np.ceil(n / max(1, len(self.beep_wave))))
            return np.tile(self.beep_wave, reps)[:n]
        t = t0 + np.arange(n) / sr
        return np.sin(2 * np.pi * self.beep_freq * t).astype(np.float32)

    def process(self, x: np.ndarray, t0: float) -> np.ndarray:
        """Process one chunk in place. x: (frames, channels) starting at t0."""
        sr = self.s.sample_rate
        t_end = t0 + len(x) / sr
        for c in self.cuts:
            if c.end + self.fade < t0 or c.start - self.fade > t_end:
                continue
            margin = self.margin
            a = max(0, int((c.start - self.fade - margin - t0) * sr))
            b = min(len(x), int(np.ceil((c.end + self.fade + margin - t0) * sr)))
            if b <= a:
                continue
            t = t0 + np.arange(a, b) / sr
            m = mute_curve(t, c, self.fade).astype(np.float32)
            w = x[a:b]
            core = m > 0.999
            if self.strategy == "centre":
                voice = w[:, self.fc].copy()
                if self.separator is not None:
                    # Film mixes spread dialogue (spill, reverb, effects) into the
                    # other speakers; separate and remove it pair by pair.
                    for pair in self.pairs:
                        if float(np.abs(w[:, pair]).max()) < 1e-5:
                            continue
                        voc = self.separator.vocals_of(w[:, pair], sr)
                        w[:, pair] -= m[:, None] * voc
                        self.stats["pairs_separated"] += 1
                # Otherwise subtract the share of the voice panned into front L/R.
                vv = float(np.dot(voice[core], voice[core]))
                for ch in self.fronts if self.separator is None else []:
                    if vv <= 1e-9:
                        break
                    other = w[core, ch]
                    gain = float(np.dot(other, voice[core]) / vv)
                    corr = float(
                        np.dot(other, voice[core])
                        / (np.sqrt(vv * float(np.dot(other, other))) + 1e-12)
                    )
                    if corr > 0.3 and 0.02 < gain < 1.5:
                        w[:, ch] -= m * gain * voice
                        self.stats["front_subtracted"] += 1
                w[:, self.fc] = voice * (1 - m)
                level = float(np.sqrt(np.mean(voice[core] ** 2))) if core.any() else 0.05
                targets = [self.fc]
            elif self.strategy == "separate":
                voc = self.separator.vocals_of(w, sr)
                w -= m[:, None] * voc
                level = float(np.sqrt(np.mean(voc[core] ** 2))) if core.any() else 0.05
                targets = list(range(w.shape[1]))
            else:
                level = float(np.sqrt(np.mean(w[core] ** 2))) if core.any() else 0.05
                w *= (1 - m)[:, None]
                targets = list(range(w.shape[1]))
            if self.mode == "beep":
                amp = float(np.clip(level * 1.2, 0.02, 0.35))
                tone = self._beep(b - a, sr, t[0]) * m * amp
                for ch in targets:
                    w[:, ch] = np.clip(w[:, ch] + tone, -1.0, 1.0)
            x[a:b] = w
        return x

    def run(self, src: str, out_audio: str, codec: str, bitrate: str, progress=None) -> dict:
        reader = PCMReader(src, self.s)
        writer = PCMWriter(out_audio, self.s, codec, bitrate)
        pos = 0
        sr = self.s.sample_rate
        try:
            for n in self._chunk_sizes():
                block = reader.read(n)
                if len(block) == 0:
                    break
                block = np.array(block, dtype=np.float32)  # writable copy
                self.process(block, pos / sr)
                writer.write(block)
                pos += len(block)
                if progress:
                    progress(pos / sr)
                if len(block) < n:
                    break
        finally:
            reader.close()
            writer.close()
        self.stats["seconds"] = pos / sr
        return self.stats
