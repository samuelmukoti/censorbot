"""Precise word boundaries.

Whisper's word timestamps come from attention weights and are often 100-300 ms
off, which is why the old tool clipped neighbouring words or let the start of a
word through. Here every word we censor is re-timed in two steps:

1. Forced alignment: a wav2vec2 CTC model aligns the known words to the audio
   of a short window around the hit (about 20 ms resolution).
2. Onset refinement: the edges are extended to where the voice energy actually
   starts and stops, which catches soft onsets ("f", "sh") that CTC models
   place late, without crossing into the neighbouring words.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("censorbot")

SR = 16000


@dataclass
class Span:
    start: float
    end: float
    score: float


ALIGNERS = ("mms", "base")


class Aligner:
    """CTC forced aligner.

    mms:  torchaudio MMS_FA, trained for alignment on varied real-world speech.
    base: WAV2VEC2_ASR_BASE_960H (LibriSpeech read speech), smaller and faster.
    """

    def __init__(self, model: str = "mms", device: Optional[str] = None):
        import torch
        import torchaudio

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model == "mms":
            bundle = torchaudio.pipelines.MMS_FA
            self.model = bundle.get_model(with_star=False).to(self.device).eval()
            labels = bundle.get_dict(star=None)  # {char: index}, blank "-" = 0
            self.index = {c.upper(): i for c, i in labels.items()}
        elif model == "base":
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self.model = bundle.get_model().to(self.device).eval()
            self.index = {c: i for i, c in enumerate(bundle.get_labels())}
        else:
            raise ValueError(f"aligner must be one of {ALIGNERS}")
        self.blank = 0
        self.sep = self.index.get("|")

    def _emission(self, audio: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(audio.astype(np.float32))[None].to(self.device)
            em, _ = self.model(x)
            em = torch.log_softmax(em, dim=-1)[0].cpu().numpy()
        return em

    def _tokens(self, word: str) -> List[int]:
        return [self.index[c] for c in word.upper() if c in self.index and c not in "-|*"]

    def align(self, audio: np.ndarray, t0: float, words: Sequence[str]) -> List[Optional[Span]]:
        """Align `words` within `audio` (16 kHz mono starting at time t0)."""
        if len(audio) < SR * 0.2:
            return [None] * len(words)
        em = self._emission(audio)
        frames = em.shape[0]
        sec_per_frame = len(audio) / frames / SR

        toks, owner = [], []
        word_tokens = [self._tokens(w) for w in words]
        for wi, wt in enumerate(word_tokens):
            if not wt:
                continue
            if toks and self.sep is not None:
                toks.append(self.sep)
                owner.append(-1)
            toks.extend(wt)
            owner.extend([wi] * len(wt))
        if not toks or len(toks) > frames:
            return [None] * len(words)

        path = _viterbi(em, toks, self.blank)
        if path is None:
            return [None] * len(words)

        spans: List[Optional[Span]] = [None] * len(words)
        first, last, scores = {}, {}, {}
        for tok_i, frame in path:
            wi = owner[tok_i]
            if wi < 0:
                continue
            first.setdefault(wi, frame)
            last[wi] = frame
            scores.setdefault(wi, []).append(em[frame, toks[tok_i]])
        for wi in first:
            spans[wi] = Span(
                t0 + first[wi] * sec_per_frame,
                t0 + (last[wi] + 1) * sec_per_frame,
                float(np.exp(np.mean(scores[wi]))),
            )
        return spans


def _viterbi(em: np.ndarray, toks: List[int], blank: int) -> Optional[List[Tuple[int, int]]]:
    """Best monotonic alignment; returns (token index, frame) for each token.

    dp[t, j] is the best score after t frames with the first j tokens emitted.
    Each frame either stays (blank or repeat of the current token) or emits
    the next token.
    """
    T, N = em.shape[0], len(toks)
    tok = np.asarray(toks)
    dp = np.full((T + 1, N + 1), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    stay_blank = em[:, blank]
    for t in range(T):
        prev = dp[t]
        # stay: blank, or repeat the token just emitted
        stay = prev.copy()
        stay[0] += stay_blank[t]
        rep = np.maximum(stay_blank[t], em[t, tok])  # j>=1: blank or same token again
        stay[1:] += rep
        adv = np.full(N + 1, -np.inf)
        adv[1:] = prev[:-1] + em[t, tok]
        dp[t + 1] = np.maximum(stay, adv)
    if not np.isfinite(dp[T, N]):
        return None
    out = []
    j = N
    for t in range(T, 0, -1):
        if j == 0:
            break
        adv = dp[t - 1, j - 1] + em[t - 1, tok[j - 1]]
        if adv >= dp[t, j] - 1e-9 and np.isfinite(adv):
            out.append((j - 1, t - 1))
            j -= 1
        elif em[t - 1, tok[j - 1]] > stay_blank[t - 1]:
            out.append((j - 1, t - 1))  # repeated frame of the same token
    if j != 0:
        return None
    out.reverse()
    return out


def refine_edges(
    audio: np.ndarray,
    t0: float,
    start: float,
    end: float,
    lo: float,
    hi: float,
    max_pre: float = 0.12,
    max_post: float = 0.20,
) -> Tuple[float, float]:
    """Grow [start, end] to where voice energy drops, staying inside [lo, hi]."""
    hop = int(0.01 * SR)
    a = max(0, int((max(lo, start - max_pre) - t0) * SR))
    b = min(len(audio), int((min(hi, end + max_post) - t0) * SR))
    seg = audio[a:b]
    if len(seg) < hop * 4:
        return start, end
    n = len(seg) // hop
    rms = np.sqrt(np.mean(seg[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 15)
    peak = db.max()
    if peak - floor < 10:  # no clear voice/silence contrast; trust alignment
        return start, end
    thresh = floor + max(6.0, (peak - floor) * 0.25)

    def frame_of(t):
        return int(np.clip(((t - t0) * SR - a) // hop, 0, n - 1))

    s, e = frame_of(start), frame_of(end)
    while s > 0 and db[s - 1] > thresh:
        s -= 1
    while e < n - 1 and db[e + 1] > thresh:
        e += 1
    new_start = t0 + (a + s * hop) / SR
    new_end = t0 + (a + (e + 1) * hop) / SR
    return min(start, max(lo, new_start)), max(end, min(hi, new_end))
