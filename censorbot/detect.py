"""Turn a transcript (and optional subtitles) into precise cuts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Sequence, Set

import numpy as np

from .align import SR, refine_edges
from .render import Cut
from .transcribe import Segment, Word
from .words import Matcher

log = logging.getLogger("censorbot")

CONTEXT_WORDS = 3
MAX_WINDOW = 10.0


@dataclass
class Cue:
    start: float
    end: float
    text: str


def load_srt(path: str) -> List[Cue]:
    import chardet
    import pysrt

    raw = open(path, "rb").read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    cues = []
    for s in pysrt.open(path, encoding=enc):
        text = re.sub(r"<[^>]+>|\{[^}]*\}", "", s.text).replace("\n", " ").strip()
        if text:
            cues.append(Cue(s.start.ordinal / 1000.0, s.end.ordinal / 1000.0, text))
    return cues


def write_censored_srt(src: str, dst: str, matcher: Matcher) -> int:
    import chardet
    import pysrt

    raw = open(src, "rb").read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    subs = pysrt.open(src, encoding=enc)
    total = 0
    for s in subs:
        s.text, n = matcher.censor_text(s.text)
        total += n
    subs.save(dst, encoding="utf-8")
    return total


def _clip(audio: np.ndarray, a: float, b: float) -> np.ndarray:
    return audio[max(0, int(a * SR)) : max(0, int(b * SR))]


class Detector:
    def __init__(
        self,
        matcher: Matcher,
        voice: np.ndarray,
        aligner,
        pre: float = 0.03,
        post: float = 0.05,
    ):
        self.m = matcher
        self.voice = voice
        # One aligner or a list tried in order (first plausible answer wins).
        if aligner is None:
            self.aligners = []
        elif isinstance(aligner, (list, tuple)):
            self.aligners = list(aligner)
        else:
            self.aligners = [aligner]
        self.aligner = self.aligners[0] if self.aligners else None
        self.pre, self.post = pre, post
        self.stats = {
            "asr_hits": 0,
            "aligned": 0,
            "fallback": 0,
            "subtitle_only": 0,
            "subtitle_unaligned": 0,
        }

    # -- transcript hits ----------------------------------------------------
    def from_words(self, words: List[Word]) -> List[Cut]:
        hits = self.m.find([w.text for w in words])
        self.stats["asr_hits"] = len(hits)
        cuts = []
        for h in hits:
            cuts.append(self._place(words, h.first, h.last))
        return cuts

    def _place(self, words: List[Word], first: int, last: int) -> Cut:
        label = " ".join(words[k].text for k in range(first, last + 1))
        lo = max(0, first - CONTEXT_WORDS)
        hi = min(len(words) - 1, last + CONTEXT_WORDS)
        while hi > lo and words[hi].end - words[lo].start > MAX_WINDOW:
            if hi - last > first - lo:
                hi -= 1
            elif lo < first:
                lo += 1
            else:
                break
        w0 = max(0.0, words[lo].start - 0.4)
        w1 = words[hi].end + 0.4
        ws, we = words[first].start, words[last].end
        spans, target = None, []
        for al in self.aligners:
            try:
                spans = al.align(
                    _clip(self.voice, w0, w1), w0, [w.text for w in words[lo : hi + 1]]
                )
            except Exception as e:  # never let alignment kill a run
                log.debug("alignment failed at %.2fs: %s", words[first].start, e)
                spans = None
            target = spans[first - lo : last - lo + 1] if spans else []
            if target and all(target) and _plausible(target, ws, we):
                break
            target = []
        if target:
            s, e = min(t.start for t in target), max(t.end for t in target)
            prev = next((sp for sp in reversed(spans[: first - lo]) if sp), None)
            nxt = next((sp for sp in spans[last - lo + 1 :] if sp), None)
            lo_b = prev.end if prev else s - 0.25
            hi_b = nxt.start if nxt else e + 0.35
            s, e = refine_edges(self.voice, 0.0, s, e, lo_b, hi_b)
            self.stats["aligned"] += 1
            return Cut(
                max(lo_b - 0.02, s - self.pre), min(hi_b + 0.02, e + self.post), [label], "asr"
            )
        self.stats["fallback"] += 1
        return Cut(max(0.0, ws - 0.12), we + 0.12, [label], "asr-unaligned")

    # -- subtitles ---------------------------------------------------------
    def from_subtitles(self, cues: List[Cue], asr_cuts: List[Cut]) -> List[Cut]:
        """Catch words the subtitles show but the transcript missed."""
        extra = []
        for cue in cues:
            tokens = cue.text.split()
            hits = self.m.find(tokens)
            if not hits:
                continue
            near = [
                c for c in asr_cuts + extra if c.end > cue.start - 1.0 and c.start < cue.end + 1.0
            ]
            if len(near) >= len(hits):
                continue
            placed = self._place_in_cue(cue, tokens, hits, near)
            extra.extend(placed)
        return extra

    def _place_in_cue(self, cue: Cue, tokens: Sequence[str], hits, near: List[Cut]) -> List[Cut]:
        w0, w1 = max(0.0, cue.start - 0.6), cue.end + 0.6
        spans = None
        if self.aligner is not None and w1 - w0 <= 15.0:
            try:
                spans = self.aligner.align(_clip(self.voice, w0, w1), w0, list(tokens))
            except Exception as e:
                log.debug("subtitle alignment failed at %.2fs: %s", cue.start, e)
        out = []
        taken: Set[int] = set()
        for h in hits:
            target = spans[h.first : h.last + 1] if spans else []
            label = " ".join(tokens[h.first : h.last + 1])
            if target and all(target):
                s, e = min(t.start for t in target), max(t.end for t in target)
                if any(c.start <= e and c.end >= s for c in near):
                    continue  # already censored via the transcript
                s, e = refine_edges(self.voice, 0.0, s, e, s - 0.15, e + 0.25)
                out.append(Cut(s - self.pre, e + self.post, [label], "subtitle"))
                self.stats["subtitle_only"] += 1
            elif id(cue) not in taken:
                # Could not locate the word: cut the voice for the whole line.
                taken.add(id(cue))
                out.append(Cut(cue.start, cue.end, [label], "subtitle-unaligned"))
                self.stats["subtitle_unaligned"] += 1
        return out


def _plausible(target, ws: float, we: float) -> bool:
    s, e = min(t.start for t in target), max(t.end for t in target)
    mid_a, mid_w = (s + e) / 2, (ws + we) / 2
    return abs(mid_a - mid_w) < 0.8 and 0.04 < e - s < max(2.5, (we - ws) * 3)


def censored_transcript_srt(
    words: List[Word], segs: List[Segment], matcher: Matcher, dst: str
) -> None:
    """SRT built from the transcript, with censored words masked."""
    import pysrt

    hit_idx = set()
    for h in matcher.find([w.text for w in words]):
        hit_idx.update(range(h.first, h.last + 1))
    subs = pysrt.SubRipFile()
    wi = 0
    for n, seg in enumerate(segs, 1):
        parts = []
        while wi < len(words) and words[wi].start < seg.end - 1e-3:
            t = words[wi].text
            parts.append(re.sub(r"[A-Za-z0-9]", "*", t) if wi in hit_idx else t)
            wi += 1
        text = " ".join(parts) or seg.text
        subs.append(
            pysrt.SubRipItem(
                index=n,
                start=pysrt.SubRipTime.from_ordinal(int(seg.start * 1000)),
                end=pysrt.SubRipTime.from_ordinal(int(seg.end * 1000)),
                text=text,
            )
        )
    subs.save(dst, encoding="utf-8")
