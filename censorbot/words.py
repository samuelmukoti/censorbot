"""Profanity matching on whole, normalised words.

The old matcher flagged any word that merely *contained* a bad word, so
"hello", "class" and "glass" were muted. Here a pattern only matches a whole
token:

* ``word``      matches the word and its plain plural (``word``, ``words``, ``wordes``)
* ``wo*d``      a glob, matched against the whole token (``*fuck*`` covers
                "motherfucker" and "fuckin")
* ``two words`` a phrase, matched against consecutive tokens
* ``!word``     (word-list files only) never censor this word

Hyphenated tokens are split, so "f-f-fuck" and "ass-kicking" are caught while
"class-act" is not. A token Whisper has already masked ("f***", "sh*t") is
treated as a hit.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Three tiers. "low" strictness censors only STRONG, "medium" adds MODERATE,
# "high" (the default) adds MILD.
STRONG = [
    "*fuck*",
    "fuk",
    "fukin",
    "fuckin",
    "fck",
    "fcking",
    "*shit*",
    "cunt*",
    "nigger*",
    "nigga*",
    "chink",
    "spic",
    "wetback*",
    "kike*",
    "gook*",
    "faggot*",
    "fag",
]
MODERATE = [
    "bitch*",
    "sonofabitch*",
    "ass",
    "*asshole*",
    "dumbass*",
    "jackass*",
    "badass*",
    "smartass*",
    "fatass*",
    "kickass",
    "hardass",
    "asshat*",
    "asswipe*",
    "dick",
    "dickhead*",
    "cock",
    "cocksucker*",
    "pussy",
    "pussies",
    "bastard*",
    "whore*",
    "slut*",
    "douche*",
    "douchebag*",
    "piss",
    "pissed",
    "pissing",
    "prick",
    "twat*",
    "wanker*",
    "tits",
    "titties",
]
MILD = [
    "damn",
    "damned",
    "damnit",
    "dammit",
    "goddamn*",
    "goddam",
    "god damn",
    "hell",
    "crap",
    "crappy",
]

# Words that a glob above would catch but are clean.
DEFAULT_ALLOW = {"shitake", "shiitake", "cockpit", "cocktail", "peacock", "hancock"}

LEVELS = {
    "low": STRONG,
    "medium": STRONG + MODERATE,
    "high": STRONG + MODERATE + MILD,
}

_EDGE = re.compile(r"^[^a-z0-9*]+|[^a-z0-9*]+$")
_MASKED = re.compile(r"^[a-z]+\*+[a-z*]*$")


def normalize(token: str) -> str:
    """Lower-case a token and strip surrounding punctuation."""
    t = token.lower().replace("’", "'").replace("‘", "'")
    t = t.replace("'", "")
    return _EDGE.sub("", t)


def _parts(token: str) -> List[str]:
    """A normalised token plus its hyphen-separated parts."""
    norm = normalize(token)
    if not norm:
        return []
    pieces = [normalize(p) for p in re.split(r"[-–—/]+", token)]
    pieces = [p for p in pieces if p]
    joined = "".join(pieces)
    out = [joined] + (pieces if len(pieces) > 1 else [])
    return list(dict.fromkeys(out))


@dataclass
class _Pattern:
    raw: str
    tokens: Tuple[str, ...]

    def token_matches(self, pat: str, word: str) -> bool:
        if any(ch in pat for ch in "*?["):
            return fnmatch.fnmatchcase(word, pat)
        return word in (pat, pat + "s", pat + "es")


@dataclass
class Hit:
    first: int  # index of the first token
    last: int  # index of the last token (inclusive)
    pattern: str


@dataclass
class Matcher:
    patterns: List[_Pattern] = field(default_factory=list)
    allow: set = field(default_factory=set)

    @classmethod
    def build(
        cls,
        strictness: str = "high",
        extra: Iterable[str] = (),
        allow: Iterable[str] = (),
    ) -> "Matcher":
        if strictness not in LEVELS:
            raise ValueError(f"strictness must be one of {sorted(LEVELS)}")
        raw = list(LEVELS[strictness]) + [e for e in extra if e]
        pats = []
        for r in dict.fromkeys(r.strip().lower() for r in raw):
            toks = tuple(normalize(t) if "*" not in t else t for t in r.split())
            if toks:
                pats.append(_Pattern(r, toks))
        # Longest phrases first so "god damn" wins over "damn".
        pats.sort(key=lambda p: -len(p.tokens))
        return cls(pats, set(DEFAULT_ALLOW) | {normalize(a) for a in allow})

    @classmethod
    def from_file(cls, path: Optional[str], strictness: str = "high") -> "Matcher":
        extra, allow = [], []
        if path:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                (allow if line.startswith("!") else extra).append(line.lstrip("!"))
        return cls.build(strictness, extra, allow)

    def word_hit(self, token: str) -> Optional[str]:
        """Return the matching pattern for a single token, or None."""
        parts = _parts(token)
        if not parts or parts[0] in self.allow:
            return None
        if _MASKED.match(parts[0]):
            return "masked"
        for p in self.patterns:
            if len(p.tokens) == 1 and any(
                w not in self.allow and p.token_matches(p.tokens[0], w) for w in parts
            ):
                return p.raw
        return None

    def find(self, tokens: Sequence[str]) -> List[Hit]:
        """Find hits in a token sequence; phrases cover all their tokens."""
        norm = [normalize(t) for t in tokens]
        hits: List[Hit] = []
        i = 0
        while i < len(tokens):
            matched = None
            for p in self.patterns:
                n = len(p.tokens)
                if n == 1 or i + n > len(tokens):
                    continue
                if all(p.token_matches(p.tokens[k], norm[i + k]) for k in range(n)):
                    matched = Hit(i, i + n - 1, p.raw)
                    break
            if matched is None:
                pat = self.word_hit(tokens[i])
                if pat:
                    matched = Hit(i, i, pat)
            if matched:
                hits.append(matched)
                i = matched.last + 1
            else:
                i += 1
        return hits

    def censor_text(self, text: str) -> Tuple[str, int]:
        """Mask hits in free text (subtitle lines). Returns (text, hit count)."""
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        tokens = [text[a:b] for a, b in spans]
        hits = self.find(tokens)
        chars = list(text)
        for h in hits:
            for k in range(h.first, h.last + 1):
                a, b = spans[k]
                for j in range(a, b):
                    if chars[j].isalnum():
                        chars[j] = "*"
        return "".join(chars), len(hits)
