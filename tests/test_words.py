import pytest

from censorbot.words import Matcher, normalize

M = Matcher.build("high")

# Every one of these was muted by the old substring matcher.
CLEAN = [
    "hello",
    "Hello,",
    "class",
    "assume",
    "passage",
    "shell",
    "Michelle",
    "assistant",
    "Hellen",
    "cocktail",
    "Dickens",
    "Massachusetts",
    "Hancock",
    "grass",
    "bass",
    "embassy",
    "shitake",
    "glass",
    "cassette",
    "Hitchcock",
    "peacock",
    "spice",
    "scrapbook",
    "therapist",
    "document",
    "skills",
    "basement",
    "class-act",
    "cockpit",
    "assassin",
    "harass",
    "compass",
    "Shellfish",
    "hellos",
    "Dickinson",
    "titanic",
]

DIRTY = [
    "fuck",
    "Fuck!",
    "fucking",
    "fuckin'",
    "motherfucker",
    "motherfuckin'",
    "f-f-fuck",
    "shit",
    "Shit.",
    "bullshit",
    "shitty",
    "horseshit",
    "ass",
    "asses",
    "asshole",
    "dumbass",
    "jackass",
    "ass-kicking",
    "bitch",
    "bitches",
    "son-of-a-bitch",
    "damn",
    "goddamn",
    "hell",
    "dick",
    "cock",
    "pussy",
    "bastard",
    "f***",
    "sh*t",
    "crap",
    "pissed",
    "cunt",
]


@pytest.mark.parametrize("word", CLEAN)
def test_clean_words_pass(word):
    assert M.word_hit(word) is None


@pytest.mark.parametrize("word", DIRTY)
def test_dirty_words_hit(word):
    assert M.word_hit(word) is not None


def test_phrase_covers_both_tokens():
    hits = M.find("oh god damn it".split())
    assert [(h.first, h.last) for h in hits] == [(1, 2)]


def test_strictness_levels():
    low = Matcher.build("low")
    assert low.word_hit("fuck") and low.word_hit("shit")
    assert low.word_hit("damn") is None and low.word_hit("ass") is None
    med = Matcher.build("medium")
    assert med.word_hit("ass") and med.word_hit("damn") is None


def test_wordlist_file(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("# comment\nfrick*\njesus christ\n!hell\n")
    m = Matcher.from_file(str(f))
    assert m.word_hit("fricking")
    assert m.word_hit("hell") is None
    hits = m.find("oh jesus christ no".split())
    assert [(h.first, h.last) for h in hits] == [(1, 2)]


def test_censor_text_keeps_punctuation():
    text, n = M.censor_text("What the hell, man? <i>Shit!</i>")
    assert n == 2
    assert "****," in text and "hell" not in text and "Shit" not in text
    assert text.startswith("What the ")


def test_normalize():
    assert normalize("“Fuck,”") == "fuck"
    assert normalize("fuckin’") == "fuckin"
