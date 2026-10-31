import numpy as np

from censorbot.align import _viterbi, refine_edges, SR


def test_viterbi_follows_peaks():
    # 3 tokens peaking at frames 2, 5 and 9 over a blank background.
    T, C = 12, 4
    em = np.full((T, C), -8.0)
    em[:, 0] = -0.1
    for frame, tok in ((2, 1), (5, 2), (9, 3)):
        em[frame, tok] = -0.05
        em[frame, 0] = -6
    path = _viterbi(em, [1, 2, 3], 0)
    assert [f for _, f in path] == [2, 5, 9]


def test_refine_edges_extends_to_voice_onset():
    audio = np.zeros(SR * 2, dtype=np.float32) + 1e-4
    audio[int(0.50 * SR) : int(1.00 * SR)] = 0.3  # the word really spans 0.50-1.00
    s, e = refine_edges(audio, 0.0, 0.58, 0.92, lo=0.3, hi=1.3)
    assert abs(s - 0.50) <= 0.011 and abs(e - 1.00) <= 0.011


def test_refine_edges_respects_neighbours():
    audio = np.zeros(SR * 2, dtype=np.float32) + 1e-4
    audio[int(0.40 * SR) : int(1.20 * SR)] = 0.3  # continuous speech
    s, e = refine_edges(audio, 0.0, 0.6, 0.9, lo=0.55, hi=0.95)
    assert s >= 0.55 and e <= 0.95
