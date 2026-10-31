import numpy as np

from censorbot.media import AudioStream
from censorbot.render import Cut, Renderer, merge_cuts, mute_curve

SR = 48000


def stream(channels=6, layout="5.1"):
    return AudioStream(
        index=1,
        order=0,
        codec="aac",
        channels=channels,
        layout=layout,
        sample_rate=SR,
        bit_rate=None,
        language="eng",
        title="",
    )


def make_51(seconds=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    voice = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    music = (0.1 * rng.standard_normal((n, 6))).astype(np.float32)
    x = music.copy()
    x[:, 2] += voice  # centre: dialogue
    x[:, 0] += 0.25 * voice  # some dialogue panned into front left
    x[:, 1] += 0.25 * voice
    return x, voice, music


def rms(a):
    return float(np.sqrt(np.mean(a**2)))


def test_mute_curve_is_smooth_and_bounded():
    t = np.arange(0, 2, 1 / SR)
    m = mute_curve(t, Cut(0.5, 1.0), 0.02)
    assert m.min() == 0 and m.max() == 1
    assert np.abs(np.diff(m)).max() < 0.002  # no step anywhere: no click
    assert m[int(0.75 * SR)] == 1 and m[int(0.2 * SR)] == 0


def test_centre_strategy_removes_voice_keeps_music():
    x, voice, music = make_51()
    cut = Cut(1.0, 2.0)
    r = Renderer(stream(), [cut], fade=0.02, separate=False)
    assert r.strategy == "centre"
    y = r.process(x.copy(), 0.0)
    core = slice(int(1.05 * SR), int(1.95 * SR))
    # Dialogue gone from the centre, and the panned copy subtracted from L/R.
    assert rms(y[core, 2]) < 1e-6
    v = voice[core]
    for ch in (0, 1):
        before = float(np.dot(x[core, ch], v) / np.dot(v, v))
        after = float(np.dot(y[core, ch], v) / np.dot(v, v))
        assert abs(after) < 0.25 * abs(before)  # panned voice down by at least 12 dB
    # Surrounds and LFE untouched; music in L/R kept.
    assert np.array_equal(y[:, 3:], x[:, 3:])
    assert rms(y[core, 0]) > 0.8 * rms(music[core, 0])
    # Outside the cut nothing changes at all.
    outside = np.r_[0 : int(0.97 * SR), int(2.03 * SR) : len(x)]
    assert np.array_equal(y[outside], x[outside])


def test_all_scope_silences_everything():
    x, _, _ = make_51()
    y = Renderer(stream(), [Cut(1.0, 2.0)], scope="all").process(x.copy(), 0.0)
    assert rms(y[int(1.05 * SR) : int(1.95 * SR)]) < 1e-6


def test_beep_replaces_voice():
    x, voice, _ = make_51()
    y = Renderer(stream(), [Cut(1.0, 2.0)], mode="beep", separate=False).process(x.copy(), 0.0)
    core = y[int(1.1 * SR) : int(1.9 * SR), 2]
    spec = np.abs(np.fft.rfft(core))
    freqs = np.fft.rfftfreq(len(core), 1 / SR)
    assert abs(freqs[spec.argmax()] - 1000) < 5


def test_chunks_never_split_a_cut():
    cuts = [Cut(19.99, 20.5), Cut(39.0, 41.0), Cut(59.9, 60.2)]
    r = Renderer(stream(), cuts, separate=False)
    bounds, pos = [], 0
    gen = r._chunk_sizes()
    for _ in range(6):
        pos += next(gen)
        bounds.append(pos / SR)
    for b in bounds:
        for c in cuts:
            assert not (c.start - r.fade - 0.05 < b < c.end + r.fade + 0.05), (b, c)


def test_chunked_render_equals_single_pass():
    x, _, _ = make_51(seconds=50)
    cuts = [Cut(19.95, 20.4), Cut(33.0, 33.5)]
    whole = Renderer(stream(), cuts, separate=False).process(x.copy(), 0.0)
    r = Renderer(stream(), cuts, separate=False)
    out, pos = [], 0
    for n in r._chunk_sizes():
        block = x[pos : pos + n].copy()
        if not len(block):
            break
        out.append(r.process(block, pos / SR))
        pos += len(block)
    assert np.allclose(np.concatenate(out), whole, atol=1e-6)


def test_merge_cuts():
    merged = merge_cuts([Cut(1.0, 1.2, ["a"]), Cut(1.3, 1.5, ["b"]), Cut(3, 3.2, ["c"])], 0.2)
    assert [(c.start, c.end) for c in merged] == [(1.0, 1.5), (3, 3.2)]
    assert merged[0].words == ["a", "b"]


def test_default_bitrate_has_headroom():
    from censorbot.media import default_bitrate

    s51 = stream()
    s51.bit_rate = 224_000
    assert default_bitrate(s51) == "640k"
    st = stream(2, "stereo")
    st.bit_rate = 128_000
    assert default_bitrate(st) == "256k"
