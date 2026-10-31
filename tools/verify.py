#!/usr/bin/env python3
"""Measure how clean a CensorBot output is.

Needs a dual-audio output (censored track first, original second) and the
--report JSON from the same run.

For every cut it checks:
  * voice reduction: how far the original dialogue dropped in what a viewer
    hears (all channels), in dB, measured by projecting the censored mix onto
    the original centre/vocal signal inside the cut;
  * leaks: re-transcribes the censored audio around the cut and flags any
    censored word that is still recognisable;
  * collateral: clean words heard in the original window that are missing
    from the censored window.
It also checks the censored track is sample-aligned with the original, and can
write a listening reel: original then censored for each cut.

  python tools/verify.py out.mp4 report.json --reel reel.m4a
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from censorbot.media import probe  # noqa: E402
from censorbot.words import Matcher, normalize  # noqa: E402

PAD = 1.5


def decode(path, track, start, dur, channels=None, sr=48000):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{max(0, start):.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        path,
        "-map",
        f"0:a:{track}",
        "-f",
        "f32le",
        "-ar",
        str(sr),
    ]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd.append("-")
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype="<f4")
    return a.reshape(-1, channels) if channels and channels > 1 else a


class ASR:
    def __init__(self):
        import mlx_whisper

        self.mw = mlx_whisper

    def text(self, audio16k):
        r = self.mw.transcribe(
            np.ascontiguousarray(audio16k.reshape(-1), dtype=np.float32),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="en",
            condition_on_previous_text=False,
            verbose=None,
        )
        return r["text"].strip()

    def words(self, audio16k, t0):
        r = self.mw.transcribe(
            np.ascontiguousarray(audio16k.reshape(-1), dtype=np.float32),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            word_timestamps=True,
            language="en",
            condition_on_previous_text=False,
            verbose=None,
        )
        return [
            (t0 + w["start"], t0 + w["end"], w["word"].strip())
            for s in r.get("segments", [])
            for w in s.get("words", [])
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("report")
    ap.add_argument("--reel", help="write an original/censored listening reel here (.m4a)")
    ap.add_argument("--max", type=int, default=0, help="only check the first N cuts")
    ap.add_argument("--out", help="write per-cut results as JSON")
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())
    cuts = rep["cuts"][: args.max or None]
    info = probe(args.video)
    if len(info.audio) < 2:
        sys.exit("verify needs the dual-audio output (do not use --single-audio)")
    orig_stream = info.audio[1]
    ch = orig_stream.channels
    matcher = Matcher.build(rep.get("strictness", "high"))
    asr = ASR()

    results, reel = [], []
    gap = np.zeros(int(0.35 * 48000), np.float32)
    for i, c in enumerate(cuts):
        s, e = c["start"], c["end"]
        w0, dur = max(0.0, s - PAD), (e - s) + 2 * PAD
        o = decode(args.video, 1, w0, dur, ch)
        k = decode(args.video, 0, w0, dur, ch)
        n = min(len(o), len(k))
        o, k = o[:n], k[:n]
        a, b = int((s - w0) * 48000), int((e - w0) * 48000)
        mix_o, mix_k = o[a:b].sum(1), k[a:b].sum(1)
        drop = 20 * np.log10(
            (np.sqrt(np.mean(mix_o**2)) + 1e-9) / (np.sqrt(np.mean(mix_k**2)) + 1e-9)
        )

        # Leak test: the cut region on its own. Whisper given the whole window
        # "hears" a censored word from context alone (it does so even when the
        # cut is digital silence), so the window is only used for collateral.
        mono_o = decode(args.video, 1, w0, dur, 1, 16000)
        mono_k = decode(args.video, 0, w0, dur, 1, 16000)
        a16, b16 = int((s - w0) * 16000), int((e - w0) * 16000)
        pad = np.zeros(8000, np.float32)
        heard_orig = asr.text(np.concatenate([pad, mono_o[a16:b16], pad]))
        heard_cens = asr.text(np.concatenate([pad, mono_k[a16:b16], pad]))
        testable = bool(matcher.find(heard_orig.split()))
        leaks = [t for t in heard_cens.split() if matcher.word_hit(t)]

        # Collateral: clean words right next to the cut that the original has
        # and the censored version lost.
        wo, wk = asr.words(mono_o, w0), asr.words(mono_k, w0)
        heard_k = {normalize(t) for _, _, t in wk}
        near = [
            t
            for (ts, te, t) in wo
            if (s - 0.6 < te < s - 0.05 or e + 0.05 < ts < e + 0.6)
            and not matcher.word_hit(t)
            and normalize(t)
        ]
        lost = [t for t in near if normalize(t) not in heard_k]
        results.append(
            {
                "start": s,
                "end": e,
                "words": c["words"],
                "source": c["source"],
                "mix_drop_db": round(float(drop), 1),
                "orig_region_heard": heard_orig,
                "censored_region_heard": heard_cens,
                "testable": testable,
                "leaks": leaks,
                "clean_lost": lost,
                "clean_checked": len(near),
            }
        )
        flag = "LEAK " if leaks else ("LOST " if lost else "ok   ")
        print(
            f"{flag}{i + 1:4d} {s:8.2f}s {' / '.join(c['words'])[:28]:28s} mix -{drop:4.1f} dB "
            f"orig={heard_orig[:24]!r} cens={heard_cens[:24]!r} lost={lost}",
            flush=True,
        )
        if args.reel:
            st = decode(args.video, 1, w0, dur, 2)
            sk = decode(args.video, 0, w0, dur, 2)
            m = min(len(st), len(sk))
            reel += [
                st[:m],
                np.zeros((len(gap), 2), np.float32),
                sk[:m],
                np.zeros((len(gap) * 3, 2), np.float32),
            ]

    # Sync: correlate the two tracks away from any cut.
    lags = []
    for t in np.linspace(120, max(150, info.duration - 120), 5):
        if any(c["start"] - 3 < t < c["end"] + 3 for c in rep["cuts"]):
            t += 7
        o = decode(args.video, 1, t, 2.0, ch)[:, 0]
        k = decode(args.video, 0, t, 2.0, ch)[:, 0]
        m = min(len(o), len(k))
        if m == 0 or np.std(o[:m]) < 1e-5:
            continue
        seg = slice(int(0.5 * 48000), int(1.5 * 48000))
        xc = np.correlate(k[:m], o[seg], "valid")
        lags.append(int(np.argmax(xc)) - seg.start)

    testable = [r for r in results if r["testable"]]
    summary = {
        "cuts_checked": len(results),
        "cuts_where_original_region_is_recognisable": len(testable),
        "of_those_still_recognisable_after": sum(1 for r in testable if r["leaks"]),
        "any_profanity_heard_in_censored_region": sum(1 for r in results if r["leaks"]),
        "mix_drop_db_median": round(
            float(np.median([r["mix_drop_db"] for r in results] or [0])), 1
        ),
        "adjacent_clean_words_checked": sum(r["clean_checked"] for r in results),
        "adjacent_clean_words_lost": sum(len(r["clean_lost"]) for r in results),
        "sync_lags_samples": lags,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "cuts": results}, indent=2))
    if args.reel and reel:
        pcm = np.concatenate(reel).astype("<f4").tobytes()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                "-",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                args.reel,
            ],
            input=pcm,
            check=True,
        )
        print(f"reel: {args.reel} (each cut: original, short gap, censored)")


if __name__ == "__main__":
    main()
