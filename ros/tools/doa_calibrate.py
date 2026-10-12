"""The microphone array's front and sense of rotation, for the speaker gaze.

Nothing moves: the tool only listens to the board's audio link. A person stands about 1.5 m from
the cart in FRONT of it, then on its LEFT, BEHIND it and on its RIGHT, and at each place says one
phrase three times (half a second of voice or more each, a pause between). Each utterance's
direction is the circular mean of the array's speech-flagged readings while it lasted; a place's
reading is the circular mean of its utterances (their spread is printed). The four readings give
``front_deg`` (the FRONT reading) and ``sign`` (+1 when the LEFT reads a quarter turn above it),
refused unless all four sit within 15 deg of their quarter turns
(:func:`pepin.expression.doa.fit_doa_map`). A chime on this Mac marks each place's end.

``--write`` puts the two numbers into config/expression.json's ``speaker.doa`` with the date and
the four readings; ``--readings F L B R`` fits four numbers read some other way (no audio).

    uv run python ros/tools/doa_calibrate.py               # listen, print the fit
    uv run python ros/tools/doa_calibrate.py --write       # ...and write it
    uv run python ros/tools/doa_calibrate.py --readings 182 271 3 92
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pepin.audio_link import AUDIO_PORT, AudioClient, board_host
from pepin.audio_signal import circular_mean_deg, level_dbfs
from pepin.expression.doa import Calibration, fit_doa_map, wrap_deg

REPO = Path(__file__).resolve().parents[2]
PLACES = ("FRONT", "LEFT", "BEHIND", "RIGHT")
CHIME = "/System/Library/Sounds/Glass.aiff"
CONFIG = REPO / "config" / "expression.json"


@dataclass(frozen=True)
class Utterance:
    """One utterance's direction: the mean of its speech readings, how many, its loudness."""

    deg: float
    speech: int
    loud_db: float


def utterances(
    client: AudioClient, threshold_db: float, hold_s: float, gap_s: float
) -> Iterator[Utterance]:
    """Every utterance the array hears: above ``threshold_db`` for ``hold_s`` or more, ended
    by ``gap_s`` of quiet; one with no speech-flagged reading is skipped."""
    onset: float | None = None
    last_loud = 0.0
    loudest = -120.0
    for frame in client.frames():
        level = level_dbfs(frame.pcm)
        t = frame.stamp_s
        if level >= threshold_db:
            if onset is None:
                onset, loudest = t, level
            last_loud, loudest = t + frame.duration_s, max(loudest, level)
            continue
        if onset is None or t - last_loud <= gap_s:
            continue
        start, end, onset = onset, last_loud, None
        if end - start < hold_s:
            continue
        speech = [float(r.deg) for r in client.doa_between(start, end) if r.speech]
        deg = circular_mean_deg(speech)
        if deg is not None:
            yield Utterance(deg, len(speech), loudest)


def spread_deg(values: list[float], centre: float) -> float:
    """The largest distance of ``values`` from ``centre`` round the circle."""
    return max((abs(wrap_deg(v - centre)) for v in values), default=0.0)


def listen(args: argparse.Namespace) -> list[float]:
    """The four places' readings, heard."""
    client = AudioClient(args.host, args.port).start(listen=True)
    heard = utterances(client, args.threshold_db, args.hold_s, args.gap_s)
    readings: list[float] = []

    def next_said() -> Utterance:
        # the board closes a fresh link once early on (the cascade sees the same and reconnects)
        nonlocal client, heard
        while True:
            try:
                return next(heard)
            except StopIteration:
                print("  (audio link lost: reconnecting)", flush=True)
                client = AudioClient(args.host, args.port).start(listen=True)
                heard = utterances(client, args.threshold_db, args.hold_s, args.gap_s)

    try:
        for place in PLACES:
            print(f"\n{place} of the cart, 1.5 m away: say the phrase {args.per_place} times")
            got: list[float] = []
            while len(got) < args.per_place:
                said = next_said()
                got.append(said.deg)
                print(
                    f"  {len(got)}: {said.deg:.0f} deg ({said.speech} speech readings,"
                    f" {said.loud_db:.0f} dBFS)",
                    flush=True,
                )
            mean = circular_mean_deg(got)
            if mean is None:
                raise SystemExit(f"{place}: the utterances cancel out; measure again")
            spread = spread_deg(got, mean)
            warn = "  (spread over 15 deg: a noisy room?)" if spread > 15.0 else ""
            print(f"  {place}: {mean:.0f} deg, spread {spread:.0f} deg{warn}", flush=True)
            readings.append(mean)
            subprocess.run(["afplay", CHIME], check=False, capture_output=True)
    finally:
        client.close()
    return readings


def write(fit: Calibration, readings: list[float], path: Path = CONFIG) -> None:
    """``speaker.doa`` of config/expression.json set to the fit, with the date and readings."""
    data = json.loads(path.read_text())
    doa = data["speaker"]["doa"]
    doa["front_deg"] = round(fit.doa_map.front_deg, 1)
    doa["sign"] = fit.doa_map.sign
    f, left, b, r = (round(v) for v in readings)
    doa["measured"] = (
        f"MEASURED {time.strftime('%Y-%m-%d')} (ros/tools/doa_calibrate.py): front {f}, left"
        f" {left}, behind {b}, right {r} deg; quarter-turn misses "
        + ", ".join(f"{m:+.0f}" for m in fit.misses_deg)
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--host", default=board_host())
    ap.add_argument("--port", type=int, default=AUDIO_PORT)
    ap.add_argument("--per-place", type=int, default=3)
    ap.add_argument("--threshold-db", type=float, default=-40.0, help="a voice's level, dBFS")
    ap.add_argument("--hold-s", type=float, default=0.5, help="the shortest utterance")
    ap.add_argument("--gap-s", type=float, default=0.4, help="the quiet that ends one")
    ap.add_argument("--readings", type=float, nargs=4, metavar=("FRONT", "LEFT", "BEHIND", "RIGHT"))
    ap.add_argument("--write", action="store_true", help="write the fit to config/expression.json")
    ap.add_argument(
        "--tol-deg", type=float, default=None,
        help="accept readings this far off their quarter turn (default: the fit's own; the camera"
        " refines the aim, so 30 is fine for an off-centre array)",
    )  # fmt: skip
    args = ap.parse_args()
    readings = list(args.readings) if args.readings else listen(args)
    front, left, back, right = readings
    fit = (
        fit_doa_map(front, left, back, right)
        if args.tol_deg is None
        else fit_doa_map(front, left, back, right, tol_deg=args.tol_deg)
    )
    print("\n" + fit.text())
    if fit.refusal is not None:
        return 1
    if args.write:
        write(fit, readings)
        print(f"written to {CONFIG.relative_to(REPO)} speaker.doa")
    else:
        print("--write puts it into config/expression.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
