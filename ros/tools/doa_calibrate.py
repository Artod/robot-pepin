"""The microphone array's front and sense of rotation, for the speaker gaze.

Nothing moves: the tool only listens to the board's audio link. A person stands about 1.5 m from
the cart in FRONT of it, then on its LEFT, BEHIND it and on its RIGHT, and at each place says one
phrase three times (half a second of voice or more each, a pause between). Each utterance's
direction is the circular mean of the array's speech-flagged readings while it lasted; a place's
reading is the circular mean of its utterances (their spread is printed). The four readings give
``front_deg`` (the FRONT reading), ``sign`` (+1 when the LEFT reads a quarter turn above it) and
the quarters the speaker gaze interpolates between (the array sits off the cart's centre, so they
are not 90 deg apart), refused when a reading is nearer another quarter than its own
(:func:`pepin.expression.doa.fit_doa_map`). A chime on this Mac marks each place's end.

``--write`` puts the four readings, the front and the sign into config/expression.json's
``speaker.doa`` with the date; ``--readings F L B R`` fits four numbers read some other way (no
audio).

``--source mac`` (2026-10-14) measures the Mac's own direction instead (:mod:`pepin.audio_doa`:
SRP-PHAT on the board's raw microphone frames, the 6-channel firmware and ``raw_mics`` in
config/audio.json; Silero v6 on the processed frames picks the voiced chunks as the ears do): the
same four places, the readings the Mac's, written to ``speaker.doa.mac_readings``; the chip's
calibration stays. The two frames differ (the Mac's angles are the geometry's, 0 along +x), so
each source has its own four readings; the fit is refused the same way, and four Mac readings far
from the chip's up to one rotation or mirror say the raw channels are not in the geometry's
order (config/audio.json ``array``).

    uv run python ros/tools/doa_calibrate.py               # listen, print the fit
    uv run python ros/tools/doa_calibrate.py --write       # ...and write it
    uv run python ros/tools/doa_calibrate.py --readings 182 271 3 92
    uv run python ros/tools/doa_calibrate.py --source mac --write
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pepin.audio_link import AUDIO_PORT, RATE, AudioClient, AudioFrame, DoaReading, board_host
from pepin.audio_signal import circular_mean_deg, level_dbfs
from pepin.expression.doa import Calibration, fit_doa_map, wrap_deg

REPO = Path(__file__).resolve().parents[2]
PLACES = ("FRONT", "LEFT", "BEHIND", "RIGHT")
CHIME = "/System/Library/Sounds/Glass.aiff"
CONFIG = REPO / "config" / "expression.json"


class Link(Protocol):
    """What the utterance finder reads: the processed frames and the directions."""

    def frames(self) -> Iterator[AudioFrame]:
        """Every frame as it arrives, until the link closes."""
        ...

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The readings within ``t0..t1`` (board clock)."""
        ...


class MacLink:
    """The audio link read for the Mac's direction: the processed frames through Silero into
    the series the Mac's source reads (as the ears feed it), the raw frames into the source on
    a thread of their own; the directions are the Mac's readings."""

    def __init__(self, client: AudioClient, *, vad_model: str, args: argparse.Namespace) -> None:
        """``vad_model`` as ``barge_in.vad_model`` names it (silero-v6, the ears' own)."""
        import numpy as np

        from pepin.audio_doa import MicDoa, SrpPhat
        from pepin.expression.tracker import SpeakerSettings
        from pepin.voice_cascade.barge_in import SileroVad, SpeechSeries, model_path

        if client.raw_channels == 0:
            raise SystemExit(
                "the board sends no raw frames (hello: raw_channels 0): the 6-channel firmware"
                " and raw_mics in config/audio.json are needed for --source mac"
            )
        knobs = SpeakerSettings.load()
        self._client = client
        self._vad = SileroVad(model_path(vad_model))
        self._series = SpeechSeries(chunk_s=self._vad.chunk / RATE)
        self.doa = MicDoa(
            SrpPhat.load(),
            self._series,
            voice_p_min=knobs.mac_voice_p_min,
            peak_ratio=knobs.mac_peak_ratio,
            reading_s=knobs.mac_reading_s,
        )
        self._buffer: Any = np.zeros(0, dtype=np.float32)
        self._np = np
        threading.Thread(target=self._pump_raw, daemon=True, name="calibrate-raw").start()

    def frames(self) -> Iterator[AudioFrame]:
        """The processed frames, each run through Silero on the way (the chunk's probability
        stamped with the laptop clock of its last sample, as the ears do)."""
        np = self._np
        for frame in self._client.frames():
            now = time.monotonic()
            samples = np.frombuffer(frame.pcm, dtype="<i2").astype(np.float32) / 32768
            self._buffer = np.concatenate([self._buffer, samples])
            n = self._vad.chunk
            while len(self._buffer) >= n:
                p = self._vad(self._buffer[:n])
                self._buffer = self._buffer[n:]
                self._series.add(now - len(self._buffer) / RATE, p)
            yield frame

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The Mac's readings within ``t0..t1`` (board clock)."""
        return self.doa.doa_between(t0, t1)

    def _pump_raw(self) -> None:
        while self._client.connected:
            raw = self._client.next_raw(timeout_s=1.0)
            if raw is not None:
                self.doa.feed(raw, time.monotonic())


@dataclass(frozen=True)
class Utterance:
    """One utterance's direction: the mean of its speech readings, how many, its loudness."""

    deg: float
    speech: int
    loud_db: float


def utterances(
    client: Link, threshold_db: float, hold_s: float, gap_s: float
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
    """The four places' readings, heard (the chip's, or the Mac's under ``--source mac``)."""

    def connect() -> tuple[AudioClient, Link]:
        client = AudioClient(args.host, args.port).start(listen=True)
        if args.source == "mac":
            return client, MacLink(client, vad_model=args.vad_model, args=args)
        return client, client

    client, link = connect()
    heard = utterances(link, args.threshold_db, args.hold_s, args.gap_s)
    readings: list[float] = []

    def next_said() -> Utterance:
        # the board closes a fresh link once early on (the cascade sees the same and reconnects)
        nonlocal client, heard
        while True:
            try:
                return next(heard)
            except StopIteration:
                print("  (audio link lost: reconnecting)", flush=True)
                client, link = connect()
                heard = utterances(link, args.threshold_db, args.hold_s, args.gap_s)

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


def write(
    fit: Calibration, readings: list[float], path: Path = CONFIG, source: str = "chip"
) -> None:
    """``speaker.doa`` of config/expression.json set to the fit: the front, the sign and the
    four readings the map interpolates between, with the date; ``source`` mac: the Mac's own
    four readings (``mac_readings``), the chip's untouched."""
    data = json.loads(path.read_text())
    doa = data["speaker"]["doa"]
    places = ("front", "left", "behind", "right")
    quarters = {p: round(v % 360.0, 1) for p, v in zip(places, readings, strict=True)}
    f, left, b, r = (round(v) for v in readings)
    tool = "ros/tools/doa_calibrate.py" + (" --source mac" if source == "mac" else "")
    measured = (
        f"MEASURED {time.strftime('%Y-%m-%d')} ({tool}): front {f}, left"
        f" {left}, behind {b}, right {r} deg; quarter-turn misses "
        + ", ".join(f"{m:+.0f}" for m in fit.misses_deg)
    )
    if source == "mac":
        doa["mac_readings"] = quarters
        doa["mac_measured"] = measured
    else:
        doa["front_deg"] = round(fit.doa_map.front_deg, 1)
        doa["sign"] = fit.doa_map.sign
        doa["readings"] = quarters
        doa["measured"] = measured
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
        "--source", choices=("chip", "mac"), default="chip",
        help="whose direction: the chip's DOA_VALUE (speaker.doa.readings) or the Mac's own from"
        " the raw microphones (speaker.doa.mac_readings; needs the 6-channel firmware)",
    )  # fmt: skip
    ap.add_argument(
        "--vad-model", default="silero-v6",
        help="--source mac: the Silero model that picks the voiced chunks (barge_in.vad_model)",
    )  # fmt: skip
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
        write(fit, readings, source=args.source)
        block = "speaker.doa.mac_readings" if args.source == "mac" else "speaker.doa"
        print(f"written to {CONFIG.relative_to(REPO)} {block}")
    else:
        print("--write puts it into config/expression.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
