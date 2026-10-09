"""Barge-in detectors replayed on recorded conversations: per session, the person's
interruptions caught, the false barge-ins and the median reaction after the person's onset.

    uv run python scripts/barge_in_eval.py data/voice/20261009 \\
        [--sessions cascade_150215 ...] [--detectors live level vad words] \\
        [--set barge_in.min_dbfs=-38 ...] [--whisper-ms 300] [--fresh] [-v]

``live`` is what the run itself did (its interrupted turns judged against the truth). The truth
(:mod:`pepin.voice_cascade.barge_eval`: the loud moments of each reply, read by Whisper) is
cached as ``<day>/barge_truth/<session>.json``; ``--fresh`` reads them again. A session is
taken when it has a mic recording and a turn that played; the bench's replays (no room, no
echo, woken "always") are skipped unless named in ``--sessions``. Each cell: caught/positives,
false triggers, median reaction in ms.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pepin.voice_cascade.barge_eval import (
    OfflineReader,
    Recording,
    Sound,
    Tally,
    Trigger,
    Verdict,
    Window,
    ground_truth,
    is_speech,
    judge,
    replay,
)
from pepin.voice_cascade.barge_in import BargeInDetector, build_detector
from pepin.voice_cascade.config import CascadeConfig

DETECTORS = ("live", "level", "vad", "words")
Detailed = Callable[[bytes], tuple[str, float, float]]


def detailed_whisper(repo: str, language: str) -> Detailed:
    """Whisper on the Mac's GPU giving the text, the lowest no-speech probability and the
    highest mean log-probability of its segments."""
    import mlx_whisper
    import numpy as np

    def read(pcm: bytes) -> tuple[str, float, float]:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        out: dict[str, Any] = mlx_whisper.transcribe(
            audio, path_or_hf_repo=repo, language=language, condition_on_previous_text=False
        )
        segments = out.get("segments", [])
        no_speech = min((float(s["no_speech_prob"]) for s in segments), default=1.0)
        logprob = max((float(s["avg_logprob"]) for s in segments), default=-9.0)
        return str(out.get("text", "")).strip(), no_speech, logprob

    read(b"\x00\x00" * 8000)  # load and warm
    return read


class Models:
    """Whisper loaded on first use, once for the run."""

    def __init__(self, config: CascadeConfig, whisper_ms: float | None) -> None:
        """``whisper_ms``: the words detector's answers this late (None: as measured)."""
        self.config, self.whisper_ms = config, whisper_ms
        self._detailed: Detailed | None = None
        self.words: OfflineReader | None = None

    def detailed(self) -> Detailed:
        """Whisper with its confidences, for the truth."""
        if self._detailed is None:
            self._detailed = detailed_whisper(self.config.whisper_repo, self.config.language)
        return self._detailed

    def reader(self) -> OfflineReader:
        """Whisper for the words detector."""
        if self.words is None:
            from pepin.voice_live.wake import MlxWhisper

            cfg = self.config
            whisper = MlxWhisper(cfg.whisper_repo, cfg.language, cfg.whisper_prompt)
            self.words = OfflineReader(whisper.transcribe, self.whisper_ms)
        return self.words


def truth_of(
    recording: Recording, cache: Path, read: Callable[[], Detailed], fresh: bool
) -> dict[int, list[Sound]]:
    """The recording's truth, from the cache or read now (and cached)."""
    path = cache / f"{recording.name}.json"
    if path.exists() and not fresh:
        rows = json.loads(path.read_text())
        truth = {int(turn): [Sound(**s) for s in sounds] for turn, sounds in rows.items()}
        for w in recording.windows:  # the rule may have changed since the reading
            for sound in truth.get(w.turn, []):
                sound.speech = is_speech(sound, w.reply)
        return truth
    truth = ground_truth(recording, read())
    cache.mkdir(parents=True, exist_ok=True)
    rows = {turn: [s.row() for s in sounds] for turn, sounds in truth.items()}
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return truth


def triggers_of(
    detector: str, rec: Recording, config: CascadeConfig, models: Models
) -> dict[int, Trigger]:
    """Each window's first barge-in: the run's own (``live``) or a replay's."""
    if detector == "live":
        return {w.turn: Trigger(w.end, "live") for w in rec.windows if w.outcome == "interrupted"}
    cfg = config.with_changes(barge_in_detector=detector)

    def make(own_text: Callable[[], str]) -> BargeInDetector:
        reader = models.reader() if detector == "words" else None
        return build_detector(cfg, reader=reader, own_text=own_text, played=rec.played)

    return replay(rec, config, make)


def window_line(w: Window, verdicts: dict[str, Verdict], sounds: list[Sound]) -> str:
    """One window's verdicts and the person's words in it."""
    cells = []
    for name, v in verdicts.items():
        at = f"@{v.trigger.t - w.start:+.2f}s" if v.trigger is not None else ""
        cells.append(f"{name}:{v.outcome}{at}")
    spoken = [f"{s.start - w.start:+.1f}s {s.text!r}" for s in sounds if s.speech]
    return f"    t{w.turn:<3}{w.outcome[:5]} {w.end - w.start:5.1f}s  {'  '.join(cells)}  {spoken}"


def row(name: str, tallies: dict[str, Tally]) -> str:
    """A table row: windows, positives, then a cell per detector."""
    first = next(iter(tallies.values()))
    cells = "".join(f"{t.cell():<20}" for t in tallies.values())
    return f"{name:<16}{first.windows:>4}{first.positives:>4}  {cells}"


def main(argv: list[str] | None = None) -> int:
    """Replay the detectors and print the table."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("day", type=Path, help="a day's directory: data/voice/YYYYMMDD")
    parser.add_argument("--sessions", nargs="*", help="cascade_* names (default: all live ones)")
    parser.add_argument("--detectors", nargs="*", default=list(DETECTORS), choices=DETECTORS)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="section.key=value"
    )
    parser.add_argument("--whisper-ms", type=float, help="Whisper's answer delay (default: as run)")
    parser.add_argument("--fresh", action="store_true", help="read the truth again")
    parser.add_argument("-v", "--verbose", action="store_true", help="every window's verdicts")
    args = parser.parse_args(argv)

    config = CascadeConfig.load(overrides=args.overrides)
    models = Models(config, args.whisper_ms)
    names = args.sessions or sorted(
        d.name for d in args.day.glob("cascade_*") if (d / "in.wav").exists()
    )
    recordings = [Recording.load(args.day / name) for name in names]
    recordings = [r for r in recordings if r.windows and (args.sessions or r.wake != "always")]

    totals = {d: Tally() for d in args.detectors}
    print("each cell: caught/positives, false triggers (F), median reaction after the onset")
    print(f"{'session':<16}{'win':>4}{'pos':>4}  " + "".join(f"{d:<20}" for d in args.detectors))
    for rec in recordings:
        truth = truth_of(rec, args.day / "barge_truth", models.detailed, args.fresh)
        tallies = {d: Tally() for d in args.detectors}
        verdicts: dict[int, dict[str, Verdict]] = {w.turn: {} for w in rec.windows}
        for detector in args.detectors:
            triggers = triggers_of(detector, rec, config, models)
            for w in rec.windows:
                sounds = truth.get(w.turn, [])
                verdict = judge(w, sounds, triggers.get(w.turn))
                verdicts[w.turn][detector] = verdict
                tallies[detector].add(verdict, any(s.speech for s in sounds))
            totals[detector].merge(tallies[detector])
        print(row(rec.name, tallies))
        if args.verbose:
            for w in rec.windows:
                print(window_line(w, verdicts[w.turn], truth.get(w.turn, [])))
    print(row("TOTAL", totals))
    if models.words is not None and models.words.ms:
        ms = sorted(models.words.ms)
        print(f"words: {len(ms)} Whisper calls, median {ms[len(ms) // 2]:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
