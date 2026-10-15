"""End-of-turn methods replayed on recorded utterances: per method, the turns cut before the
person's last word (and whether the loop would have joined the pieces) and the close after it.

    uv run --group stt-gigaam python scripts/turn_eval.py data/voice/20261011 data/voice/20261012 \\
        [--live-cuts scratch/prosody_1008] [--pauses 0.25 0.5 0.8] [--methods silence-0.40 ...] \\
        [--read-ms 60] [--set ears.continuation_s=0.8 ...] [--cache DIR] [-v]

A path is a day (its ``cascade_*`` conversations) or one conversation; a run in progress or one
whose turns an end-of-turn model closed is left out (name the finished ones to pin a set).
The sets: every turn as spoken, then each turn with a pause of each ``--pauses`` length
inserted at up to two word boundaries (:mod:`pepin.voice_cascade.turn_eval`). The transcriber
is GigaAM v3 (the cascade's engine, its token times placing the pauses); its early reads land
``--read-ms`` after the ask (``--judge-ms`` more with the Namo model). Each row: clips cut /
clips (cuts: merged by the continuation, split), the close after the last word p50 / p90 /
mean, and what closed the turns. Then each text judge on the early reads: at the true end and
in an inserted pause, complete / unsure / incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pepin.audio_link import RATE
from pepin.overlay import read_json
from pepin.voice_cascade.barge_in import SileroVad, model_path
from pepin.voice_cascade.config import CascadeConfig, apply_overrides
from pepin.voice_cascade.turn import SmartTurn, TurnWorker
from pepin.voice_cascade.turn_eval import (
    METHODS,
    Clip,
    Method,
    Result,
    Tally,
    TimedReader,
    Word,
    boundaries,
    replay,
    session_clips,
    text_rows,
    text_table,
    uses_smart_turn,
    wav_clips,
    with_pause,
)
from pepin.voice_cascade.turn_text import (
    NamoClassifier,
    NamoJudge,
    RulesJudge,
    TextJudge,
)
from pepin.voice_live.wake import clean_transcript

ROOT = Path(__file__).resolve().parents[1]


class GigaAmTimes:
    """GigaAM v3 through onnx-asr with token times: the transcript and the words' starts."""

    def __init__(self, model: str = "gigaam-v3-e2e-rnnt", threads: int = 4) -> None:
        """Loaded once (the Hugging Face cache)."""
        from pepin.voice_cascade.stt.runtime import CPU, onnx_runtime

        load_model, onnxruntime = onnx_runtime()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        self._model: Any = load_model(
            model, sess_options=options, providers=[CPU]
        ).with_timestamps()
        self.ms: list[float] = []

    def _run(self, pcm: bytes) -> Any:
        from pepin.voice_cascade.stt.runtime import samples

        t0 = time.perf_counter()
        out = self._model.recognize(samples(pcm), sample_rate=RATE)
        self.ms.append((time.perf_counter() - t0) * 1000)
        return out

    def transcribe(self, pcm: bytes) -> str:
        """The words of ``pcm``."""
        return clean_transcript(str(self._run(pcm).text))

    def words(self, pcm: bytes) -> list[Word]:
        """The words of ``pcm`` with their first token's time (a token opening with a space
        opens a word; punctuation joins the word before)."""
        out = self._run(pcm)
        words: list[list[Any]] = []
        for token, t in zip(out.tokens, out.timestamps, strict=False):
            if token.startswith(" ") or not words:
                words.append([token.strip(), float(t)])
            else:
                words[-1][0] += token
        return [(w, t) for w, t in words if any(c.isalpha() for c in w)]


class Cache:
    """Texts and word times by the sha1 of their audio, kept in ``directory`` between runs."""

    def __init__(self, directory: Path) -> None:
        """Read what is there."""
        self.directory = directory
        self.texts: dict[str, str] = self._load("texts.json")
        self.words: dict[str, list[Word]] = {
            k: [(w, float(t)) for w, t in v] for k, v in self._load("words.json").items()
        }

    def _load(self, name: str) -> dict[str, Any]:
        path = self.directory / name
        return json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        """Write both."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False))
        (self.directory / "words.json").write_text(json.dumps(self.words, ensure_ascii=False))


def method_names() -> dict[str, Method]:
    """The methods by their command-line names (the label with dashes)."""
    return {m.label.replace(" ", "-"): m for m in METHODS}


def clips_with_pauses(
    clips: Sequence[Clip], pauses: Sequence[float], words_of: Any, config: CascadeConfig
) -> dict[str, list[Clip]]:
    """The sets: as spoken, and with each pause inserted at up to two boundaries a clip."""
    sets: dict[str, list[Clip]] = {"as spoken": list(clips)}
    places = {c.name: boundaries(c, words_of(c)) for c in clips}
    for pause in pauses:
        sets[f"pause +{pause:.2f} s"] = [
            with_pause(c, at, pause, config.start_db) for c in clips for at in places[c.name]
        ]
    return sets


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench; print the tables."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("days", nargs="*", type=Path, help="data/voice/<day> or cascade_* dirs")
    ap.add_argument("--live-cuts", type=Path, help="a directory of wavs/ and out/manifest.json")
    ap.add_argument("--pauses", type=float, nargs="*", default=[0.25, 0.5, 0.8])
    ap.add_argument("--methods", nargs="*", help=", ".join(method_names()))
    ap.add_argument("--read-ms", type=float, default=60.0, help="the early read lands this late")
    ap.add_argument("--judge-ms", type=float, default=15.0, help="the text model's own time")
    ap.add_argument("--set", action="append", default=[], help="section.key=value for all")
    ap.add_argument("--cache", type=Path, help="default: scratch/turn_eval")
    ap.add_argument("--limit", type=int, help="at most this many clips (a quick look)")
    ap.add_argument("-v", "--verbose", action="store_true", help="every cut and wait")
    args = ap.parse_args(argv)

    data = read_json(ROOT / "config" / "voice_cascade.json")
    base = CascadeConfig.from_dict(apply_overrides(data, args.set))
    names = method_names()
    chosen = [names[n] for n in args.methods] if args.methods else list(METHODS)
    vad = SileroVad(model_path(base.barge_in_vad_model))  # the gate opens on speech, as live

    clips: list[Clip] = []
    for path in args.days:
        found = [path] if (path / "events.jsonl").exists() else sorted(path.glob("cascade_*"))
        for directory in found:
            if (directory / "in.wav").exists() and (directory / "summary.json").exists():
                clips += session_clips(directory, base, vad)  # summary.json: the run closed
    if args.live_cuts is not None:
        clips += wav_clips(args.live_cuts, base, vad)
    clips = clips[: args.limit] if args.limit else clips
    print(f"{len(clips)} turns: {sum(c.source == 'cascade' for c in clips)} from the cascade's"
          f" records, {sum(c.source == 'live' for c in clips)} live cuts", flush=True)  # fmt: skip

    print("loading GigaAM (token times)...", flush=True)
    giga = GigaAmTimes(base.stt["gigaam"].model, base.stt["gigaam"].threads)
    cache = Cache(args.cache or ROOT / "scratch" / "turn_eval")

    def words_of(clip: Clip) -> list[Word]:
        key = hashlib.sha1(clip.pcm).hexdigest()
        if key not in cache.words:
            cache.words[key] = giga.words(clip.pcm)
        return cache.words[key]

    sets = clips_with_pauses(clips, args.pauses, words_of, base)
    reader = TimedReader(giga.transcribe, args.read_ms)
    reader.cache = cache.texts
    smart: SmartTurn | None = None
    judges: dict[str, TextJudge] = {"rules": RulesJudge()}
    if any(m.text not in (None, "rules") for m in chosen):
        print("loading the Namo text model...", flush=True)
        cfg = base
        judges["namo-ru"] = NamoJudge(
            NamoClassifier("namo-ru"),
            complete_p=cfg.turn_complete_p,
            incomplete_p=cfg.turn_incomplete_p,
        )
    if any(uses_smart_turn(base.with_changes(**m.changes)) for m in chosen):
        print("loading Smart Turn...", flush=True)
        smart = SmartTurn()

    reads: list[Result] = []
    for set_name, set_clips in sets.items():
        print(f"\n{set_name}: {len(set_clips)} clips", flush=True)
        for method in chosen:
            config = base.with_changes(**method.changes)
            judge = judges.get(method.text or "")
            reader.read_ms = args.read_ms + (
                args.judge_ms if method.text not in (None, "rules") else 0
            )
            tally = Tally()
            for clip in set_clips:
                turn = TurnWorker(smart, inline=True) if smart and uses_smart_turn(config) else None
                result = replay(clip, config, reader=reader, turn=turn, judge=judge, vad=vad)
                tally.add(result, config.continuation_s)
                if method.label == "two-gates rules":
                    reads.append(result)
                if args.verbose:
                    print_clip(method, result, config)
            print(f"  {method.label:<26}{tally.row()}", flush=True)
        cache.save()

    if reads:
        rows = text_rows(reads, judges)
        print(f"\ntext judges on the early reads ({len(rows)}):")
        print("\n".join("  " + line for line in text_table(rows, list(judges))))
        if args.verbose:
            for row in rows:
                cells = " ".join(f"{n} {j.verdict}" for n, j in row.verdicts.items())
                print(f"    {row.where:<6}{cells:<44}{row.text!r}")
    agreement(sets["as spoken"], base, reader, vad)
    if giga.ms:
        print(f"GigaAM: {len(giga.ms)} calls, median {statistics.median(giga.ms):.0f} ms")
    cache.save()
    return 0


def print_clip(method: Method, result: Result, config: CascadeConfig) -> None:
    """A clip's cuts and its close, when something happened worth reading."""
    cuts = result.cuts()
    final = result.final()
    if not cuts and (final is None or final.closed_by in (None, "silence")):
        return
    clip = result.clip
    marks = []
    for c in cuts:
        merged = "merged" if result.merged(c, config.continuation_s) else "split"
        marks.append(f"cut {c.ended_s:.2f}s by {c.closed_by} ({c.verdict}: {c.why}) {merged}")
    if final is not None:
        marks.append(f"close +{(final.t - clip.end_s) * 1000:.0f} ms by {final.closed_by}"
                     f" ({final.verdict}: {final.why})")  # fmt: skip
    print(f"    [{method.label}] {clip.name} {clip.text!r}: " + "; ".join(marks))


def agreement(
    clips: Sequence[Clip], base: CascadeConfig, reader: TimedReader, vad: SileroVad
) -> None:
    """The replay against the record: silence 0.4 replayed should close where the run did."""
    config = base.with_changes(silence_s=0.4)
    off = []
    cut = 0
    for clip in clips:
        if clip.source != "cascade":
            continue
        result = replay(clip, config, reader=reader, vad=vad)
        final = result.final()
        cut += bool(result.cuts())
        if final is not None:
            off.append(abs(final.ended_s - clip.end_s) * 1000)
    if off:
        within = sum(1 for x in off if x <= 60)
        print(f"\nthe replay against the record (silence 0.4, {len(off)} turns): the last word"
              f" within 60 ms of the run's in {within}, median off {statistics.median(off):.0f}"
              f" ms; {cut} turns cut (the run's own joins)")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
