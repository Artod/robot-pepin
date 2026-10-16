"""Talk to Pepin through its own voice loop, the cascade: local ears -> Claude -> local voice.

The board's array -> a loudness-cut utterance -> Whisper on the Mac (and Praat's tag line) ->
"Пепин" opens a conversation -> Claude streams the reply, sentence by sentence, calling the
robot's tools -> Qwen3-TTS on the Mac speaks each sentence as it arrives -> the board's speaker.
Every turn prints its timeline (person stopped -> cut -> text -> token -> sentence -> tts ->
play -> end, in ms) with the cache tokens and the price. Every switch: config/voice_cascade.json,
or --set section.key=value for one run. Logs: data/voice/<day>/cascade_<HHMMSS>/; outside a
conversation each utterance is a row of data/voice/<day>/cascade_heard.jsonl and, unless it is a
noise line, its audio in data/voice/<day>/heard/<HHMMSS_mmm>.wav (ears.heard_audio).

The key comes from the environment (the repo's gitignored .env, loaded by the shell):

    set -a; . ./.env; set +a
    uv run python scripts/voice_cascade.py                          # the robot, for real
    uv run python scripts/voice_cascade.py --brain sonnet --set flags.barge_in=true
    uv run python scripts/voice_cascade.py --brain flashlite     # Gemini (GEMINI_API_KEY)
    uv run python scripts/voice_cascade.py --brain groq-qwen     # Groq (GROQ_API_KEY)
    uv run python scripts/voice_cascade.py --set ears.turn_model=smart-turn-v3.2   # Smart Turn
    uv run python scripts/voice_cascade.py --set ears.turn_model=two-gates  # words + pause
    uv run python scripts/voice_cascade.py --set transcriber.engine=whispercpp  # Neural Engine
    uv run python scripts/voice_cascade.py --voice gemini-lite   # Gemini TTS (GEMINI_API_KEY)

Switch the brain, the voice or the speech-to-text engine of the running loop (from another
terminal; it writes data/voice/control.json and prints the loop's answers: a brain is ready in
about a second, a voice once its model has loaded, a transcriber between utterances):

    uv run python scripts/voice_cascade.py switch --brain sonnet --voice qwen3-1.7b
    uv run python scripts/voice_cascade.py switch --transcriber whispercpp
    uv run python scripts/voice_cascade.py switch --flag gestures=on   # config/expression.json
    uv run python scripts/voice_cascade.py switch --flag doze=off      # the face awake when idle
    uv run python scripts/voice_cascade.py switch --flag people=on     # config/people.json
    uv run python scripts/voice_cascade.py switch --knob vision.source=snapshot  # models.json

The pictures (the speaker gaze's faces, person recognition, ``see``) come from the perception
service on this Mac (``ros/models.sh start perception``, fed by camera_stream); the voice process
holds none. ``vision.source=snapshot`` goes back to ustreamer's snapshots decoded here, for a
day the service is broken. Each turn's line ends with the vision cell (frames in and detected a
second, the detector's ms, the faces and people in the newest record).

The other speech-to-text engines (pepin.voice_cascade.stt: gigaam, parakeet, canary, voxtral)
need their runtime once, ``uv sync --group stt-<engine>`` (several --group flags add several),
and then switch the same way (``--transcriber gigaam``, or engine:model); the loop prints the
load time and the process's footprint before, with both engines and after the old one closed.

A test with no person and no sound in the flat (phrases spoken by macOS say, the robot's speech
written to a WAV, the tools on fakes; a drive takes 5 s):

    uv run python scripts/voice_cascade.py --say "Пепин, где ты?" --say "Езжай к принтеру" \\
        --speaker-wav /tmp/pepin.wav --fake-robot 5 --turns 3

No API call at all (the brain repeats what it heard): --brain echo.

The latency bench (recorded utterances through the real Whisper, brain and voice; p50/p90 per
stage at the end, the turns in data/voice/<day>/cascade_bench_<HHMMSS>.json):

    uv run python scripts/voice_cascade.py --bench scratch/prosody_1008/wavs --bench-n 15
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from pepin.audio_doa import MicDoa, SrpPhat
from pepin.audio_link import board_host
from pepin.deployment import config_file
from pepin.expression.finder import SwitchedFinder, robot_finder
from pepin.expression.places import GoalPlaces
from pepin.expression.tracker import SpeakerGazeWorker, SpeakerSettings, robot_gaze
from pepin.expression.wheels import WheelsExecutor, WheelsGuard, robot_wheels
from pepin.overlay import read_json
from pepin.people.worker import robot_people
from pepin.perception.client import HealthWatch, PerceptionClient
from pepin.perception.config import PerceptionConfig
from pepin.perception.source import KNOB as VISION_KNOB
from pepin.perception.source import SOURCES as VISION_SOURCES
from pepin.perception.source import VisionSource
from pepin.tools import TOOLS, Robot
from pepin.tools.clients import ChannelSightings
from pepin.tools.fakes import FakeCamera, FakeGoalServer, WallClock, fake_robot
from pepin.voice_cascade.barge_in import speech_series
from pepin.voice_cascade.bench import ReplayMic, utterances
from pepin.voice_cascade.config import (
    CONFIG_FILE,
    TRANSCRIBER_ENGINES,
    TWO_GATES,
    CascadeConfig,
    voice_preset,
)
from pepin.voice_cascade.control import (
    DONE_STATES,
    FLAG,
    KNOB,
    ControlFile,
    Switchboard,
    answers_to,
    flag_value,
    write_request,
)
from pepin.voice_cascade.engines import (
    ECHO,
    NOT_FOR_THE_BRAIN,
    BrainSetup,
    VoiceShop,
    brain_key,
    build_brain,
)
from pepin.voice_cascade.fillers import clip_lines
from pepin.voice_cascade.loop import UNANSWERED, CascadeLoop, Mic
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechWorker
from pepin.voice_cascade.stages import Tagger, Transcriber
from pepin.voice_cascade.tagger import ProsodyTagger, emotion_model
from pepin.voice_cascade.timeline import report
from pepin.voice_cascade.transcriber import (
    Choice,
    SwitchingTranscriber,
    build_transcriber,
    choice_of,
    configured,
    wanted_from,
)
from pepin.voice_cascade.turn import SmartTurn, TurnWorker
from pepin.voice_cascade.turn_text import TextJudge, build_judge
from pepin.voice_live.budget import Ledger
from pepin.voice_live.config import LiveConfig, Prices
from pepin.voice_live.events import Events, StatePrinter
from pepin.voice_live.sources import BoardLink, ScriptedMic, WavSpeaker, synthesize

REPO = Path(__file__).resolve().parents[1]
FAKE_ACCEPT_S = 0.3  # a fake drive is taken this long after the go
SWITCH_WAIT_S = 120.0  # a voice's model may take a minute to load the first time


def main_checkout() -> Path:
    """The main checkout (from a worktree, <main>/.claude/worktrees/<name>): where scratch/
    and the model caches live."""
    return REPO.parents[2] if REPO.parent.parent.name == ".claude" else REPO


def fake_view_jpeg() -> bytes:
    """The fake camera's picture: a red circle on white, a valid JPEG a model can describe
    (it refuses a broken one; a description naming the circle proves the picture arrived)."""
    from PIL import Image, ImageDraw

    out = io.BytesIO()
    picture = Image.new("RGB", (320, 240), (255, 255, 255))
    ImageDraw.Draw(picture).ellipse((110, 70, 210, 170), fill=(220, 20, 20))
    picture.save(out, format="JPEG")
    return out.getvalue()


def ledger_for(config: CascadeConfig, path: Path) -> Ledger:
    """The voice_live ledger, holding the cascade's own caps."""
    caps = LiveConfig(
        max_sessions_per_hour=config.max_sessions_per_hour,
        daily_budget_cad=config.daily_budget_cad,
        prices=Prices(usd_to_cad=config.usd_to_cad),
    )
    return Ledger(path, caps)


def parse_flags(items: list[str], known: set[str]) -> dict[str, bool]:
    """``["gestures=on", "speaker_gaze=off"]`` as ``{name: on}``; ``ValueError`` naming an item
    that is not ``NAME=on|off`` or a name config/expression.json's flags do not carry."""
    flags: dict[str, bool] = {}
    for item in items:
        name, _, value = item.partition("=")
        wanted = flag_value(value)
        if not name or wanted is None:
            raise ValueError(f"--flag {item!r}: NAME=on or NAME=off")
        if name not in known:
            raise ValueError(f"--flag {name!r}: one of {', '.join(sorted(known))}")
        flags[name] = wanted
    return flags


KNOBS: dict[str, tuple[str, ...]] = {VISION_KNOB: VISION_SOURCES}  # live knobs, their values


def parse_knobs(items: list[str], known: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """``["vision.source=snapshot"]`` as ``{name: value}``; ``ValueError`` naming an item that
    is not ``NAME=VALUE``, a knob the loop does not carry or a value it does not take."""
    knobs: dict[str, str] = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            raise ValueError(f"--knob {item!r}: NAME=VALUE")
        if name not in known:
            raise ValueError(f"--knob {name!r}: one of {', '.join(sorted(known))}")
        if value not in known[name]:
            raise ValueError(f"--knob {name}: {value!r} is not one of {', '.join(known[name])}")
        knobs[name] = value
    return knobs


def expression_flags() -> set[str]:
    """The live flags: config/expression.json's ``flags`` (but the note) and the people flag
    (config/people.json, ``pepin.people.config.FLAG``)."""
    from pepin.people.config import FLAG as PEOPLE_FLAG

    data = read_json(config_file("expression.json"))
    return {name for name in data.get("flags", {}) if name != "note"} | {PEOPLE_FLAG}


class Body:
    """The gesture engine of this run (config/expression.json's ``gestures`` switch at the start,
    live through ``switch --flag``), its parts, the marker vocabulary, and what to close."""

    def __init__(self, engine: Any, vocabulary: Any, closers: list[Any]) -> None:
        """Built by :meth:`build`."""
        self.engine, self.vocabulary, self.closers = engine, vocabulary, closers

    @classmethod
    def build(
        cls,
        host: str,
        player: Player,
        *,
        fake: bool,
        face: bool,
        logs: Path,
        wheels: WheelsExecutor | None = None,
    ) -> Body:
        """The engine over the real arm, head, face, eyes and ``wheels`` (``fake``: a fake arm
        that answers at rest and a head door that only records; ``face`` False: a face and eyes
        that only record; no ``wheels``: the body never moves), every performance, cue and
        answer logged to ``logs/<day>/cascade_gestures.jsonl`` and the notable ones printed."""
        from pepin.arm_motion import load_arm_motion
        from pepin.expression.animation import PartName
        from pepin.expression.animator import GestureEngine, engine_config
        from pepin.expression.arm_link import ArmClient
        from pepin.expression.book import (
            face_names,
            load_book,
            style_names,
            vocabulary,
            with_rest,
            with_styles,
        )
        from pepin.expression.door import OverlayDoor
        from pepin.expression.executors import Executor, NoBody
        from pepin.expression.fakes import FakeArmLink, FakeEyesChannel, FakeFace, FakeHeadDoor
        from pepin.expression.head import ClipLook
        from pepin.expression.parts import (
            ArmPart,
            EyesPart,
            FacePart,
            HeadDoor,
            HeadPart,
            load_neck_limits,
        )
        from pepin.expression.trajectory import ArmLink, ArmStatus
        from pepin.voice_cascade.log import append_line, day_dir

        def record(kind: str, fields: dict[str, Any]) -> None:
            wall = time.time()
            row = {"t": round(wall, 3), "kind": kind, **fields}
            append_line(day_dir(logs, wall) / "cascade_gestures.jsonl", row)
            if kind != "cue" or not fields.get("ok"):
                print(f"  (gesture {kind}: {json.dumps(fields, ensure_ascii=False)})", flush=True)

        def answered(answer: Any, t: float) -> None:
            record("arm", {"id": answer.id, "status": answer.status, "reason": answer.reason})

        def head_answered(request: ClipLook, refusal: str | None) -> None:
            target = request["target"]
            record("head", {"source": request["source"], "keys": len(target["keys"]),
                            "loop": target.get("loop", False),
                            "own_wheels": target.get("own_wheels", False), "ok": refusal is None,
                            "why": refusal or ""})  # fmt: skip

        motion = load_arm_motion()
        knobs = read_json(config_file("expression.json"))

        def load() -> Any:
            data = read_json(config_file("expression.json"))
            book = with_styles(with_rest(load_book(), motion.rest_deg), style_names())
            return book, engine_config(data)

        closers: list[Any] = []
        link: ArmLink
        if fake:
            link = FakeArmLink(state=ArmStatus(0.0, motion.rest(), False, "off"))
        else:
            client = ArmClient(host, on_answer=answered).start()
            link = client
            closers.append(client)
        arm_knobs = knobs.get("parts", {}).get("arm", {})
        head_knobs = knobs.get("parts", {}).get("head", {})
        face_knobs = knobs.get("parts", {}).get("face", {})
        eyes_knobs = knobs.get("parts", {}).get("eyes", face_knobs)
        door: HeadDoor = FakeHeadDoor() if fake else OverlayDoor()
        fades = knobs.get("layers", {}).get("fade_s", {})
        head = HeadPart(
            door,
            latency_s=float(head_knobs.get("latency_s", 0.1)),
            limits=load_neck_limits(),
            fade_s=float(fades.get("head", 0.35) if isinstance(fades, dict) else fades),
            on_answer=head_answered,
            own_wheels=wheels.gesture_moving if wheels is not None else None,
        )
        channel: Any = FakeFace()
        eyes: Any = FakeEyesChannel()
        if face:  # the face and the eyes: one connection, the expression source's
            from pepin.head_link import HeadClient

            channel = eyes = HeadClient(host, source="expression").start()
            closers.append(channel)
        arm = ArmPart(
            link,
            motion,
            latency_s=float(arm_knobs.get("latency_s", 0.06)),
            speed_scale=float(arm_knobs.get("speed_scale", 0.2)),
            knot_hz=float(arm_knobs.get("knot_hz", 25.0)),
        )
        parts: dict[PartName, Executor] = {
            "arm": arm,
            "face": FacePart(channel, latency_s=float(face_knobs.get("latency_s", 0.05))),
            "eyes": EyesPart(eyes, latency_s=float(eyes_knobs.get("latency_s", 0.05))),
            "head": head,
            "body": wheels if wheels is not None else NoBody(),
        }
        on = bool(knobs.get("flags", {}).get("gestures", False))
        engine = GestureEngine(parts, player, load, on=on, record=record).start()
        return cls(engine, vocabulary(engine.book, face_names()), closers)

    def close(self) -> None:
        """The engine's thread stopped, the links let go (the board rests the arm itself)."""
        self.engine.close()
        for closer in self.closers:
            closer.close()


def switch(argv: list[str]) -> int:
    """``switch --brain X --voice Y --transcriber Z --flag NAME=on``: ask the running loop for
    a switch and print its answers until each part is applied or has failed."""
    ap = argparse.ArgumentParser(
        prog="voice_cascade.py switch",
        description="switch the running loop's brain, voice, speech-to-text engine or features",
    )
    ap.add_argument("--brain", help="a brain preset, or echo")
    ap.add_argument("--voice", help="a voice preset (or an engine's name: qwen3, gemini, say)")
    ap.add_argument(
        "--transcriber",
        help=f"{', '.join(TRANSCRIBER_ENGINES)}, or engine:model (whispercpp:large-v3)",
    )
    ap.add_argument(
        "--flag", action="append", default=[], metavar="NAME=on|off",
        help="a feature's switch, live: gestures, speaker_gaze (config/expression.json),"
        " people (config/people.json)",
    )  # fmt: skip
    ap.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE",
        help="a live knob: vision.source=service|snapshot (the pictures: the perception"
        " service, or ustreamer's snapshots decoded here)",
    )  # fmt: skip
    ap.add_argument("--config", type=Path, help="default: config/voice_cascade.json")
    ap.add_argument("--logs", type=Path, default=REPO / "data" / "voice")
    ap.add_argument("--wait-s", type=float, default=SWITCH_WAIT_S)
    args = ap.parse_args(argv)
    if not (args.brain or args.voice or args.transcriber or args.flag or args.knob):
        ap.error("--brain, --voice, --transcriber, --flag, --knob, or several")
    try:
        flags = parse_flags(args.flag, expression_flags())
        knobs = parse_knobs(args.knob, KNOBS)
    except ValueError as error:
        print(error)
        return 2
    data = read_json(args.config or config_file(CONFIG_FILE))
    brains = [*data.get("brain", {}).get("presets", {}), ECHO]
    if args.brain and args.brain not in brains:
        print(f"brain {args.brain!r}: one of {', '.join(brains)}")
        return 2
    if args.voice:
        try:
            voice_preset(data.get("voice", {}), args.voice)
        except ValueError as error:
            print(error)
            return 2
    if args.transcriber and choice_of(args.transcriber, CascadeConfig.load(args.config)) is None:
        print(f"transcriber {args.transcriber!r}: {', '.join(TRANSCRIBER_ENGINES)} or engine:model")
        return 2
    request = write_request(args.logs, args.brain, args.voice, args.transcriber, flags, knobs)
    print(
        f"asked ({request.id}): brain {args.brain}, voice {args.voice},"
        f" transcriber {args.transcriber}, flags {flags or None}, knobs {knobs or None}",
        flush=True,
    )
    asked = (("brain", args.brain), ("voice", args.voice), ("transcriber", args.transcriber))
    parts = (
        {what for what, name in asked if name}
        | {f"{FLAG} {name}" for name in flags}
        | {f"{KNOB} {name}" for name in knobs}
    )
    deadline = time.monotonic() + args.wait_s
    shown = 0
    failed = False
    while time.monotonic() < deadline:
        rows = answers_to(args.logs, request.id)
        for row in rows[shown:]:
            extra = {k: v for k, v in row.items() if k not in ("id", "t", "what", "name", "state")}
            print(f"  {row['what']} {row['name']}: {row['state']} {json.dumps(extra)}", flush=True)
        shown = len(rows)
        done = {
            f"{row['what']} {row['name']}" if row["what"] in (FLAG, KNOB) else row["what"]
            for row in rows
            if row["state"] in DONE_STATES
        }
        failed = any(row["state"] == "failed" for row in rows)
        if parts <= done:
            return 1 if failed else 0
        time.sleep(0.2)
    if not shown:
        print(f"no answer in {args.wait_s:.0f} s: is scripts/voice_cascade.py running"
              f" (logs {args.logs})?")  # fmt: skip
    else:
        print(f"not finished in {args.wait_s:.0f} s")
    return 3


def main() -> int:
    if sys.argv[1:2] == ["switch"]:
        return switch(sys.argv[2:])
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, help="default: config/voice_cascade.json")
    ap.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one config value for this run (JSON value), e.g. flags.barge_in=true",
    )  # fmt: skip
    ap.add_argument(
        "--brain",
        help="a brain preset (haiku, sonnet, opus, flashlite, flash, flash38, groq-oss,"
        " groq-qwen), or echo: no API, repeats the words",
    )
    ap.add_argument(
        "--voice",
        help="a voice preset (qwen3-0.6b, qwen3-1.7b, qwen3-vd, gemini-lite, gemini-flash, say)"
        " or an engine's name (qwen3, gemini, say: its first preset)",
    )
    ap.add_argument("--host", default=board_host(), help="the board (PEPIN_HOST)")
    ap.add_argument("--say", action="append", help="speak this into the loop instead of the mic")
    ap.add_argument("--speaker-wav", type=Path, help="write the robot's speech here, no sound")
    ap.add_argument("--tools", help="comma-separated: only these tools (default: all but say)")
    ap.add_argument(
        "--fake-robot", type=float, metavar="DRIVE_S",
        help="no robot: the tools act on fakes (nothing moves), a drive takes DRIVE_S seconds",
    )  # fmt: skip
    ap.add_argument("--turns", type=int, help="exit after this many of the person's turns")
    ap.add_argument(
        "--bench", type=Path, metavar="DIR",
        help="replay the utterances of DIR (WAVs, or a live_* session) and print p50/p90",
    )  # fmt: skip
    ap.add_argument("--bench-n", type=int, default=15, help="how many utterances to replay")
    ap.add_argument("--gap-s", type=float, default=1.0, help="quiet before each replayed phrase")
    ap.add_argument("--no-face", action="store_true", help="leave the head's face alone")
    ap.add_argument("--logs", type=Path, default=REPO / "data" / "voice")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(name)s: %(message)s")

    base = list(args.set)
    if args.bench is not None:
        base.append("ears.wake=always")
    echo = args.brain == ECHO

    def configure(brain: str | None, voice: str | None) -> CascadeConfig:
        """The run's config with these presets (None or the echo: the file's own)."""
        extra = [f"brain.preset={brain}"] if brain and brain != ECHO else []
        extra += [f"voice.preset={voice}"] if voice else []
        return CascadeConfig.load(args.config, [*base, *extra])

    config = configure(args.brain, args.voice)

    wanted: set[str] | None = None
    if args.tools:
        names = [t.name for t in TOOLS if t.name not in NOT_FOR_THE_BRAIN]
        wanted = {n.strip() for n in args.tools.split(",") if n.strip()}
        unknown = wanted - set(names)
        if unknown:
            print(f"unknown tools: {', '.join(sorted(unknown))}; the tools are {', '.join(names)}")
            return 2

    def make_brain(config: CascadeConfig, name: str) -> BrainSetup:
        return build_brain(config, echo=name == ECHO, wanted=wanted)

    ledger = ledger_for(config, args.logs / "cascade_ledger.jsonl")
    if not echo:
        key = brain_key(config)
        if not os.environ.get(key):
            print(f"{key} missing: set -a; . ./.env; set +a (the main checkout's .env)")
            return 2
        refusal = ledger.refusal()
        if refusal:
            print(f"refusing to start: {refusal}")
            return 3
    setup = make_brain(config, ECHO if echo else config.brain.preset or config.brain.model)
    brain, allowed, moving = setup.brain, setup.allowed, setup.moving

    bench = args.bench is not None
    fake_drive_s = args.fake_robot if args.fake_robot is not None else 5.0 if bench else None
    # The one camera channel: the perception service's client and the live vision.source,
    # shared by the gaze's finder, person recognition and see.
    try:
        channel = PerceptionConfig.load()
    except (OSError, ValueError) as error:
        print(f"config/models.json perception: {error}; its defaults")
        channel = PerceptionConfig()
    vision = VisionSource(channel.source)
    perception = PerceptionClient(channel.url)
    if fake_drive_s is not None:
        clock = WallClock()
        step_s = max(0.0, fake_drive_s - FAKE_ACCEPT_S) / 3  # accepted, 2 reports, done
        goals = FakeGoalServer(clock, step_s=step_s, accept_s=FAKE_ACCEPT_S)
        robot = fake_robot(clock=clock, goals=goals, camera=FakeCamera(jpeg=fake_view_jpeg()))
        print(f"FAKE ROBOT: nothing moves; a drive takes {fake_drive_s:.1f} s", flush=True)
    else:
        robot = Robot.connect(vision=vision, perception=perception)

    def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return TOOLS.call(name, arguments, robot)

    events = Events(StatePrinter())
    quiet_run = args.speaker_wav is not None or bench
    face: Any = None
    doze: Any = None  # the face asleep outside a conversation (flag doze)
    dozing_head: Any = None
    if not quiet_run and not args.no_face:  # the robot speaks: its face follows
        from pepin.expression.doze import DOZE, Doze, doze_config
        from pepin.face_events import VoiceStateFace
        from pepin.head_link import HeadClient

        face = VoiceStateFace(HeadClient(args.host, source="voice").start())
        events.subscribe(face)
        dozing_head = HeadClient(args.host, source=DOZE).start()
        doze = Doze(dozing_head, lambda: doze_config(read_json(config_file("expression.json"))))

    stamp = time.strftime("%H%M%S")
    speaker: WavSpeaker | BoardLink
    link: BoardLink | None = None
    if bench:
        day = args.logs / time.strftime("%Y%m%d")
        speaker = WavSpeaker(args.speaker_wav or day / f"cascade_bench_{stamp}.wav")
    elif args.speaker_wav is not None:
        speaker = WavSpeaker(args.speaker_wav)
    else:
        link = BoardLink(args.host)
        speaker = link
    player = Player(
        speaker,
        rate=speaker.play_rate,
        lead_s=config.playback_lead_s,
        block_s=config.playback_block_s,
        on_level=lambda value: events.emit("speaking", value),
    )

    shop = VoiceShop(config, main_checkout())
    try:
        voice = shop.voice(config.voice)
    except RuntimeError as error:  # a cloud voice without its key
        print(error)
        return 2
    speech = SpeechWorker(voice, player, clips=config.openers, asides=clip_lines(config))
    choice = configured(config)
    print(f"loading the voice ({config.voice.name}) and the transcriber ({choice.label})...")
    speech.start()  # the voice loads on its own thread meanwhile
    server_log = args.logs / "whisper_server.log"

    def load_transcriber(chosen: Choice) -> Transcriber:
        return build_transcriber(config, chosen, main_checkout(), log=server_log)

    control = ControlFile(args.logs)  # the one reader of control.json; the loop polls it
    transcriber = SwitchingTranscriber(
        load_transcriber(choice),
        choice,
        load_transcriber,
        wanted_from(control, config),
        report=lambda line: print(line, flush=True),
    )
    turn: TurnWorker | None = None
    judge: TextJudge | None = None
    sound = config.turn_model
    if config.turn_model == TWO_GATES:
        sound = None if config.turn_acoustic == "pause" else config.turn_acoustic
        words = config.turn_text_model or "rules"
        print(f"end of turn: two gates, the words by {words}, the sound by"
              f" {config.turn_acoustic}", flush=True)  # fmt: skip
        judge = build_judge(
            config.turn_text_model,
            complete_p=config.turn_complete_p,
            incomplete_p=config.turn_incomplete_p,
        )
    if sound is not None:
        print(f"loading the end-of-turn model ({sound})...", flush=True)
        turn = TurnWorker(SmartTurn(sound))
    tagger: Tagger | None = None
    if config.prosody_tags:
        tagger = ProsodyTagger(config.prosody, emotion=emotion_model(config.prosody.emotion_model))

    holder: list[CascadeLoop] = []

    def ready() -> bool:
        return bool(holder) and holder[0].quiet()

    mic: Mic
    replay: ReplayMic | None = None
    turns = args.turns
    if bench:
        assert args.bench is not None
        bench_path = args.bench if args.bench.is_absolute() else main_checkout() / args.bench
        if not bench_path.exists():
            bench_path = args.bench
        found = utterances(bench_path, args.bench_n)
        print(f"bench: {len(found)} utterances from {bench_path}", flush=True)
        replay = ReplayMic([pcm for _, pcm in found], ready, gap_s=args.gap_s)
        mic = replay
        turns = turns or len(found)
    elif args.say:
        mic = ScriptedMic([synthesize(text) for text in args.say], ready, gap_s=args.gap_s)
        turns = turns or len(args.say)
    else:
        mic = link or BoardLink(args.host)
    gaze: SpeakerGazeWorker | None = None
    finder: SwitchedFinder | None = None
    watch: HealthWatch | None = None
    on_robot = isinstance(mic, BoardLink) and fake_drive_s is None
    if on_robot:  # one face finder for the speaker gaze and person recognition
        # The perception service's records, or (vision.source snapshot) the snapshot finder,
        # built and loaded only when that switch is turned.
        finder = robot_finder(args.host, vision, perception, wait_s=channel.look_wait_s)
        watch = HealthWatch(perception).start()  # the vision cell, read from memory
    people = robot_people(  # voices only without the robot's camera
        finder, service=perception if on_robot else None, source=vision
    )
    robot.people = people  # the remember_person / forget_person tools' door
    # The wheels' one owner for the gaze's turn and the gestures' body (no I/O until a motion;
    # each moves only behind its own flag); the guard stops a gesture's when the person speaks.
    wheels: WheelsExecutor | None = None
    if on_robot:
        try:
            wheels = robot_wheels(args.host)
        except (OSError, ValueError, TypeError) as exc:  # the voice starts whatever the knobs say
            print(f"no wheels for expression (config/expression.json wheels: {exc})")
    guards = [WheelsGuard(wheels)] if wheels is not None else []
    mac_doa: MicDoa | None = None
    if on_robot:  # the robot, its array, its head
        assert isinstance(mic, BoardLink)
        # The Mac's own voice direction from the array's raw microphones (speaker.doa.direction
        # mac): fed by the link's raw frames, its Silero probabilities the ears' once they exist.
        try:
            knobs = SpeakerSettings.load()
            mac_doa = MicDoa(
                SrpPhat.load(),
                voice_p_min=knobs.mac_voice_p_min,
                peak_ratio=knobs.mac_peak_ratio,
                reading_s=knobs.mac_reading_s,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"no Mac voice direction (config/audio.json, expression.json: {exc}): the chip's")
        else:
            mic.listen_raw(mac_doa.feed)
            print(
                f"Mac voice direction ready: the board streams {mic.raw_channels} raw mics (0: the"
                " 2-channel firmware or raw_mics off, the chip's direction then)",
                flush=True,
            )
        gaze = robot_gaze(args.host, mic, finder=finder, on_face=people.seen_face, wheels=wheels,
                          mac=mac_doa)  # fmt: skip
        # the approach tool's speaker: the person this gaze holds, on the channel's own frames
        robot.sightings = ChannelSightings(perception, vision, gaze, wait_s=channel.look_wait_s)
        robot.gaze = gaze  # and its body turn to the voice, when nobody is in view
        people.on_named = gaze.named  # a sure verdict names the person the gaze holds

    body = Body.build(
        args.host,
        player,
        fake=fake_drive_s is not None,
        face=not quiet_run and not args.no_face,
        logs=args.logs,
        wheels=wheels,
    )
    robot.animator = body.engine  # the gesture tool's door
    if gaze is not None:  # attention first: the engine reads it, a body turn asks for the arm
        body.engine.attention = gaze  # the gesture space's person too
        gaze.arm = body.engine
    # The gesture space's room: the goal server's places book and the robot's map pose, read on
    # a thread of their own (a down server: a place's name gets its floor until it answers).
    room = GoalPlaces(robot.goals).start()
    body.engine.resolver.room = room
    # The alive layer (riding gestures): the voice's own nuclei and melody from every reply chunk,
    # the listeners' requests into the engine, the quiet rule over the engine's renderings, the
    # arm kept up after a mark while a long reply's posture is due or up.
    from pepin.expression.listeners import Alive
    from pepin.expression.prosody import ProsodyListener

    prosody = ProsodyListener()
    alive = Alive(body.engine, prosody, say=lambda line: print(line, flush=True))
    body.engine.taps.append(prosody)
    body.engine.scene_hooks.append(alive.scene)
    body.engine.scheduler.stay_up = alive.stays_up
    # On the word: the ladder plans with the arm's planner and units, renders the book's other
    # versions and floors through the engine's renderer and scene hooks, and takes the voice's
    # nuclei as the word clock's AUDIO moment; a barge-in's record reads the same clock.
    body.engine.scheduler.attach(
        body.engine.parts, lambda: body.engine.renderer, nuclei=prosody,
        hooks=body.engine.scene_hooks,
    )  # fmt: skip
    player.words = body.engine.scheduler.words
    switchboard = Switchboard(
        control,
        configure=lambda brain, voice: configure(brain, voice),
        build_brain=make_brain,
        build_voice=shop.voice,
        brain=setup.name,
        voice=config.voice.name,
        keep_voice=shop.keep,
        transcriber=transcriber,
    )
    loop = CascadeLoop(
        config,
        mic=mic,
        player=player,
        transcriber=transcriber,
        tagger=tagger,
        brain=brain,
        speech=speech,
        tools=run_tool,
        allowed=allowed,
        moving=moving,
        halt=robot.halt,
        events=events,
        logs=args.logs,
        ledger=ledger if setup.paid else None,
        turn=turn,
        switchboard=switchboard,
        observers=[*guards, body.engine, alive, people]
        + ([gaze] if gaze is not None else [])
        + ([doze] if doze is not None else []),
        flags=[body.engine, people]
        + ([gaze] if gaze is not None else [])
        + ([doze] if doze is not None else []),
        taps=[body.engine],
        vocabulary=body.vocabulary,
        people=people,
        judge=judge,
        knobs=[vision],
        cells=[watch.cell] if watch is not None else [],
    )
    loop.paid_ledger = ledger
    holder.append(loop)
    if gaze is not None:
        gaze.journal = loop.record  # the gaze's decisions in each conversation's events.jsonl
        gaze.series = speech_series(loop.ears.detector)  # the ears' Silero tags the array's DOA
        if mac_doa is not None:
            mac_doa.series = gaze.series  # ...and picks the raw chunks the Mac's direction reads
    alive.journal = loop.record  # pulse, moment, quiet and alive events, the same way
    body.engine.journal = loop.record  # the gesture space's catchment and repeat rows
    alive.start()
    if doze is not None:
        doze.start()
    b = config.brain
    print(
        f"listening: {setup.name} = {b.model} (thinking {b.thinking}, effort {b.effort}, cache"
        f" {b.cache}), voice {config.voice.name} ({config.voice.model}), wake {config.wake};"
        f" barge_in {config.barge_in}, backchannel {config.backchannel}, prosody_tags"
        f" {config.prosody_tags}, turn_model {config.turn_model}, speculative {b.speculative},"
        f" first sentence {b.first_sentence_words or 'as the persona says'}, gestures"
        f" {'on' if body.engine.on else 'off'}, people {'on' if people.on else 'off'},"
        f" doze {'on' if doze is not None and doze.on else 'off'},"
        f" vision.source {vision.value}"
        f"{f' ({watch.cell()})' if watch is not None else ''}. Logs:"
        f" {args.logs}; switch: scripts/voice_cascade.py switch --brain NAME --voice NAME"
        " --flag gestures=on --knob vision.source=snapshot",
        flush=True,
    )
    try:
        asyncio.run(loop.run(turns=turns))
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
        if moving and fake_drive_s is None:
            print("halt:", robot.halt(), flush=True)
    finally:
        alive.close()
        room.close()
        if gaze is not None:
            gaze.close()  # the head let go before the links close
        if wheels is not None:
            wheels.close()
        people.close()
        if watch is not None:
            watch.close()
        speech.close()
        if turn is not None:
            turn.close()
        transcriber.close()  # whisper.cpp's server, when that engine is in use
        if face is not None:
            face.close()
        if doze is not None:
            doze.close()  # the face let go: nobody listens any more
            dozing_head.close()
        body.close()
        for closable in (mic, speaker):
            close = getattr(closable, "close", None)
            if close is not None:
                close()
    person = [t for t in loop.turns if t["kind"] == "person" and t["outcome"] not in UNANSWERED]
    if person:
        print("\n" + "\n".join(report(person)), flush=True)
        ears = loop.ears
        print(
            f"Whisper started in a pause {ears.early_asked} times, {ears.early_dropped} of them"
            " for a pause the person talked through",
            flush=True,
        )
    if bench and person:
        out = args.logs / time.strftime("%Y%m%d") / f"cascade_bench_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": b.model,
            "turn_model": config.turn_model,
            "thinking": b.thinking,
            "effort": b.effort,
            "voice": config.voice.name,
            "voice_model": config.voice.model,
            "source": str(args.bench),
            "report": report(person),
            "whisper_early": {"asked": loop.ears.early_asked, "dropped": loop.ears.early_dropped},
            "turns": loop.turns,
        }
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
        print(f"bench: {out}", flush=True)
    if loop.ledger is not None or not echo:
        print(f"today's cascade estimate: {ledger.today_usd() * config.usd_to_cad:.4f} CAD")
    return 0


if __name__ == "__main__":
    code = main()
    # Everything is written and closed by now. MLX (0.32) destroys its thread-local compile
    # cache after Python has finalised and now and then segfaults there (2 of 6 runs with a
    # switched voice, 2026-10-10: exit 139 after the report), so the Python exit handlers run
    # and the C teardown is skipped.
    atexit._run_exitfuncs()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
