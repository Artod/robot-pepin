"""Talk to Pepin through its own voice loop, the cascade: local ears -> Claude -> local voice.

The board's array -> a loudness-cut utterance -> Whisper on the Mac (and Praat's tag line) ->
"Пепин" opens a conversation -> Claude streams the reply, sentence by sentence, calling the
robot's tools -> Qwen3-TTS on the Mac speaks each sentence as it arrives -> the board's speaker.
Every turn prints its timeline (person stopped -> cut -> text -> token -> sentence -> tts ->
play -> end, in ms) with the cache tokens and the price. Every switch: config/voice_cascade.json,
or --set section.key=value for one run. Logs: data/voice/<day>/cascade_<HHMMSS>/.

The key comes from the environment (the repo's gitignored .env, loaded by the shell):

    set -a; . ./.env; set +a
    uv run python scripts/voice_cascade.py                          # the robot, for real
    uv run python scripts/voice_cascade.py --brain sonnet --set flags.barge_in=true
    uv run python scripts/voice_cascade.py --brain flashlite     # Gemini (GEMINI_API_KEY)
    uv run python scripts/voice_cascade.py --brain groq-qwen     # Groq (GROQ_API_KEY)
    uv run python scripts/voice_cascade.py --set ears.turn_model=smart-turn-v3.2   # Smart Turn
    uv run python scripts/voice_cascade.py --set transcriber.engine=whispercpp  # Neural Engine
    uv run python scripts/voice_cascade.py --voice gemini-lite   # Gemini TTS (GEMINI_API_KEY)

Switch the brain, the voice or the speech-to-text engine of the running loop (from another
terminal; it writes data/voice/control.json and prints the loop's answers: a brain is ready in
about a second, a voice once its model has loaded, a transcriber between utterances):

    uv run python scripts/voice_cascade.py switch --brain sonnet --voice qwen3-1.7b
    uv run python scripts/voice_cascade.py switch --transcriber whispercpp

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

from pepin.audio_link import board_host
from pepin.deployment import config_file
from pepin.tools import TOOLS, Robot
from pepin.tools.fakes import FakeCamera, FakeGoalServer, WallClock, fake_robot
from pepin.voice_cascade.bench import ReplayMic, utterances
from pepin.voice_cascade.config import (
    CONFIG_FILE,
    TRANSCRIBER_ENGINES,
    CascadeConfig,
    voice_preset,
)
from pepin.voice_cascade.control import (
    DONE_STATES,
    ControlFile,
    Switchboard,
    answers_to,
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
from pepin.voice_cascade.loop import CascadeLoop, Mic
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


def switch(argv: list[str]) -> int:
    """``switch --brain X --voice Y --transcriber Z``: ask the running loop for a switch and
    print its answers until each part is applied or has failed."""
    ap = argparse.ArgumentParser(
        prog="voice_cascade.py switch",
        description="switch the running loop's brain, voice or speech-to-text engine",
    )
    ap.add_argument("--brain", help="a brain preset, or echo")
    ap.add_argument("--voice", help="a voice preset (or an engine's name: qwen3, gemini, say)")
    ap.add_argument(
        "--transcriber",
        help=f"{', '.join(TRANSCRIBER_ENGINES)}, or engine:model (whispercpp:large-v3)",
    )
    ap.add_argument("--config", type=Path, help="default: config/voice_cascade.json")
    ap.add_argument("--logs", type=Path, default=REPO / "data" / "voice")
    ap.add_argument("--wait-s", type=float, default=SWITCH_WAIT_S)
    args = ap.parse_args(argv)
    if not (args.brain or args.voice or args.transcriber):
        ap.error("--brain, --voice, --transcriber, or several")
    data = json.loads((args.config or config_file(CONFIG_FILE)).read_text())
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
    request = write_request(args.logs, args.brain, args.voice, args.transcriber)
    print(
        f"asked ({request.id}): brain {args.brain}, voice {args.voice},"
        f" transcriber {args.transcriber}",
        flush=True,
    )
    asked = (("brain", args.brain), ("voice", args.voice), ("transcriber", args.transcriber))
    parts = {what for what, name in asked if name}
    deadline = time.monotonic() + args.wait_s
    shown = 0
    failed = False
    while time.monotonic() < deadline:
        rows = answers_to(args.logs, request.id)
        for row in rows[shown:]:
            extra = {k: v for k, v in row.items() if k not in ("id", "t", "what", "name", "state")}
            print(f"  {row['what']} {row['name']}: {row['state']} {json.dumps(extra)}", flush=True)
        shown = len(rows)
        done = {row["what"] for row in rows if row["state"] in DONE_STATES}
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
    if fake_drive_s is not None:
        clock = WallClock()
        step_s = max(0.0, fake_drive_s - FAKE_ACCEPT_S) / 3  # accepted, 2 reports, done
        goals = FakeGoalServer(clock, step_s=step_s, accept_s=FAKE_ACCEPT_S)
        robot = fake_robot(clock=clock, goals=goals, camera=FakeCamera(jpeg=fake_view_jpeg()))
        print(f"FAKE ROBOT: nothing moves; a drive takes {fake_drive_s:.1f} s", flush=True)
    else:
        robot = Robot.connect()

    def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return TOOLS.call(name, arguments, robot)

    events = Events(StatePrinter())
    quiet_run = args.speaker_wav is not None or bench
    face: Any = None
    if not quiet_run and not args.no_face:  # the robot speaks: its face follows
        from pepin.face_events import VoiceStateFace
        from pepin.head_link import HeadClient

        face = VoiceStateFace(HeadClient(args.host, source="voice").start())
        events.subscribe(face)

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
    clips = (
        *(config.backchannel_lines if config.backchannel else ()),
        config.tool_filler,
        *config.openers,
    )
    clips = tuple(line for line in clips if line)
    speech = SpeechWorker(voice, player, clips=clips)
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
    if config.turn_model is not None:
        print(f"loading the end-of-turn model ({config.turn_model})...", flush=True)
        turn = TurnWorker(SmartTurn(config.turn_model))
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
    )
    loop.paid_ledger = ledger
    holder.append(loop)
    b = config.brain
    print(
        f"listening: {setup.name} = {b.model} (thinking {b.thinking}, effort {b.effort}, cache"
        f" {b.cache}), voice {config.voice.name} ({config.voice.model}), wake {config.wake};"
        f" barge_in {config.barge_in}, backchannel {config.backchannel}, prosody_tags"
        f" {config.prosody_tags}, turn_model {config.turn_model}. Logs: {args.logs}; switch:"
        " scripts/voice_cascade.py switch --brain NAME --voice NAME",
        flush=True,
    )
    try:
        asyncio.run(loop.run(turns=turns))
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
        if moving and fake_drive_s is None:
            print("halt:", robot.halt(), flush=True)
    finally:
        speech.close()
        if turn is not None:
            turn.close()
        transcriber.close()  # whisper.cpp's server, when that engine is in use
        if face is not None:
            face.close()
        for closable in (mic, speaker):
            close = getattr(closable, "close", None)
            if close is not None:
                close()
    person = [t for t in loop.turns if t["kind"] == "person" and t["outcome"] != "merged"]
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
