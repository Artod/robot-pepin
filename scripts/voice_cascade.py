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
    uv run python scripts/voice_cascade.py --set ears.turn_model=smart-turn-v3.2   # Smart Turn

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
from pepin.tools.schemas import gemini_function_declarations
from pepin.voice_cascade.bench import ReplayMic, utterances
from pepin.voice_cascade.brain import ClaudeBrain, tool_definitions
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.fake import ScriptedBrain
from pepin.voice_cascade.gemini_brain import GeminiBrain
from pepin.voice_cascade.loop import CascadeLoop, Mic
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechWorker
from pepin.voice_cascade.stages import Brain, Tagger, Voice
from pepin.voice_cascade.tagger import ProsodyTagger, emotion_model
from pepin.voice_cascade.timeline import report
from pepin.voice_cascade.turn import SmartTurn, TurnWorker
from pepin.voice_cascade.voice import QwenVoice, SayVoice
from pepin.voice_live.budget import Ledger
from pepin.voice_live.config import LiveConfig, Prices
from pepin.voice_live.events import Events, StatePrinter
from pepin.voice_live.sources import BoardLink, ScriptedMic, WavSpeaker, synthesize
from pepin.voice_live.wake import MlxWhisper

REPO = Path(__file__).resolve().parents[1]
NOT_FOR_THE_BRAIN = {"say"}  # the brain speaks through the voice, not through a tool
FAKE_ACCEPT_S = 0.3  # a fake drive is taken this long after the go


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


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, help="default: config/voice_cascade.json")
    ap.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one config value for this run (JSON value), e.g. flags.barge_in=true",
    )  # fmt: skip
    ap.add_argument(
        "--brain",
        help="a brain preset (haiku, sonnet, opus, flashlite, flash, flash38), or echo: no API,"
        " repeats the words",
    )
    ap.add_argument("--voice", choices=["qwen3", "say"], help="override voice.engine")
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

    overrides = list(args.set)
    echo = args.brain == "echo"
    if args.brain and not echo:
        overrides.append(f"brain.preset={args.brain}")
    if args.voice:
        overrides.append(f"voice.engine={args.voice}")
    if args.bench is not None:
        overrides.append("ears.wake=always")
    config = CascadeConfig.load(args.config, overrides)

    names = [t.name for t in TOOLS if t.name not in NOT_FOR_THE_BRAIN]
    if args.tools:
        wanted = {n.strip() for n in args.tools.split(",") if n.strip()}
        unknown = wanted - set(names)
        if unknown:
            print(f"unknown tools: {', '.join(sorted(unknown))}; the tools are {', '.join(names)}")
            return 2
        names = [n for n in names if n in wanted]
    allowed = set(names)
    moving = {t.name for t in TOOLS if t.moves and t.name in allowed}

    brain: Brain
    ledger: Ledger | None = None
    if echo:
        brain = ScriptedBrain()
    else:
        key = "GEMINI_API_KEY" if config.brain.provider == "gemini" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            print(f"{key} missing: set -a; . ./.env; set +a (the main checkout's .env)")
            return 2
        ledger = ledger_for(config, args.logs / "cascade_ledger.jsonl")
        refusal = ledger.refusal()
        if refusal:
            print(f"refusing to start: {refusal}")
            return 3
        persona = config_file(config.brain.persona_file).read_text()
        if config.brain.provider == "gemini":
            declarations = [d for d in gemini_function_declarations(TOOLS) if d["name"] in allowed]
            brain = GeminiBrain(
                config.brain, persona=persona, declarations=declarations, price=config.price
            )
        else:
            brain = ClaudeBrain(
                config.brain,
                persona=persona,
                tools=tool_definitions(TOOLS, allowed),
                price=config.price,
            )

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

    voice: Voice
    if config.voice_engine == "qwen3":
        hf_home = Path(config.tts_hf_home) if config.tts_hf_home else None
        if hf_home is not None and not hf_home.is_absolute():
            hf_home = main_checkout() / hf_home
        voice = QwenVoice(
            config.tts_model,
            voice=config.tts_voice,
            language=config.tts_language,
            chunk_s=config.tts_chunk_s,
            hf_home=hf_home,
        )
    else:
        voice = SayVoice(config.say_voice, config.say_rate_wpm)
    clips = (*(config.backchannel_lines if config.backchannel else ()), config.tool_filler)
    clips = tuple(line for line in clips if line)
    speech = SpeechWorker(voice, player, clips=clips)
    print(f"loading the voice ({config.voice_engine}) and Whisper ({config.whisper_repo})...")
    speech.start()  # the voice loads on its own thread meanwhile
    whisper = MlxWhisper(config.whisper_repo, config.language, config.whisper_prompt)
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

    loop = CascadeLoop(
        config,
        mic=mic,
        player=player,
        transcriber=whisper,
        tagger=tagger,
        brain=brain,
        speech=speech,
        tools=run_tool,
        allowed=allowed,
        moving=moving,
        halt=robot.halt,
        events=events,
        logs=args.logs,
        ledger=ledger,
        turn=turn,
    )
    holder.append(loop)
    b = config.brain
    print(
        f"listening: {b.model} (thinking {b.thinking}, effort {b.effort}, cache {b.cache}),"
        f" voice {config.voice_engine}, wake {config.wake}; barge_in {config.barge_in},"
        f" backchannel {config.backchannel}, prosody_tags {config.prosody_tags},"
        f" turn_model {config.turn_model}. Logs: {args.logs}",
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
        if face is not None:
            face.close()
        for closable in (mic, speaker):
            close = getattr(closable, "close", None)
            if close is not None:
                close()
    person = [t for t in loop.turns if t["kind"] == "person" and t["outcome"] != "merged"]
    if person:
        print("\n" + "\n".join(report(person)), flush=True)
    if bench and person:
        out = args.logs / time.strftime("%Y%m%d") / f"cascade_bench_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": b.model,
            "turn_model": config.turn_model,
            "thinking": b.thinking,
            "effort": b.effort,
            "voice": config.voice_engine,
            "source": str(args.bench),
            "report": report(person),
            "turns": loop.turns,
        }
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
        print(f"bench: {out}", flush=True)
    if ledger is not None:
        print(f"today's cascade estimate: {ledger.today_usd() * config.usd_to_cad:.4f} CAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
