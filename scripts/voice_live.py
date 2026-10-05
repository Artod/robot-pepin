"""Talk to Pepin over the Gemini Live API: native audio both ways, the robot's tools, a local wake
gate in front of every paid second.

The board's array -> a loudness-cut segment -> Whisper on the Mac -> "Пепин" opens (or resumes)
a Live session -> the conversation streams both ways until "пока" / "всё, спасибо", the model's
end_conversation, 25 s of silence, or a cap. The robot's speech plays on the board's speaker as
it streams. Caps, prices and timings: config/voice_live.json. Logs: data/voice/<day>/.

    uv run python scripts/voice_live.py                  # the robot, for real
    uv run python scripts/voice_live.py --dry-run        # no Google: a local beep answers

A test with no person and no sound in the flat (phrases spoken by macOS say, the robot's
speech written to a WAV, only read-only tools):

    uv run python scripts/voice_live.py --say "Пепин, где ты?" --say "Всё, спасибо, пока" \\
        --speaker-wav /tmp/pepin.wav --tools where_am_i,list_places --sessions 1

A drive with nothing moving (the tools act on fakes; a drive takes 5 s):

    uv run python scripts/voice_live.py --say "Пепин, езжай к принтеру" \\
        --speaker-wav /tmp/pepin.wav --fake-robot 5 --sessions 1

The key: GEMINI_API_KEY in the environment or in the repo's gitignored .env (the main checkout's,
from a worktree). scripts/voice.py stays the per-utterance fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from pepin.audio_link import board_host
from pepin.face_events import VoiceStateFace
from pepin.head_link import HeadClient
from pepin.tools import TOOLS, Robot
from pepin.tools.fakes import FakeGoalServer, WallClock, fake_robot
from pepin.tools.schemas import gemini_function_declarations
from pepin.voice_live.app import VoiceLoop
from pepin.voice_live.audio import Pacer
from pepin.voice_live.budget import Ledger
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events, StatePrinter
from pepin.voice_live.fake import DryRunLive
from pepin.voice_live.live import GeminiLive, LiveConnector, declarations
from pepin.voice_live.session import Conversation, Frame, ResumeStore
from pepin.voice_live.sources import BoardLink, ScriptedMic, WavSpeaker, synthesize
from pepin.voice_live.wake import MlxWhisper

REPO = Path(__file__).resolve().parents[1]
NOT_FOR_LIVE = {"say"}  # the model speaks for itself
FAKE_ACCEPT_S = 0.3  # a fake drive is taken this long after the go (the recorder's start)


def env_file() -> Path:
    """The repo's .env; from a worktree (<main>/.claude/worktrees/<name>) the main checkout's."""
    if (REPO / ".env").is_file() or REPO.parent.parent.name != ".claude":
        return REPO / ".env"
    return REPO.parents[2] / ".env"


def load_env(path: Path) -> None:
    """KEY=VALUE lines into the environment; nothing is printed."""
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, help="default: config/voice_live.json")
    ap.add_argument("--host", default=board_host(), help="the board (PEPIN_HOST)")
    ap.add_argument("--dry-run", action="store_true", help="no Google: a local beep answers")
    ap.add_argument("--say", action="append", help="speak this into the loop instead of the mic")
    ap.add_argument("--speaker-wav", type=Path, help="write the robot's speech here, no sound")
    ap.add_argument(
        "--say-gap-s", type=float, default=1.2,
        help="each --say waits for the robot to be idle or listening this long",
    )  # fmt: skip
    ap.add_argument("--tools", help="comma-separated: only these tools (default: all but say)")
    ap.add_argument("--sessions", type=int, help="exit after this many sessions")
    ap.add_argument(
        "--fake-robot", type=float, metavar="DRIVE_S",
        help="no robot: the tools act on fakes (nothing moves), a drive takes DRIVE_S seconds",
    )  # fmt: skip
    ap.add_argument(
        "--no-face", action="store_true",
        help="leave the head's face alone (it follows the voice when the board speaks)",
    )  # fmt: skip
    ap.add_argument("--max-session-s", type=float, help="override the cap for this run")
    ap.add_argument("--idle-close-s", type=float, help="override the idle timeout for this run")
    ap.add_argument("--logs", type=Path, default=REPO / "data" / "voice")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname).1s %(name)s: %(message)s")

    config = LiveConfig.load(args.config).with_overrides(
        max_session_s=args.max_session_s, idle_close_s=args.idle_close_s
    )
    names = [t.name for t in TOOLS if t.name not in NOT_FOR_LIVE]
    if args.tools:
        wanted = {n.strip() for n in args.tools.split(",") if n.strip()}
        unknown = wanted - set(names)
        if unknown:
            print(f"unknown tools: {', '.join(sorted(unknown))}; the tools are {', '.join(names)}")
            return 2
        names = [n for n in names if n in wanted]
    moving = {t.name for t in TOOLS if t.moves and t.name in names}
    allowed = set(names)

    live: LiveConnector
    ledger: Ledger | None = None
    if args.dry_run:
        live = DryRunLive()
    else:
        load_env(env_file())
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY missing (the environment or the repo's .env)")
            return 2
        ledger = Ledger(args.logs / "ledger.jsonl", config)
        refusal = ledger.refusal()
        if refusal:
            print(f"refusing to start: {refusal}")
            return 3
        registry = [d for d in gemini_function_declarations(TOOLS) if d["name"] in allowed]
        live = GeminiLive(config, declarations(registry, moving))

    if args.fake_robot is not None:
        clock = WallClock()
        step_s = max(0.0, args.fake_robot - FAKE_ACCEPT_S) / 3  # accepted, 2 reports, done
        goals = FakeGoalServer(clock, step_s=step_s, accept_s=FAKE_ACCEPT_S)
        robot = fake_robot(clock=clock, goals=goals)
        print(f"FAKE ROBOT: nothing moves; a drive takes {args.fake_robot:.1f} s", flush=True)
    else:
        robot = Robot.connect()

    def run_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name not in allowed:
            return {"ok": False, "why": f"{name} is not available in this conversation"}
        return TOOLS.call(name, arguments, robot)

    events = Events(StatePrinter())
    face: VoiceStateFace | None = None
    if args.speaker_wav is None and not args.no_face:  # the robot speaks: its face follows
        face = VoiceStateFace(HeadClient(args.host, source="voice").start())
        events.subscribe(face)
    speaker: WavSpeaker | BoardLink
    link: BoardLink | None = None
    if args.speaker_wav is not None:
        speaker = WavSpeaker(args.speaker_wav)
    else:
        link = BoardLink(args.host)
        speaker = link
    pacer = Pacer(
        speaker,
        rate=speaker.play_rate,
        lead_s=config.playback_lead_s,
        on_level=lambda value: events.emit("speaking", value),
    )
    stop_pacer = pacer.start()

    mic: ScriptedMic | BoardLink
    if args.say:

        def ready() -> bool:
            return events.state in ("idle", "listening") and not pacer.speaking()

        mic = ScriptedMic([synthesize(text) for text in args.say], ready, gap_s=args.say_gap_s)
    else:
        mic = link or BoardLink(args.host)

    print(f"loading the wake gate ({config.whisper_repo})...", flush=True)
    whisper = MlxWhisper(config.whisper_repo, initial_prompt=config.whisper_prompt)
    logs = args.logs
    resume = ResumeStore(logs / "resume.json", config.resume_valid_s)

    conversations: list[Conversation] = []

    def conversation(frames: asyncio.Queue[Frame]) -> Conversation:
        conversations.append(
            Conversation(
                live,
                config,
                frames=frames,
                pacer=pacer,
                tools=run_tool,
                moving=moving,
                halt=robot.halt,
                events=events,
                logs=logs,
                ledger=ledger,
                resume=resume,
                transcriber=whisper,
            )
        )
        return conversations[-1]

    loop = VoiceLoop(
        config,
        mic=mic,
        pacer=pacer,
        transcriber=whisper,
        conversation_factory=conversation,
        events=events,
        logs=logs,
        ledger=ledger,
    )
    mode = "DRY RUN (no Google)" if args.dry_run else f"{config.model}, voice {config.voice}"
    print(
        f"listening ({mode}); say 'Пепин, ...'. Caps: {config.max_session_s:.0f} s a session,"
        f" {config.max_sessions_per_hour} sessions an hour, {config.daily_budget_cad:.2f} CAD a"
        f" day (estimate). Logs: {logs}",
        flush=True,
    )
    try:
        asyncio.run(loop.run(max_sessions=args.sessions))
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
        if any(c.moved for c in conversations):
            print("halt:", robot.halt(), flush=True)
    finally:
        stop_pacer.set()
        if face is not None:
            face.close()
        for closable in (mic, speaker):
            close = getattr(closable, "close", None)
            if close is not None:
                close()
    if ledger is not None:
        print(f"today's estimate: {ledger.today_usd() * config.prices.usd_to_cad:.4f} CAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
