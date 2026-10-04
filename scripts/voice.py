"""Talk to Pepin (a first, unpolished loop).

The array's voice from the board -> an utterance cut by its
loudness -> Gemini hears the audio itself (no separate speech-to-text), decides whether it was
addressed ("Пепин"), calls the robot's tools (pepin.tools: go_to, where_am_i, list_places, ...)
-> its answer is spoken through the speaker on the board (macOS say rendered, played by the
board's audio server). Every turn is logged to data/voice/<day>.jsonl with timings, and every
utterance heard (answered or not) is kept as data/voice/<day>/<HHMMSS_mmm>.wav ("wav" in its line).

    uv run python scripts/voice.py [--model gemini-3.1-flash-lite]

The key: GEMINI_API_KEY in the environment or in the repo's gitignored .env.

Nothing here is tuned: the loudness gate adapts to the room's floor, the model is told to answer
IGNORE to anything not addressed to it. Ctrl-C stops listening (and cancels a drive in flight).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import struct
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

from pepin.audio_link import FRAME_SAMPLES, RATE, AudioClient
from pepin.face_events import VoiceFace
from pepin.head_link import HeadClient
from pepin.tools import TOOLS, Robot
from pepin.tools.registry import render
from pepin.tools.schemas import gemini_function_declarations

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / ".env"  # gitignored
SYSTEM = (
    "You are Pepin, a small home robot (a wheeled cart with a camera head) living in a flat with"
    " Artem. You hear a microphone in the room: most of what you hear is NOT for you. Act only"
    " when the speaker addresses you by name (Пепин / Pepin, possibly misheard: Пепен, Пипин,"
    " Пепи) or clearly continues a conversation you are in. If not addressed, answer exactly"
    " IGNORE and call no tool. When addressed: use the tools to act (list_places gives the names"
    " go_to accepts; never invent a place). Facts about yourself and the flat (where you are,"
    " what places exist, what you see) come ONLY from a tool call made now: where_am_i for"
    " 'where are you', never a guess. Then answer in the speaker's language in one or two"
    " short spoken sentences, no markdown, no lists. Before a long action (a drive) you may say"
    " what you are about to do. Start EVERY reply with a first line 'HEARD: <what the person"
    " said, verbatim, in their language>' and then the line 'SAY: <your spoken answer>' (or"
    " 'SAY: IGNORE'). Do not call the say tool: your SAY line is spoken for you."
)


def load_env(path: Path) -> None:
    """KEY=VALUE lines into the environment (the gitignored .env of the main checkout)."""
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def rms_db(pcm: bytes) -> float:
    """The frame's loudness in dBFS."""
    n = len(pcm) // 2
    if not n:
        return -120.0
    samples = struct.unpack(f"<{n}h", pcm)
    mean = sum(s * s for s in samples) / n
    return 10 * math.log10(mean / 32768**2 + 1e-12)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_name(t: float) -> str:
    """``<YYYYMMDD>/<HHMMSS_mmm>.wav`` for local time ``t``: the utterance's path under the logs."""
    stamp = time.localtime(t)
    return f"{time.strftime('%Y%m%d/%H%M%S', stamp)}_{int(t % 1 * 1000):03d}.wav"


def save_wav(pcm: bytes, logs: Path, t: float) -> str:
    """The utterance as ``<logs>/<wav_name(t)>``; the path relative to ``logs``."""
    rel = wav_name(t)
    (logs / rel).parent.mkdir(parents=True, exist_ok=True)
    (logs / rel).write_bytes(wav_bytes(pcm))
    return rel


class Ears:
    """Utterances out of the frame stream: start when the level stands START_DB above the room's
    floor for 3 frames, end after SILENCE_S below it; the floor follows the quiet frames."""

    START_DB = 10.0
    SILENCE_S = 0.8
    MAX_S = 15.0
    PRE_S = 0.3

    def __init__(self, client: AudioClient) -> None:
        self.client = client
        self.floor = -55.0
        self.muted_until = 0.0
        self.on_start: Any = None  # called when an utterance starts (the face listens)
        self.reconnect: Any = None

    def utterance(self) -> bytes | None:
        pre: list[bytes] = []
        voiced: list[bytes] = []
        loud_run = quiet = 0
        talking = False
        start = 0.0
        silent = 0
        while True:
            frame = self.client.next_frame(timeout_s=1.0)
            if frame is None:
                silent += 1
                if silent >= 3 and self.reconnect is not None:  # the link died: a new one
                    print("  (audio link lost: reconnecting)", flush=True)
                    with contextlib.suppress(Exception):
                        self.client.close()
                    self.client = self.reconnect()
                    silent = 0
                continue
            silent = 0
            if time.monotonic() < self.muted_until:
                continue  # our own voice through the speaker
            pcm = frame.pcm
            db = rms_db(pcm)
            loud = db > self.floor + self.START_DB
            if not talking:
                self.floor = 0.98 * self.floor + 0.02 * db if not loud else self.floor
                pre.append(pcm)
                pre = pre[-int(self.PRE_S * RATE / FRAME_SAMPLES) :]
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= 3:
                    talking, start, voiced, quiet = True, time.monotonic(), list(pre), 0
                    if self.on_start is not None:
                        self.on_start()
                continue
            voiced.append(pcm)
            quiet = 0 if loud else quiet + 1
            if (
                quiet * FRAME_SAMPLES / RATE >= self.SILENCE_S
                or time.monotonic() - start > self.MAX_S
            ):
                return b"".join(voiced)


class Mind:
    """Gemini with the robot's tools: one heard utterance -> tool calls -> the spoken answer."""

    def __init__(self, model: str, robot: Robot) -> None:
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client()
        self.model = model
        self.robot = robot
        self.history: list[Any] = []
        self.turns: list[list[Any]] = []
        self.tools = [types.Tool(function_declarations=gemini_function_declarations(TOOLS))]
        self.face: VoiceFace | None = None  # --face: the turn's moments on the head's face

    def ask(self, contents: list[Any]) -> Any:
        t = self.types
        for model in (self.model, "gemini-3.5-flash"):
            for thinking in (t.ThinkingConfig(thinking_budget=0), None):
                config = t.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    tools=self.tools,
                    temperature=0.2,
                    max_output_tokens=800,
                    **({"thinking_config": thinking} if thinking else {}),
                )
                try:
                    return self.client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except Exception as error:  # a refused thinking budget: without it; 503: next model
                    print(f"  !! {model}: {str(error)[:100]}", flush=True)
                    if "INVALID_ARGUMENT" not in str(error):
                        break
        raise RuntimeError("no model answered")

    def hear(self, pcm: bytes, log: dict[str, Any]) -> str:
        t = self.types
        turn = t.Content(
            role="user", parts=[t.Part.from_bytes(data=wav_bytes(pcm), mime_type="audio/wav")]
        )
        contents = [*self.history, turn]
        spoken = ""
        for _ in range(6):
            t0 = time.monotonic()
            response = self.ask(contents)
            log.setdefault("model_ms", []).append(round((time.monotonic() - t0) * 1000))
            content = response.candidates[0].content
            contents.append(content)
            parts = list(content.parts or [])
            text = "".join(
                p.text
                for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            )
            for line in text.splitlines():
                if line.startswith("HEARD:"):
                    log["heard"] = line[6:].strip()
                    print(f"heard> {log['heard']}", flush=True)
                if line.startswith("SAY:"):
                    spoken = line[4:].strip()
            calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if spoken and spoken != "IGNORE" and calls:
                self.speak(spoken, log)  # "Еду к принтеру" before the drive
                spoken = ""
            if not calls:
                break
            replies = []
            for call in calls:
                args = dict(call.args or {})
                print(f"  -> {call.name}({args})", flush=True)
                t1 = time.monotonic()
                result = TOOLS.call(call.name, args, self.robot)
                shown, _ = render(result)
                print(f"  <- {shown[:300]}", flush=True)
                log.setdefault("tools", []).append(
                    {
                        "name": call.name,
                        "args": args,
                        "result": shown[:500],
                        "ms": round((time.monotonic() - t1) * 1000),
                    }
                )
                try:
                    payload = json.loads(shown)
                except ValueError:
                    payload = {"text": shown}
                replies.append(t.Part.from_function_response(name=call.name, response=payload))
            contents.append(t.Content(role="user", parts=replies))
        if spoken and spoken != "IGNORE":
            # whole turns only: a history cut inside a tool exchange starts with a function
            # response, which Gemini refuses (400 "function response turn ... after a call")
            self.turns.append(contents[len(self.history) :])
            self.turns = self.turns[-4:]
            self.history = [c for turn in self.turns for c in turn]
        return spoken

    def speak(self, text: str, log: dict[str, Any]) -> float:
        print(f"pepin> {text}", flush=True)
        if self.face is not None:
            self.face.speaking()  # the board's audio server moves the mouth while it plays
        t0 = time.monotonic()
        seconds = float(self.robot.speech.say(text))
        log.setdefault("said", []).append(
            {"text": text, "s": round(seconds, 1), "ms": round((time.monotonic() - t0) * 1000)}
        )
        return seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", default="gemini-3.1-flash-lite"
    )  # 3.7 s a turn with a tool call, measured
    ap.add_argument("--host", default=os.environ.get("PEPIN_HOST", "10.0.0.187"))
    ap.add_argument(
        "--face",
        action="store_true",
        help="show the turn on the head's face (listening, thinking, speaking) through the"
        " board's head server (pepin.head_server, :3340)",
    )
    args = ap.parse_args()
    load_env(ENV_FILE)
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY missing (the environment or the repo's .env)")
        return 2
    robot = Robot.connect()
    mind = Mind(args.model, robot)
    face = VoiceFace(HeadClient(args.host, source="voice").start()) if args.face else None
    mind.face = face

    def connect() -> AudioClient:
        link = AudioClient(args.host).start(listen=True)

        def keepalive() -> None:  # the board closes a link that says nothing for 5 s
            while True:
                time.sleep(2.0)
                try:
                    link._send({"cmd": "status"})
                except Exception:
                    return

        threading.Thread(target=keepalive, daemon=True).start()
        return link

    client = connect()
    ears = Ears(client)
    ears.reconnect = connect
    ears.on_start = face.listening if face is not None else None
    logs = REPO / "data" / "voice"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{time.strftime('%Y%m%d')}.jsonl"
    print(
        f"listening on {args.host} (model {args.model}); say 'Пепин, ...'. Log: {log_path}",
        flush=True,
    )
    try:
        while True:
            pcm = ears.utterance()
            if not pcm:
                continue
            seconds = len(pcm) / 2 / RATE
            if seconds < 0.5:
                if face is not None:
                    face.done()
                continue
            log: dict[str, Any] = {
                "t": time.time(),
                "audio_s": round(seconds, 2),
                "floor_db": round(ears.floor, 1),
            }
            log["wav"] = wav_name(log["t"])  # written beside the model call, off its latency
            threading.Thread(target=save_wav, args=(pcm, logs, log["t"]), daemon=True).start()
            t0 = time.monotonic()
            print(f"\n[{seconds:.1f} s of speech] thinking...", flush=True)
            if face is not None:
                face.thinking()
            try:
                answer = mind.hear(pcm, log)
            except Exception as error:
                log["error"] = repr(error)[:300]
                print(f"  !! {error}", flush=True)
                answer = ""
            if answer:
                spoken_s = mind.speak(answer, log)
                ears.muted_until = time.monotonic() + spoken_s + 0.6
            else:
                print("  (not for me)", flush=True)
            if face is not None:
                face.done()
            log["turn_ms"] = round((time.monotonic() - t0) * 1000)
            with log_path.open("a") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("\nstopping:", robot.halt(), flush=True)
    finally:
        ears.client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
