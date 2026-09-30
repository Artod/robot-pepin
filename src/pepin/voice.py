"""The voice pipeline's attach point: wake word -> speech to text / LLM -> text to speech.

NOTHING IS CHOSEN YET. Gemini Live (through Pipecat) and a local cascade are both open; the
choice follows the first days with the array on the robot. What is fixed is the hand-off, and
:func:`run_voice_pipeline` is the one place the pipeline plugs into:

- in: the robot's hearing, 20 ms s16le mono frames at 16 kHz, already echo-cancelled,
  beamformed and noise-suppressed by the array (:class:`pepin.audio_link.AudioFrame`), and the
  voice's direction in the array's frame (:class:`pepin.audio_link.DoaReading`);
- out: speech as s16le mono PCM at the link's play rate through
  :class:`pepin.audio_link.Speaker` — the array's own output, so the echo canceller removes
  the robot's voice from what it hears; ``flush()`` on a barge-in, ``play_end()`` after each
  utterance. :func:`pepin.audio_link.play_paced` sends a finished utterance at real time.

Run the idle pipeline against the robot (it proves the wiring and does nothing else)::

    uv run python -m pepin.voice --host 10.0.0.187
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable

from pepin.audio_link import AUDIO_PORT, AudioClient, AudioFrame, DoaReading, Speaker, board_host
from pepin.log import setup_logging

logger = logging.getLogger(__name__)


def run_voice_pipeline(
    frames: Iterable[AudioFrame],
    latest_doa: Callable[[], DoaReading | None],
    speaker: Speaker,
) -> int:
    """THE ATTACH POINT of the voice pipeline. Receives the robot's hearing frame by frame and
    can ask for the voice's direction at any moment; does nothing with them yet. Returns the
    number of frames it consumed (when the stream ends).

    Wake word -> STT/LLM -> TTS attaches here: a wake-word model fed every frame; after the
    wake, the frames to the chosen speech-to-speech session or STT; its answer to ``speaker``;
    ``latest_doa()`` at the wake says where to turn the head (a separate task).
    """
    consumed = 0
    for _frame in frames:
        consumed += 1  # the pipeline goes here
    return consumed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the (idle) voice pipeline on the robot.")
    parser.add_argument("--host", default=board_host())
    parser.add_argument("--port", type=int, default=AUDIO_PORT)
    args = parser.parse_args()
    setup_logging("voice")
    client = AudioClient(args.host, args.port).start()
    logger.info("connected: %s", client.hello)
    try:
        frames = run_voice_pipeline(client.frames(), client.latest_doa, client)
        logger.info("stream ended after %d frames", frames)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
