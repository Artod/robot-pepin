"""The head camera's MJPEG stream copied next to a run's recording, on the board.

A clip is a plain copy of the stream (curl, a few percent of a core, no re-encoding), started
when a run opens and ended when it closes, so a laptop that cannot reach the camera still gets
the video with the recording. Both recorders own one — the JSONL tape's and the bag's — and the
clip lands beside either under the run's own stem (:func:`pepin.tape.camera_clip_path`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from pepin.tape import camera_clip_path

CAMERA_STREAM = "http://127.0.0.1:8080/stream"  # ustreamer on the board's host network
CLIP_MAX_S = 1800  # curl's own limit: a clip nobody stops is a bug, as a tape nobody stops is


class CameraClip:
    """One run's video: ``start`` copies the stream beside the recording, ``stop`` ends it."""

    def __init__(self, logger: Any, stream: str = CAMERA_STREAM) -> None:
        self._log = logger
        self._stream = stream
        self._process: subprocess.Popen[bytes] | None = None
        self._path: Path | None = None

    def start(self, recording: Path) -> None:
        """Begin a clip beside ``recording``; a stream that cannot be opened is logged, not
        raised — no drive is lost over its video."""
        self.stop()
        self._path = camera_clip_path(recording)
        try:
            self._process = subprocess.Popen(
                ["curl", "-s", "-m", str(CLIP_MAX_S), self._stream, "-o", str(self._path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._process = None
            self._log.warning(f"camera clip not started: {exc}")

    def stop(self) -> None:
        """End the clip; a stream that was never reachable leaves an empty file, removed here."""
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
        path, self._path = self._path, None
        if path is not None and path.exists() and path.stat().st_size == 0:
            path.unlink()
