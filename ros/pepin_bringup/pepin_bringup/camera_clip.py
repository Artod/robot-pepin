"""The head camera's MJPEG stream copied next to a run's recording, by the run's recorder.

A clip is a plain copy of the stream (curl, a few percent of a core, no re-encoding), started
when a run opens and ended when it closes. Both recorders own one — the JSONL tape's and the
bag's — and the clip lands beside either under the run's own stem
(:func:`pepin.tape.camera_clip_path`), so EVERY run has its video whoever sent the goal
(ros/goto.sh, the tray, the voice tools): the owner of the run is the owner of its film.

The raw copy keeps every part's headers, and the URL asks ustreamer for its extra ones
(``?extra_headers=1``): the capture stamp beside the send stamp, which is what
``ros/tools/clip_to_bag.py`` dates the frames by when it turns a clip into a camera bag
(:func:`pepin.mjpeg.capture_time`).

WHERE THE STREAM IS: ``config/camera.json``'s own ``stream`` (the one camera_stream reads), its
``{board}`` filled with ``PEPIN_HOST`` — the board's address on the laptop, where the recorders
run beside Nav2 since 2026-10-01. Until 2026-10-04 the URL was the board's loopback
(127.0.0.1:8080): right while the recorders ran on the board, silently empty since they moved —
curl found nothing on the container's loopback, the empty file was removed and nothing was said,
so the laptop's runs had no clip of their own. So the clip says aloud now: when it starts, when it
has no bytes ``CHECK_S`` in (and starts once more), when the stream ends mid-run, and its size or
its absence when it stops.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pepin.tape import camera_clip_path

CAMERA_CONFIG = "/ws/config/camera.json"  # the containers' mount of config/
BOARD_LOOPBACK = "127.0.0.1"  # PEPIN_HOST unset: the recorder runs on the board itself
# The URL when config/camera.json cannot be read: ustreamer's port, with the capture stamps
FALLBACK_STREAM = "http://{board}:8080/stream?extra_headers=1"
CLIP_MAX_S = 1800  # curl's own limit: a clip nobody stops is a bug, as a tape nobody stops is
CHECK_S = 4.0  # a clip with no bytes this long after its start is said aloud and started again


def camera_stream(
    environ: Mapping[str, str] | None = None, config: str | Path = CAMERA_CONFIG
) -> str:
    """The stream a run's clip copies: config/camera.json's ``stream`` of the active camera,
    its ``{board}`` the ``PEPIN_HOST`` of ``environ`` (the board's loopback without one)."""
    env = os.environ if environ is None else environ
    board = env.get("PEPIN_HOST", BOARD_LOOPBACK) or BOARD_LOOPBACK
    try:
        from pepin.camera import CameraConfig

        return CameraConfig.load(config, board=board, environ=env).stream
    except (OSError, ValueError, KeyError, TypeError):
        return FALLBACK_STREAM.format(board=board)


class CameraClip:
    """One run's video: ``start`` copies the stream beside the recording, ``check`` (once a
    second from the recorder's timer) says a clip that is not growing, ``stop`` ends it."""

    def __init__(
        self,
        logger: Any,
        stream: str | None = None,
        check_s: float = CHECK_S,
        clock: Any = time.monotonic,
    ) -> None:
        self._log = logger
        self._stream = camera_stream() if stream is None else stream
        self._check_s = check_s
        self._now = clock
        self._process: subprocess.Popen[bytes] | None = None
        self._path: Path | None = None
        self._started = 0.0
        self._retried = False
        self._checked = False
        self._ended_said = False

    @property
    def stream(self) -> str:
        """The URL every clip of this recorder copies."""
        return self._stream

    @property
    def path(self) -> Path | None:
        """The clip being written, if any."""
        return self._path

    def start(self, recording: Path) -> None:
        """Begin a clip beside ``recording``; a stream that cannot be opened is logged, not
        raised — no drive is lost over its video."""
        self.stop()
        self._path = camera_clip_path(recording)
        self._retried = False
        self._checked = False
        self._ended_said = False
        self._spawn()
        self._log.info(f"camera clip: {self._stream} -> {self._path}")

    def _spawn(self) -> None:
        assert self._path is not None
        self._started = self._now()
        try:
            self._process = subprocess.Popen(
                ["curl", "-sS", "-m", str(CLIP_MAX_S), self._stream, "-o", str(self._path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._process = None
            self._log.error(f"!! camera clip not started: {exc}")

    def _size(self) -> int:
        try:
            return self._path.stat().st_size if self._path is not None else 0
        except OSError:
            return 0

    def _said(self, process: subprocess.Popen[bytes] | None) -> str:
        """curl's own last words (stderr) once it has exited, or its exit status."""
        if process is None or process.poll() is None:
            return "curl is still waiting"
        err = b""
        if process.stderr is not None:
            try:
                err = process.stderr.read() or b""
            except (OSError, ValueError):
                err = b""
        text = err.decode(errors="replace").strip().splitlines()
        return text[-1] if text else f"curl exited with {process.returncode}"

    def check(self) -> None:
        """Once a second while a run is open: a clip with no bytes ``check_s`` after its start is
        said aloud and started once more, a second miss is an error; a stream that ended before
        the run did is said once."""
        if self._path is None:
            return
        process = self._process
        if not self._checked and self._now() - self._started >= self._check_s:
            if self._size() > 0:
                self._checked = True
            elif not self._retried:
                self._log.warning(
                    f"!! the camera clip is NOT recording ({self._said(process)};"
                    f" {self._stream}): starting it again"
                )
                self._end(process)
                self._retried = True
                self._spawn()
            else:
                self._checked = True
                self._log.error(
                    f"!! still no camera clip: this run has no picture ({self._said(process)};"
                    f" {self._stream})"
                )
        elif (
            self._checked
            and not self._ended_said
            and process is not None
            and process.poll() is not None
            and self._size() > 0
        ):
            self._ended_said = True
            self._log.warning(
                f"!! the camera stream ended before the run did ({self._said(process)})"
            )

    def _end(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    def stop(self) -> None:
        """End the clip and say what it came to: its size, or why there is none (an empty file
        is removed)."""
        process, self._process = self._process, None
        path, self._path = self._path, None
        if path is None:
            return
        self._end(process)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > 0:
            self._log.info(f"camera clip: {size / 1048576:.1f} MB, {path}")
            return
        path.unlink(missing_ok=True)
        self._log.error(f"!! no camera clip for this run ({self._said(process)}; {self._stream})")
