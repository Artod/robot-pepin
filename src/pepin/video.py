"""Record the robot's camera on the board during a session and fetch the file afterwards.

The camera is a USB device on the Orange Pi, so the recording runs there:
ffmpeg copies the MJPEG frames served by the board's camera streamer (the
same stream the dashboard shows) into an .mkv without re-encoding, cheap on
the Zero 3. Started and stopped over ssh by the session scripts, the file is
then copied to ``data/videos/`` for demo editing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

STREAM_URL = "http://127.0.0.1:8080/stream"  # ustreamer (pepin-camera.service) on the board
REMOTE_DIR = "/opt/pepin/videos"


class CameraRecorder:
    """ffmpeg on the board writing ``<name>.mkv``; ``stop()`` fetches it to ``local_dir``."""

    def __init__(
        self,
        host: str,
        name: str,
        local_dir: str | Path = "data/videos",
        source: str = STREAM_URL,
    ) -> None:
        """Records ``source`` (by default the board's own MJPEG stream) into ``<name>.mkv``.

        The camera device itself is held by the streamer service, so recording
        taps the stream instead of the device; the dashboard keeps its picture.
        """
        self._ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"root@{host}"]
        self._host = host
        self._remote = f"{REMOTE_DIR}/{name}.mkv"
        self._pidfile = f"{REMOTE_DIR}/{name}.pid"
        self._local = Path(local_dir) / f"{name}.mkv"
        self._source = source
        self.started = False

    def start(self) -> None:
        """Launch ffmpeg on the board; ``started`` means it was still alive a second later."""
        log = f"{REMOTE_DIR}/ffmpeg.log"
        # "mkdir && nohup ... &" would background the whole list as a subshell: $! would be
        # the subshell, whose stdout keeps the ssh session open forever. Hence ";" not "&&".
        cmd = (
            f"mkdir -p {REMOTE_DIR}; "
            f"nohup ffmpeg -loglevel error -y -i {self._source} -c:v copy {self._remote} "
            f"> {log} 2>&1 < /dev/null & echo $! > {self._pidfile}; sleep 1; "
            f"kill -0 $(cat {self._pidfile}) && echo started || tail -2 {log}"
        )
        try:
            result = subprocess.run([*self._ssh, cmd], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("camera recording did not start: ssh timed out")
            return
        self.started = "started" in result.stdout
        if self.started:
            logger.info("camera recording -> %s:%s", self._host, self._remote)
        else:
            logger.warning(
                "camera recording did not start: %s", (result.stdout + result.stderr).strip()[:160]
            )

    def stop(self) -> Path | None:
        """Stop ffmpeg gracefully (SIGINT finalises the file) and copy the video to the laptop."""
        if not self.started:
            return None
        # SIGINT to our own ffmpeg only (by pid): a second recorder on the board must survive.
        subprocess.run(
            [*self._ssh, f"kill -INT $(cat {self._pidfile}) 2>/dev/null; sleep 1.5"],
            capture_output=True,
            timeout=20,
        )
        self._local.parent.mkdir(parents=True, exist_ok=True)
        fetched = subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                f"root@{self._host}:{self._remote}",
                str(self._local),
            ],
            capture_output=True,
            timeout=600,
        )
        if fetched.returncode != 0:
            logger.warning("could not fetch the video: %s", fetched.stderr.decode()[:120])
            return None
        logger.info("camera video saved to %s", self._local)
        return self._local
