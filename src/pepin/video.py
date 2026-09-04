"""Record the robot's camera on the board during a session and fetch the file afterwards.

The camera is a USB device on the Orange Pi, so the recording runs there:
ffmpeg copies the camera's MJPEG stream into an .mkv without re-encoding
(cheap on the Zero 3). Started and stopped over ssh by the session scripts,
the file is then copied next to the session recording for demo editing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

OVERVIEW_CAMERA = "/dev/v4l/by-id/usb-XIFT_webcam_AC310_20250819-video-index0"
REMOTE_DIR = "/opt/pepin/videos"


class CameraRecorder:
    """ffmpeg on the board writing ``<name>.mkv``; ``stop()`` fetches it to ``local_dir``."""

    def __init__(
        self,
        host: str,
        name: str,
        local_dir: str | Path = "data/videos",
        device: str = OVERVIEW_CAMERA,
        size: str = "1280x720",
        fps: int = 30,
    ) -> None:
        self._ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"root@{host}"]
        self._host = host
        self._remote = f"{REMOTE_DIR}/{name}.mkv"
        self._local = Path(local_dir) / f"{name}.mkv"
        self._device, self._size, self._fps = device, size, fps
        self.started = False

    def start(self) -> None:
        cmd = (
            f"mkdir -p {REMOTE_DIR} && nohup ffmpeg -loglevel error -y -f v4l2 -input_format mjpeg "
            f"-video_size {self._size} -framerate {self._fps} -i {self._device} -c:v copy "
            f"{self._remote} > {REMOTE_DIR}/ffmpeg.log 2>&1 < /dev/null & echo started"
        )
        result = subprocess.run([*self._ssh, cmd], capture_output=True, text=True, timeout=20)
        self.started = "started" in result.stdout
        if self.started:
            logger.info("camera recording -> %s:%s", self._host, self._remote)
        else:
            logger.warning("camera recording did not start: %s", result.stderr.strip()[:120])

    def stop(self) -> Path | None:
        """Stop ffmpeg gracefully (SIGINT finalises the file) and copy the video to the laptop."""
        if not self.started:
            return None
        subprocess.run(
            [*self._ssh, "pkill -INT -f '[f]fmpeg .*-f v4l2'; sleep 1.5"],
            capture_output=True,
            timeout=20,
        )
        self._local.parent.mkdir(parents=True, exist_ok=True)
        fetched = subprocess.run(
            ["scp", "-q", f"root@{self._host}:{self._remote}", str(self._local)],
            capture_output=True,
            timeout=600,
        )
        if fetched.returncode != 0:
            logger.warning("could not fetch the video: %s", fetched.stderr.decode()[:120])
            return None
        logger.info("camera video saved to %s", self._local)
        return self._local
