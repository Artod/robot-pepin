#!/usr/bin/env python3
"""A drive's board-side camera clip turned into a camera bag: the replay path of the head.

The drives record no image topics (the bag would not fit the board's WiFi or its disk), but every
run leaves ``<run>_cam.mjpeg`` beside its bag: curl's raw copy of ustreamer's multipart stream
(``pepin_bringup.camera_clip``), which KEEPS every part's headers, and with ``?extra_headers=1``
those carry the V4L2 capture stamp. This tool reads the clip part by part, dates each frame by
its capture (:func:`pepin.mjpeg.capture_time` ``grab``: ``grab + (X-Timestamp - send)``), cuts
and rectifies it exactly as camera_stream does (:class:`pepin_bringup.stereo_frames.StereoFrames`)
and writes the four stereo topics into an MCAP bag beside the drive's:

    ros/clip_to_bag.sh 0512                        # the wrapper: the laptop image, no network
    python3 /repo/ros/tools/clip_to_bag.py /rec/0512_..._cam.mjpeg --out /rec/0512_..._cam.bag

then ``ros2 bag play --clock <drive>.bag <run>_cam.bag`` replays the camera beside the drive.

A part without the grab headers falls back to its send stamp and is COUNTED (``unstamped``);
``--require-grab`` refuses such a clip outright (a VIO or Kalibr replay on send stamps carries
1-68 ms of bimodal jitter no time offset can absorb, 2026-10-02).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

DEFAULT_CONFIG = "/repo/config/camera.json"


class BagWriter(Protocol):
    """Where the messages go: rosbag2 in the container, a list in the tests."""

    def write(self, topic: str, msg: Any, stamp_ns: int) -> None:
        """One message on ``topic`` with the bag's receive time ``stamp_ns``."""

    def close(self) -> None:
        """Finish the bag."""


@dataclass
class ClipReport:
    """What a conversion saw: frames written, parts without grab stamps, undecodable parts, and
    the send-grab lag of every part that carried both (milliseconds)."""

    frames: int = 0
    unstamped: int = 0
    undecodable: int = 0
    skipped: int = 0
    lags_ms: list[float] = field(default_factory=list)
    first_stamp: float | None = None
    last_stamp: float | None = None

    def text(self) -> str:
        """One line for the terminal."""
        lag = "no grab headers"
        if self.lags_ms:
            ordered = sorted(self.lags_ms)
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
            lag = f"send-grab median/p90 {statistics.median(ordered):.0f}/{p90:.0f} ms"
        span = (
            f", {self.last_stamp - self.first_stamp:.1f} s"
            if self.first_stamp is not None and self.last_stamp is not None
            else ""
        )
        return (
            f"{self.frames} frames{span}, {self.unstamped} unstamped (send fallback),"
            f" {self.undecodable} undecodable, {self.skipped} outside --start/--end; {lag}"
        )


class NoGrabStampsError(RuntimeError):
    """The clip carries no capture stamps and the caller required them."""


def convert(
    stream: IO[bytes],
    frames: Any,
    writer: BagWriter,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    require_grab: bool = False,
) -> ClipReport:
    """Every part of ``stream`` through ``frames`` (a StereoFrames) into ``writer``.

    ``start_s``/``end_s`` are seconds after the first part's stamp. Under ``require_grab`` the
    first part without the capture stamp raises :class:`NoGrabStampsError`."""
    import cv2
    import numpy as np
    from pepin_bringup.msgs import stamp_from_seconds

    from pepin.mjpeg import capture_time, has_grab, parts, send_lag_s

    report = ClipReport()
    for headers, body in parts(stream):
        stamp = capture_time(headers, "grab")
        if stamp is None:
            report.unstamped += 1
            if require_grab:
                raise NoGrabStampsError("a part without any time stamp")
            continue
        if not has_grab(headers):
            report.unstamped += 1
            if require_grab:
                raise NoGrabStampsError(
                    "the clip has no capture stamps (recorded without ?extra_headers=1?)"
                )
        lag = send_lag_s(headers)
        if lag is not None:
            report.lags_ms.append(lag * 1e3)
        if report.first_stamp is None:
            report.first_stamp = stamp
        offset = stamp - report.first_stamp
        if offset < start_s or (end_s is not None and offset > end_s):
            report.skipped += 1
            continue
        picture = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if picture is None:
            report.undecodable += 1
            continue
        stamp_ns = round(stamp * 1e9)
        for topic, msg in frames.messages(picture, stamp_from_seconds(stamp)):
            writer.write(topic, msg, stamp_ns)
        report.frames += 1
        report.last_stamp = stamp
    return report


class Rosbag2Writer:
    """An MCAP bag through rosbag2_py (the laptop image), the four stereo topics declared.

    Raw rectified eyes are 1.9 MB a frame, 19 MB/s at 10 Hz (a 2 s clip wrote 40.3 MiB); MCAP's
    ``zstd_fast`` chunk compression is on unless ``compress`` is False."""

    def __init__(self, out: Path, compress: bool = True) -> None:
        import rosbag2_py
        from pepin_bringup.stereo_frames import TOPIC_TYPES

        preset = "zstd_fast" if compress else ""
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(
                uri=str(out), storage_id="mcap", storage_preset_profile=preset
            ),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        # rosbag2 0.26 (this image's Jazzy) wants the topic id first; it numbers them itself.
        for index, (name, kind) in enumerate(TOPIC_TYPES.items()):
            self._writer.create_topic(
                rosbag2_py.TopicMetadata(id=index, name=name, type=kind, serialization_format="cdr")
            )

    def write(self, topic: str, msg: Any, stamp_ns: int) -> None:
        """One message, serialised."""
        from rclpy.serialization import serialize_message

        self._writer.write(topic, serialize_message(msg), stamp_ns)

    def close(self) -> None:
        """rosbag2 writes the summary when the writer goes."""
        del self._writer


def stereo_frames(config: Path, camera: str | None) -> Any:
    """The StereoFrames of the named (or active) rig from config/camera.json and its calibration."""
    from pepin_bringup.stereo_frames import StereoFrames

    from pepin.camera import CameraConfig
    from pepin.stereo import Rectifier, SideBySide, StereoCalibration

    cfg = CameraConfig.load(config, name=camera)
    rig = cfg.rig
    if rig is None:
        raise SystemExit(f"{cfg.name} is not a stereo rig: nothing to rectify")
    calibration_file = rig.calibration_path(config.parent)
    calibration = StereoCalibration.load(calibration_file)
    rectifier = Rectifier.from_calibration(calibration, mask_folds=True)
    source = f"{calibration.method} {calibration.date}, rms {calibration.rms_px:.2f} px"
    return StereoFrames(SideBySide(rig.upside_down), rectifier, cfg.optical_frame, source)


def default_out(clip: Path) -> Path:
    """``<run>_cam.mjpeg`` -> ``<run>_cam.bag`` beside it."""
    return clip.with_suffix(".bag")


def parse(argv: list[str]) -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("clip", type=Path, help="<run>_cam.mjpeg (the board-side clip)")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--camera", default=None, help="the rig by name (default: active)")
    parser.add_argument("--out", type=Path, default=None, help="default: <clip>.bag beside it")
    parser.add_argument("--start", type=float, default=0.0, help="seconds after the first frame")
    parser.add_argument("--end", type=float, default=None, help="seconds after the first frame")
    parser.add_argument("--require-grab", action="store_true", help="refuse send-stamped clips")
    parser.add_argument("--no-compress", action="store_true", help="MCAP without zstd chunks")
    return parser.parse_args(argv)


def first_part_has_grab(path: Path) -> bool:
    """Whether the clip's first part carries the capture stamp (checked before a bag exists)."""
    from pepin.mjpeg import has_grab, parts

    with path.open("rb") as stream:
        for headers, _body in parts(stream):
            return has_grab(headers)
    return False


def main(argv: list[str] | None = None) -> int:
    """Convert one clip; prints the report and the output path."""
    args = parse(sys.argv[1:] if argv is None else argv)
    out = args.out or default_out(args.clip)
    if out.exists():
        print(f"{out} exists: remove it first", file=sys.stderr)
        return 2
    if args.require_grab and not first_part_has_grab(args.clip):
        print("refused: the clip has no capture stamps (no ?extra_headers=1)", file=sys.stderr)
        return 3
    frames = stereo_frames(args.config, args.camera)
    writer = Rosbag2Writer(out, compress=not args.no_compress)
    try:
        with args.clip.open("rb") as stream:
            report = convert(
                stream,
                frames,
                writer,
                start_s=args.start,
                end_s=args.end,
                require_grab=args.require_grab,
            )
    except NoGrabStampsError as error:
        writer.close()
        print(f"refused mid-clip: {error}; {out} holds the frames before it", file=sys.stderr)
        return 3
    writer.close()
    print(f"{args.clip.name}: {report.text()}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
