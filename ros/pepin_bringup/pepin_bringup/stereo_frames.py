"""The stereo head's four messages from one side-by-side frame, built in one place.

Two writers need exactly the same messages: :mod:`pepin_bringup.camera_stream` live, and
``ros/tools/clip_to_bag.py`` turning a drive's board-side clip into a camera bag for the offline
replays. ROS stereo wants FOUR messages with ONE stamp and the LEFT eye's optical frame: the left
picture (bgr8) with the rectified pinhole (no distortion, R identity, P with Tx 0), and the right
picture (mono8 unless ``right_colour``) with the same pinhole and ``P[0, 3] = -fx * baseline`` —
which is how everything downstream reads the baseline off the wire rather than out of a config.
"""

from __future__ import annotations

from typing import Any

from sensor_msgs.msg import CameraInfo

from pepin.camera import Optics
from pepin.stereo import Rectifier, SideBySide
from pepin_bringup.msgs import image_from_array

LEFT_IMAGE_TOPIC = "/camera/image"
LEFT_INFO_TOPIC = "/camera/camera_info"
RIGHT_IMAGE_TOPIC = "/camera/right/image"
RIGHT_INFO_TOPIC = "/camera/right/camera_info"
STEREO_TOPICS = (LEFT_IMAGE_TOPIC, LEFT_INFO_TOPIC, RIGHT_IMAGE_TOPIC, RIGHT_INFO_TOPIC)
TOPIC_TYPES = {
    LEFT_IMAGE_TOPIC: "sensor_msgs/msg/Image",
    LEFT_INFO_TOPIC: "sensor_msgs/msg/CameraInfo",
    RIGHT_IMAGE_TOPIC: "sensor_msgs/msg/Image",
    RIGHT_INFO_TOPIC: "sensor_msgs/msg/CameraInfo",
}


def camera_info(size: tuple[int, int], lens: Optics, frame_id: str) -> Any:
    """One ``sensor_msgs/CameraInfo`` for a picture of ``size`` with these optics, in
    ``frame_id`` and with no stamp yet (the frame's own is put on at publish)."""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width, info.height = size
    info.distortion_model = "plumb_bob"
    info.k, info.d, info.r, info.p = lens.camera_info_arrays()
    return info


def rectified_optics(rectifier: Rectifier, source: str) -> Optics:
    """The ONE pinhole both eyes share after rectification, at the calibration's size."""
    return Optics(
        rectifier.fx,
        rectifier.fy,
        rectifier.cx,
        rectifier.cy,
        rectifier.width,
        rectifier.height,
        (),
        True,
        source,
    )


def right_camera_info(rectifier: Rectifier, lens: Optics, frame_id: str) -> Any:
    """The right eye's ``CameraInfo``: the shared pinhole with ``P[0, 3] = -fx * baseline``."""
    info = camera_info((rectifier.width, rectifier.height), lens, frame_id)
    info.p = list(info.p)
    info.p[3] = rectifier.right_projection_tx()
    return info


def eye_messages(
    left: Any,
    right: Any | None,
    left_info: Any,
    right_info: Any | None,
    stamp: Any,
    frame_id: str,
    right_colour: bool = False,
) -> list[tuple[str, Any]]:
    """The messages of one frame as ``(topic, msg)``, all under ``stamp``: the left pair always,
    the right pair only with both a right picture and its info (no baseline, no right eye)."""
    left_info.header.stamp = stamp
    out = [
        (LEFT_IMAGE_TOPIC, image_from_array(left, "bgr8", stamp, frame_id)),
        (LEFT_INFO_TOPIC, left_info),
    ]
    if right is not None and right_info is not None:
        right_info.header.stamp = stamp
        encoding = "bgr8" if right_colour else "mono8"
        out += [
            (RIGHT_IMAGE_TOPIC, image_from_array(right, encoding, stamp, frame_id)),
            (RIGHT_INFO_TOPIC, right_info),
        ]
    return out


class StereoFrames:
    """One side-by-side frame in, the four rectified messages out (the offline writer's path;
    camera_stream times the same steps one by one into its tally)."""

    def __init__(
        self,
        split: SideBySide,
        rectifier: Rectifier,
        frame_id: str,
        source: str = "stereo calibration",
        right_colour: bool = False,
    ) -> None:
        self.split = split
        self.rectifier = rectifier
        self.frame_id = frame_id
        self.right_colour = right_colour
        self.optics = rectified_optics(rectifier, source)
        size = (rectifier.width, rectifier.height)
        self.left_info = camera_info(size, self.optics, frame_id)
        self.right_info = right_camera_info(rectifier, self.optics, frame_id)

    def eyes(self, frame: Any) -> tuple[Any, Any]:
        """The two rectified eyes of a side-by-side frame (the right one grey unless colour)."""
        import cv2

        left, right = self.split.eyes(frame)
        if not self.right_colour:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return self.rectifier.rectify(left, right)

    def messages(self, frame: Any, stamp: Any) -> list[tuple[str, Any]]:
        """The four messages of one side-by-side frame under ``stamp``; the infos are fresh
        per frame, so a writer that keeps the messages keeps each frame's own stamp."""
        left, right = self.eyes(frame)
        size = (self.rectifier.width, self.rectifier.height)
        left_info = camera_info(size, self.optics, self.frame_id)
        right_info = right_camera_info(self.rectifier, self.optics, self.frame_id)
        return eye_messages(
            left, right, left_info, right_info, stamp, self.frame_id, self.right_colour
        )


__all__ = [
    "LEFT_IMAGE_TOPIC",
    "LEFT_INFO_TOPIC",
    "RIGHT_IMAGE_TOPIC",
    "RIGHT_INFO_TOPIC",
    "STEREO_TOPICS",
    "TOPIC_TYPES",
    "StereoFrames",
    "camera_info",
    "eye_messages",
    "rectified_optics",
    "right_camera_info",
]
