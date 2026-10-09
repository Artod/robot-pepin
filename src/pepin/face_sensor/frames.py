"""Where the face sensor's frames come from: the board's MJPEG stream, a recorded clip, a folder of
stills, or a fake.

A :class:`Frame` carries its stamp and decodes on demand, so a sensor running at 3 fps on a
20 fps stream decodes only the frames it looks at. A ``live`` source delivers in real time and the
runner takes the newest frame each turn (:class:`Newest`: frames that arrive while the pipeline is
busy are dropped, never queued); a recording is sampled by its own stamps instead, so a clip gives
the same frames however fast the laptop is.

The head camera's frame holds both eyes side by side (1600x600, the module mounted upside down);
:func:`cut_eye` keeps one eye with :class:`pepin.stereo.SideBySide`, and leaves a frame that is
not side by side (a mono camera, a still) as it is.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.mjpeg import capture_time, parts

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]
STILL_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MJPEG_SUFFIXES = (".mjpeg", ".mjpg")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
STREAM_TIMEOUT_S = 5.0
RETRY_S = 2.0


@dataclass(frozen=True)
class Frame:
    """One picture of one eye: its capture stamp (seconds), its index in the source, and the
    decoder that produces the BGR image when someone looks."""

    stamp: float
    index: int
    load: Callable[[], Image]

    def image(self) -> Image:
        """The decoded BGR image (decoded again on every call)."""
        return self.load()


class FrameSource(Protocol):
    """A stream of frames; ``live`` says they arrive in real time (take the newest)."""

    live: bool

    def frames(self) -> Iterator[Frame]:
        """Frames in order until the source ends or is closed."""
        ...

    def close(self) -> None:
        """Stop delivering; a blocked read ends."""
        ...


def cut_eye(image: Image, eye: str, upside_down: bool) -> Image:
    """One eye of a side-by-side frame (width at least twice the height), upright; ``full`` or a
    frame that is not side by side comes back as it is."""
    if eye == "full" or image.shape[1] < 2 * image.shape[0]:
        return image
    from pepin.stereo import SideBySide

    left, right = SideBySide(upside_down).eyes(image)
    return left if eye == "left" else right


def decode_jpeg(body: bytes, eye: str, upside_down: bool) -> Image:
    """A JPEG part decoded to BGR and cut to one eye; ``ValueError`` when it does not decode."""
    import cv2

    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("a JPEG part did not decode")
    return cut_eye(np.asarray(image, dtype=np.uint8), eye, upside_down)


def _stamp(headers: dict[str, str], mode: str) -> float:
    """A part's capture stamp by ``mode`` (``grab``, ``send``), else the laptop's clock now."""
    if mode in ("grab", "send"):
        stamp = capture_time(headers, mode)
        if stamp is not None:
            return stamp
    return time.time()


class BoardStream:
    """The board's MJPEG stream (ustreamer), reconnecting after a loss; ``opener`` is
    replaceable so a test can feed a scripted stream."""

    live = True

    def __init__(
        self,
        url: str,
        *,
        eye: str = "left",
        upside_down: bool = False,
        stamp: str = "grab",
        opener: Callable[[str, float], IO[bytes]] | None = None,
        retry_s: float = RETRY_S,
    ) -> None:
        """Prepare a reader of ``url``; nothing connects until :meth:`frames` is iterated."""
        self.url = url
        self._eye = eye
        self._upside_down = upside_down
        self._stamp = stamp
        self._opener = opener or _urlopen
        self._retry_s = retry_s
        self._closed = threading.Event()
        self._stream: IO[bytes] | None = None

    def frames(self) -> Iterator[Frame]:
        """Every part of the stream as a lazily decoded frame, across reconnections."""
        index = 0
        while not self._closed.is_set():
            try:
                with self._opener(self.url, STREAM_TIMEOUT_S) as stream:
                    self._stream = stream
                    for headers, body in parts(stream):
                        if self._closed.is_set():
                            return
                        yield Frame(_stamp(headers, self._stamp), index, self._decoder(body))
                        index += 1
                logger.warning("camera stream ended; reconnecting")
            except Exception as error:
                # Whatever a dropped WiFi or a restarting board throws (a reset socket, a
                # truncated part, a response closed under the read by close()) costs one
                # reconnection, never the sensor.
                if self._closed.is_set():
                    return
                logger.warning("camera stream not reachable (%s); retrying", error)
            finally:
                self._stream = None
            self._closed.wait(self._retry_s)

    def close(self) -> None:
        """End the iteration; the socket under a blocked read is closed."""
        self._closed.set()
        stream = self._stream
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()

    def _decoder(self, body: bytes) -> Callable[[], Image]:
        return lambda: decode_jpeg(body, self._eye, self._upside_down)


def _urlopen(url: str, timeout: float) -> IO[bytes]:
    response: IO[bytes] = urllib.request.urlopen(url, timeout=timeout)
    return response


class MjpegFile:
    """A recorded clip of the board's stream (``<run>_cam.mjpeg``, curl's copy of the parts with
    their headers), stamped like the live stream."""

    live = False

    def __init__(
        self, path: str | Path, *, eye: str = "left", upside_down: bool = False, stamp: str = "grab"
    ) -> None:
        """Prepare a reader of the clip at ``path``."""
        self.path = Path(path)
        self._eye = eye
        self._upside_down = upside_down
        self._stamp = stamp
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        """Every part of the clip; a part without a stamp is placed 50 ms after the last one."""
        last = 0.0
        with self.path.open("rb") as stream:
            for index, (headers, body) in enumerate(parts(stream)):
                if self._closed:
                    return
                stamp = capture_time(headers, self._stamp) if self._stamp != "arrival" else None
                last = stamp if stamp is not None else last + 0.05
                yield Frame(last, index, self._decoder(body))

    def close(self) -> None:
        """Stop at the next part."""
        self._closed = True

    def _decoder(self, body: bytes) -> Callable[[], Image]:
        return lambda: decode_jpeg(body, self._eye, self._upside_down)


class VideoFile:
    """A video file read with OpenCV, stamped by its own presentation times; decoded as read."""

    live = False

    def __init__(self, path: str | Path, *, eye: str = "left", upside_down: bool = False) -> None:
        """Prepare a reader of the video at ``path``."""
        self.path = Path(path)
        self._eye = eye
        self._upside_down = upside_down
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        """Every frame of the video in order."""
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise ValueError(f"{self.path}: OpenCV cannot open this video")
        try:
            index = 0
            while not self._closed:
                ok, image = capture.read()
                if not ok:
                    return
                stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1e3
                eye = cut_eye(np.asarray(image, dtype=np.uint8), self._eye, self._upside_down)
                yield Frame(stamp, index, _const(eye))
                index += 1
        finally:
            capture.release()

    def close(self) -> None:
        """Stop at the next frame."""
        self._closed = True


class Stills:
    """Still images (a folder in name order, or a list of files), each a frame ``1 / fps`` apart
    on a clock that starts at ``start``; ``repeat`` offers every image that many times in a row."""

    live = False

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        fps: float = 3.0,
        repeat: int = 1,
        start: float = 0.0,
        eye: str = "full",
        upside_down: bool = False,
    ) -> None:
        """Prepare the images; a folder among ``paths`` stands for its images in name order."""
        self.paths = list(still_paths(paths))
        if not self.paths:
            raise ValueError(f"no images in {[str(p) for p in paths]}")
        self._period = 1.0 / fps
        self._repeat = repeat
        self._start = start
        self._eye = eye
        self._upside_down = upside_down
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        """Each image ``repeat`` times, one period apart."""
        index = 0
        for path in self.paths:
            for _ in range(self._repeat):
                if self._closed:
                    return
                stamp = self._start + index * self._period
                yield Frame(stamp, index, self._reader(path))
                index += 1

    def close(self) -> None:
        """Stop at the next image."""
        self._closed = True

    def _reader(self, path: Path) -> Callable[[], Image]:
        def read() -> Image:
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"{path}: not an image OpenCV can read")
            return cut_eye(np.asarray(image, dtype=np.uint8), self._eye, self._upside_down)

        return read


class FakeFrames:
    """Given images at given stamps (default 1 / 3 s apart): the stand-in source of the tests."""

    def __init__(
        self, images: Sequence[Image], stamps: Sequence[float] | None = None, *, live: bool = False
    ) -> None:
        """Hold ``images``; ``stamps`` must match their count when given."""
        if stamps is not None and len(stamps) != len(images):
            raise ValueError("one stamp per image")
        self.images = list(images)
        self.stamps = list(stamps) if stamps is not None else [i / 3.0 for i in range(len(images))]
        self.live = live
        self.decoded = 0
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        """The images in order."""
        for index, (image, stamp) in enumerate(zip(self.images, self.stamps, strict=True)):
            if self._closed:
                return
            yield Frame(stamp, index, self._counted(image))

    def close(self) -> None:
        """Stop at the next image."""
        self._closed = True

    def _counted(self, image: Image) -> Callable[[], Image]:
        def load() -> Image:
            self.decoded += 1
            return image

        return load


def _const(image: Image) -> Callable[[], Image]:
    return lambda: image


def still_paths(paths: Sequence[str | Path]) -> Iterator[Path]:
    """The image files among ``paths``, a folder expanded to its images in name order."""
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in STILL_SUFFIXES)
        else:
            yield path


def open_source(
    spec: str,
    *,
    eye: str,
    upside_down: bool,
    stamp: str,
    still_fps: float,
    stream_url: str | None = None,
) -> FrameSource:
    """A source from a command-line word: ``board`` (the stream at ``stream_url``), an http URL,
    an ``.mjpeg`` clip, a video file, an image, or a folder of images."""
    if spec == "board":
        if not stream_url:
            raise ValueError("the board's stream URL is unknown (config/camera.json)")
        return BoardStream(stream_url, eye=eye, upside_down=upside_down, stamp=stamp)
    if spec.startswith(("http://", "https://")):
        return BoardStream(spec, eye=eye, upside_down=upside_down, stamp=stamp)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(spec)
    suffix = path.suffix.lower()
    if suffix in MJPEG_SUFFIXES:
        return MjpegFile(path, eye=eye, upside_down=upside_down, stamp=stamp)
    if suffix in VIDEO_SUFFIXES:
        return VideoFile(path, eye=eye, upside_down=upside_down)
    return Stills([path], fps=still_fps, eye=eye, upside_down=upside_down)


class Newest:
    """The newest frame of a live source, read in a daemon thread: :meth:`take` hands out each
    frame at most once, and frames nobody took in time are counted as skipped."""

    def __init__(self, source: FrameSource) -> None:
        """Start reading ``source`` in the background."""
        self._source = source
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._taken = -1
        self.skipped = 0
        self.ended = False
        self._thread = threading.Thread(target=self._run, name="face-frames", daemon=True)
        self._thread.start()

    def take(self, timeout: float) -> Frame | None:
        """The newest frame not handed out yet, waiting up to ``timeout`` s for one."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or self._frame.index == self._taken:
                remaining = deadline - time.monotonic()
                if self.ended or remaining <= 0:
                    return None
                self._cond.wait(remaining)
            frame = self._frame
            self._taken = frame.index
            return frame

    def close(self) -> None:
        """Close the source; the reader thread ends with it."""
        self._source.close()
        self._thread.join(timeout=STREAM_TIMEOUT_S)

    def _run(self) -> None:
        try:
            for frame in self._source.frames():
                with self._cond:
                    previous = self._frame
                    if previous is not None and previous.index != self._taken:
                        self.skipped += 1
                    self._frame = frame
                    self._cond.notify_all()
        except Exception:
            logger.exception("frame source failed")
        finally:
            with self._cond:
                self.ended = True
                self._cond.notify_all()


def sample(frames: Iterator[Frame], rate_hz: float) -> Iterator[Frame]:
    """A recording's frames thinned to ``rate_hz`` by their own stamps: a frame is kept when it is
    at least one period after the last kept one (a 20 fps clip at 3 fps keeps every 7th)."""
    period = 1.0 / rate_hz
    next_at: float | None = None
    for frame in frames:
        if next_at is not None and frame.stamp < next_at - 1e-6:
            continue
        next_at = frame.stamp + period
        yield frame


def describe(source: FrameSource) -> str:
    """One word on where the frames come from, for the report line."""
    path: Any = getattr(source, "path", None) or getattr(source, "url", None)
    return str(path) if path is not None else type(source).__name__
