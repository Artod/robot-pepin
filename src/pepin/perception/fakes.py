"""A scripted perception service for the tests: records, frames and embeddings, no sockets.

:class:`FakePerception` answers the :class:`pepin.perception.client.Perception` calls from what a
test scripted; a call that would wait lets its ``timeout_s`` pass on the test's clock (a callable
with a ``sleep``, as :class:`pepin.tools.fakes.FakeClock`), so a long poll that finds nothing
costs simulated time, never real time. ``down`` makes every call answer None, as the client
does while the service is not there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pepin.face_sensor.models import Detection
from pepin.perception import URL
from pepin.perception.client import Jpeg, Vector
from pepin.perception.record import Record


def unit_vector(seed: int, size: int = 512) -> Vector:
    """A repeatable unit vector for an embedding."""
    v = np.random.default_rng(seed).standard_normal(size).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v), dtype=np.float32)


@dataclass
class FakePerception:
    """:class:`pepin.perception.client.Perception` from a script (see the module)."""

    records: list[Record] = field(default_factory=list)
    frames: list[Jpeg] = field(default_factory=list)
    clock: Any = None  # a callable with sleep(seconds), or None: waits cost nothing
    embedding: Callable[[int, Detection], Vector | None] = lambda seq, det: unit_vector(seq)
    health_answer: dict[str, Any] = field(default_factory=dict)
    down: bool = False
    url: str = URL
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def latest(
        self, since: int = 0, min_stamp: float | None = None, timeout_s: float = 0.0
    ) -> Record | None:
        """The newest scripted record past ``since`` and ``min_stamp``; else the wait passes."""
        self.calls.append(("latest", {"since": since, "min_stamp": min_stamp,
                                      "timeout_s": timeout_s}))  # fmt: skip
        if self.down:
            return None
        found = [
            r for r in self.records if r.seq > since and (min_stamp is None or r.stamp >= min_stamp)
        ]
        if found:
            return max(found, key=lambda r: r.seq)
        self._wait(timeout_s)
        return None

    def frame(self, seq: int) -> Jpeg | None:
        """Frame ``seq`` when scripted."""
        self.calls.append(("frame", {"seq": seq}))
        if self.down:
            return None
        return next((f for f in self.frames if f.seq == seq), None)

    def still(self, min_stamp: float | None = None, timeout_s: float = 0.0) -> Jpeg | None:
        """The newest scripted frame at ``min_stamp`` or later; else, after the wait, the
        newest one marked not still."""
        self.calls.append(("still", {"min_stamp": min_stamp, "timeout_s": timeout_s}))
        if self.down or not self.frames:
            return None
        newest = max(self.frames, key=lambda f: f.seq)
        if min_stamp is None or newest.stamp >= min_stamp:
            return newest
        self._wait(timeout_s)
        return replace(newest, still=False)

    def embed(self, seq: int, detection: Detection) -> Vector | None:
        """The scripted embedding of a face in a frame the service holds."""
        self.calls.append(("embed", {"seq": seq, "box": detection.box}))
        if self.down:
            return None
        if not any(r.seq == seq for r in self.records) and not any(
            f.seq == seq for f in self.frames
        ):
            return None
        return self.embedding(seq, detection)

    def health(self) -> dict[str, Any] | None:
        """The scripted ``/health`` answer."""
        return None if self.down else dict(self.health_answer)

    def _wait(self, seconds: float) -> None:
        sleep = getattr(self.clock, "sleep", None)
        if callable(sleep) and seconds > 0:
            sleep(seconds)
