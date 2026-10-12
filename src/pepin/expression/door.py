"""The arbiter's door as the speaker gaze uses it (:class:`pepin.expression.speaker.GazeDoor`).

Every call is one HTTP exchange on :mod:`pepin.gaze_link` from the speaker gaze's thread. An aim
is answered when the head has settled there; one queued behind a better look (a ``look`` the
person asked for) would only be answered when that look ends, so the wait is cut at
``aim_wait_s`` and the aim is taken as queued: the request lives on in the arbiter, renewed like
any. A door that does not answer is a refusal in words, never an exception.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pepin.expression.head import BAND, SPEAKER
from pepin.expression.speaker import GazeAnswer, HeadView
from pepin.gaze import HOME, Aim, Speed
from pepin.gaze_link import GAZE_URL, GazeClient

AWAY = "the arbiter's door does not answer"


class SpeakerDoor:
    """:class:`pepin.expression.speaker.GazeDoor` for one source (``voice.speaker``) over the
    door's JSON, in the expression band."""

    def __init__(
        self,
        url: str = GAZE_URL,
        *,
        source: str = SPEAKER,
        timeout_s: float = 2.0,
        aim_wait_s: float = 2.0,
        client: GazeClient | None = None,
    ) -> None:
        """The door at ``url``; ``aim_wait_s``: how long an aim waits for its answer."""
        self.client = client if client is not None else GazeClient(source, url, timeout_s)
        self.aim_wait_s = aim_wait_s
        self.calls = 0  # every exchange, for the report line's calls per minute

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float) -> GazeAnswer:
        """Hold the head at ``aim`` (an ``angles`` look held for its TTL, no depth frames)."""
        request = {
            "kind": "angles",
            "band": BAND,
            "target": {"pan_rad": aim.pan_rad, "tilt_rad": aim.tilt_rad},
            "frames": 0,
            "hold": True,
            "ttl_s": ttl_s,
            "speed": speed,
        }
        self.calls += 1
        try:
            return answer_of(self.client.look(request, self.aim_wait_s))
        except TimeoutError:
            return GazeAnswer("queued", f"no answer in {self.aim_wait_s:.1f} s: held behind")
        except (OSError, ValueError) as exc:
            return GazeAnswer("denied", f"{AWAY}: {exc}")

    def renew(self) -> int:
        """Restart this source's TTL; how many of its requests live (0 when the door is away)."""
        self.calls += 1
        try:
            return self.client.renew()
        except (OSError, ValueError):
            return 0

    def release(self) -> None:
        """Let the head go now (home, unless another request holds it)."""
        self.calls += 1
        try:
            self.client.release()
        except (OSError, ValueError):
            return

    def state(self) -> HeadView | None:
        """The arbiter's state; None when the door does not answer."""
        self.calls += 1
        try:
            return view_of(self.client.state())
        except (OSError, ValueError):
            return None


def answer_of(data: Mapping[str, Any]) -> GazeAnswer:
    """The door's JSON answer to an aim."""
    pan, tilt = data.get("pan_rad"), data.get("tilt_rad")
    head = Aim(float(pan), float(tilt)) if pan is not None and tilt is not None else None
    return GazeAnswer(str(data.get("status", "")), str(data.get("reason", "")), head)


def view_of(data: Mapping[str, Any]) -> HeadView:
    """``GET /state`` as the speaker gaze reads it."""
    pan, tilt = _number(data.get("pan_rad")), _number(data.get("tilt_rad"))
    head = Aim(pan, tilt) if pan is not None and tilt is not None else None
    until = data.get("blind_until")
    overlay = data.get("overlay")
    return HeadView(
        head=head,
        holder=str(data.get("source", "")),
        band=int(data.get("band", HOME)),
        blind=bool(data.get("blind", False)),
        blind_until=float(until) if isinstance(until, (int, float)) else None,
        driving=bool(data.get("driving", False)),
        overlay=str(overlay.get("source", "")) if isinstance(overlay, dict) else "",
    )


def _number(value: Any) -> float | None:
    """A finite JSON number, None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None
