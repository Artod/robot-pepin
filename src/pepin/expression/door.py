"""The arbiter's door as expression uses it: the speaker gaze's (:class:`SpeakerDoor`, its
:class:`pepin.expression.speaker.GazeDoor`) and the head gestures' (:class:`OverlayDoor`, the
:class:`pepin.expression.parts.HeadDoor`).

Every call is one HTTP exchange on :mod:`pepin.gaze_link` from its owner's worker thread. An aim
is answered when the head has settled there; one queued behind a better look (a ``look`` the
person asked for) would only be answered when that look ends, so the wait is cut at
``aim_wait_s`` and the aim is taken as queued: the request lives on in the arbiter, renewed like
any. A finite overlay is answered when it ends and a refusal at once, so its wait is cut at
``answer_wait_s``: no answer by then means it plays. A door that does not answer is a refusal in
words, never an exception.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pepin.expression.head import BAND, SPEAKER, ClipLook
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


class OverlayClient(Protocol):
    """One source's overlay calls (:class:`pepin.gaze_link.GazeClient`'s)."""

    def clip(
        self,
        keys: Sequence[Mapping[str, float]],
        *,
        ttl_s: float,
        band: int = ...,
        loop: bool = ...,
        idle: bool = ...,
        timeout_s: float | None = ...,
    ) -> dict[str, Any]:
        """``POST /look`` of a clip; the answer, ``OSError`` without a door."""
        ...

    def release(self, fade_s: float = ...) -> int:
        """``POST /release``; how many went."""
        ...


class OverlayDoor:
    """:class:`pepin.expression.parts.HeadDoor` over the door's JSON: one client per source (a
    gesture's, the idle layer's), each overlay in the expression band."""

    def __init__(
        self,
        url: str = GAZE_URL,
        *,
        timeout_s: float = 1.0,
        answer_wait_s: float = 0.2,
        client: Callable[[str], OverlayClient] | None = None,
    ) -> None:
        """The door at ``url``; ``answer_wait_s``: how long a look waits for a refusal (one
        comes in milliseconds; a playing overlay answers only when it ends)."""
        self._client = client or (lambda source: GazeClient(source, url, timeout_s))
        self._clients: dict[str, OverlayClient] = {}
        self.answer_wait_s = answer_wait_s
        self.calls = 0

    def look(self, request: ClipLook) -> str | None:
        """Start the overlay: None when it plays, the refusal in words otherwise."""
        target = request["target"]
        keys = [
            {"at_s": k["at_s"], "pan_rad": k["pan_rad"], "tilt_rad": k["tilt_rad"],
             "speed_deg_s": k["speed_deg_s"]}
            for k in target["keys"]
        ]  # fmt: skip
        self.calls += 1
        try:
            answer = self.client(request["source"]).clip(
                keys,
                ttl_s=request["ttl_s"],
                band=request["band"],
                loop=target.get("loop", False),
                idle=target.get("idle", False),
                timeout_s=self.answer_wait_s,
            )
        except TimeoutError:
            return None  # no refusal in time: it plays, and is answered when it ends
        except (OSError, ValueError) as exc:
            return f"{AWAY}: {exc}"
        if answer.get("status") == "denied":
            return str(answer.get("reason") or "denied")
        return None

    def withdraw(self, source: str, fade_s: float) -> None:
        """``source``'s overlay released over ``fade_s`` (nothing when the door is away)."""
        self.calls += 1
        try:
            self.client(source).release(fade_s)
        except (OSError, ValueError):
            return

    def client(self, source: str) -> OverlayClient:
        """``source``'s client, made the first time."""
        if source not in self._clients:
            self._clients[source] = self._client(source)
        return self._clients[source]


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
