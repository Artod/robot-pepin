"""Live settings between two processes of one container: a small JSON file one writes and the
other reads on every call.

WHY A FILE. RTAB-Map's Python adapters (ros/xfeat) run inside RTAB-Map's own process, called from
its C++ for every detection and every match, with no ROS node of their own to hold a parameter —
and a switch that needs RTAB-Map rebuilt or restarted to move is exactly what the operator must
never wait for. The node that owns the switch (pepin_bringup.rtabmap_frame, rule 19: a live flag
in its FLAGS table, in its report line) runs in the SAME container, so the cheapest channel
between the two is the container's own filesystem: the owner writes :data:`REGISTRATION_FILE`
atomically (a temporary file renamed over it) whenever a flag moves, and a reader pays one
``stat`` per call and parses the file only when its mtime changed. No ROS in RTAB-Map's
interpreter, no thread, no socket. A missing or broken file reads as the defaults — the adapters
work with rtabmap_frame down, as they did before it existed.

THE OTHER WAY, the same mechanism: the adapters write their counters to :data:`REGISTRATION_STATUS`
once a minute, and rtabmap_frame prints them in its report line, so what the switch is doing is
seen where the switch is.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Both ends are in pepin-vslam, so a path in the container's own /tmp needs no mount; the
# environment moves it (a test, a second RTAB-Map in a throwaway container), read when a reader or
# a writer is made (:func:`registration_file`, :func:`registration_status`).
REGISTRATION_FILE = "/tmp/pepin/registration.json"
REGISTRATION_STATUS = "/tmp/pepin/registration_status.json"
BACKENDS = ("service", "local", "auto")


def registration_file() -> Path:
    """Where the registration settings live: ``PEPIN_REGISTRATION_FILE``, else
    :data:`REGISTRATION_FILE`."""
    return Path(os.environ.get("PEPIN_REGISTRATION_FILE") or REGISTRATION_FILE)


def registration_status() -> Path:
    """Where the adapters' counters live: ``PEPIN_REGISTRATION_STATUS``, else
    :data:`REGISTRATION_STATUS`."""
    return Path(os.environ.get("PEPIN_REGISTRATION_STATUS") or REGISTRATION_STATUS)


class LiveFile:
    """A JSON object on disk: :meth:`read` re-parses it only when the file changed (its mtime,
    size or inode), :meth:`write` replaces it atomically, :meth:`write_if_changed` only when the
    values differ."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._seen: tuple[int, int, int] | None = None
        self._values: dict[str, Any] = {}
        self._written: dict[str, Any] | None = None

    def read(self) -> dict[str, Any]:
        """The file's object; ``{}`` when it is missing, unreadable or not an object."""
        try:
            st = os.stat(self.path)
        except OSError:
            self._seen, self._values = None, {}
            return {}
        seen = (st.st_mtime_ns, st.st_size, st.st_ino)  # a rename is a new inode
        if seen != self._seen:
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                data = {}
            self._values = dict(data) if isinstance(data, dict) else {}
            self._seen = seen
        return dict(self._values)

    def write(self, values: Mapping[str, Any]) -> None:
        """Replace the file with ``values`` in one rename, creating its directory."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(dict(values), f, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._written = dict(values)

    def write_if_changed(self, values: Mapping[str, Any]) -> bool:
        """:meth:`write` when ``values`` differ from the last written (or the file is gone);
        whether it wrote."""
        if self._written == dict(values) and self.path.exists():
            return False
        self.write(values)
        return True


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class RegistrationSettings:
    """What RTAB-Map's XFeat and LighterGlue adapters do on each call: where they compute
    (``service``, ``local`` or ``auto``), how long they wait for the service, and XFeat's
    keypoint cap. The defaults are the environment's (``PEPIN_REGISTRATION_BACKEND``,
    ``PEPIN_REGISTRATION_TIMEOUT_S``, ``PEPIN_XFEAT_TOP_K``), else auto, 0.5 s and 2048."""

    backend: str = "auto"
    timeout_s: float = 0.5
    top_k: int = 2048

    @classmethod
    def defaults(cls) -> RegistrationSettings:
        """The environment's values over the class defaults."""
        backend = os.environ.get("PEPIN_REGISTRATION_BACKEND", cls.backend)
        return cls(
            backend=backend if backend in BACKENDS else cls.backend,
            timeout_s=_env_float("PEPIN_REGISTRATION_TIMEOUT_S", cls.timeout_s),
            top_k=_env_int("PEPIN_XFEAT_TOP_K", cls.top_k),
        )

    @classmethod
    def parse(cls, values: Mapping[str, Any], base: RegistrationSettings) -> RegistrationSettings:
        """``values`` (the file's object) over ``base``; a value of the wrong kind or out of
        bounds keeps ``base``'s."""
        backend = values.get("backend", base.backend)
        try:
            timeout = float(values.get("timeout_s", base.timeout_s))
        except (TypeError, ValueError):
            timeout = base.timeout_s
        try:
            top_k = int(values.get("top_k", base.top_k))
        except (TypeError, ValueError):
            top_k = base.top_k
        return cls(
            backend=backend if backend in BACKENDS else base.backend,
            timeout_s=timeout if 0.0 < timeout <= 30.0 else base.timeout_s,
            top_k=top_k if 1 <= top_k <= 16384 else base.top_k,
        )

    def to_json(self) -> dict[str, Any]:
        """The object the owner writes."""
        return asdict(self)


class RegistrationSwitch:
    """The adapters' side: the settings in force now, read from :data:`REGISTRATION_FILE` on
    every call (one ``stat``), the environment's defaults under whatever the file does not say."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._file = LiveFile(path or registration_file())
        self._base = RegistrationSettings.defaults()

    def settings(self) -> RegistrationSettings:
        """The settings for this call."""
        return RegistrationSettings.parse(self._file.read(), self._base)


class StatusBoard:
    """One process's counters in a shared file, several writers (the detector and the matcher)
    each under its own key: ``publish(name, fields)`` merges this writer's block in."""

    def __init__(
        self, path: str | Path | None = None, clock: Callable[[], float] = time.time
    ) -> None:
        self._file = LiveFile(path or registration_status())
        self._clock = clock

    def publish(self, name: str, fields: Mapping[str, Any]) -> None:
        """Write this writer's block, stamped with the wall clock, beside the others'; a failure
        to write is swallowed (the counters are for people, the registration must go on)."""
        try:
            blocks = self._file.read()
            blocks[name] = {**dict(fields), "at": round(self._clock(), 1)}
            self._file.write(blocks)
        except OSError:
            pass

    def read(self) -> dict[str, Any]:
        """Every writer's last block."""
        return self._file.read()
