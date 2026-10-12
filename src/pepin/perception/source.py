"""Where the voice process's pictures come from, switched live: ``vision.source``.

``service`` (the default): the perception service, fed by camera_stream; the voice process holds
no pixels and runs no network. ``snapshot``: the path before it, for a day the service or the
camera half is broken: the gaze's finder grabs ustreamer's ``/snapshot`` and runs SCRFD and
Vision in the voice process, ``see`` takes its own snapshot, person recognition embeds locally.
A switch takes the next look and the next ``see``; nothing restarts (``scripts/voice_cascade.py
switch --knob vision.source=snapshot``, answered in ``control.jsonl`` like a flag).

The start value is config/models.json's ``perception.source``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

SERVICE, SNAPSHOT = "service", "snapshot"
SOURCES = (SERVICE, SNAPSHOT)
KNOB = "vision.source"
CONFIG_FILE = "models.json"


class VisionSource:
    """The live knob (see the module); thread-safe reads."""

    def __init__(self, value: str = SERVICE) -> None:
        if value not in SOURCES:
            raise ValueError(f"{KNOB} is one of {', '.join(SOURCES)}, not {value!r}")
        self._value = value
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str | Path | None = None) -> VisionSource:
        """The start value from config/models.json's ``perception.source`` (``service`` when
        the file or the key is missing)."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        try:
            block = json.loads(Path(path).read_text()).get("perception", {})
        except (OSError, ValueError):
            return cls()
        value = block.get("source", SERVICE) if isinstance(block, dict) else SERVICE
        return cls(str(value))

    @property
    def knob(self) -> str:
        """Its name in ``control.json``'s ``knobs``."""
        return KNOB

    @property
    def choices(self) -> tuple[str, ...]:
        """The values it takes."""
        return SOURCES

    @property
    def value(self) -> str:
        """``service`` or ``snapshot``, now."""
        with self._lock:
            return self._value

    @property
    def service(self) -> bool:
        """Whether the perception service is the source now."""
        return self.value == SERVICE

    def set(self, value: str) -> str | None:
        """Switch to ``value``; None when applied, else why not."""
        if value not in SOURCES:
            return f"{KNOB} is one of {', '.join(SOURCES)}, not {value!r}"
        with self._lock:
            self._value = value
        return None
