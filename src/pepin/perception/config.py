"""config/models.json's ``perception`` block: the service's port and pace, and its consumers'
waits. The detector's weights and threshold stay config/face_sensor.json's (``models``) and the
identity net's file config/people.json's (``face.model``): one place each, read by every path
that loads them, so the service and the snapshot fallback run the same networks."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pepin.perception import PORT
from pepin.perception.source import SERVICE, SOURCES

FACE_DEVICES = ("ane", "cpu", "coreml", "mps", "none")
PERSON_DETECTORS = ("vision", "none")


@dataclass(frozen=True)
class PerceptionConfig:
    """The ``perception`` block (see config/models.json's note for each number's reason)."""

    port: int = PORT
    source: str = SERVICE
    detect_hz: float = 10.0
    ring_s: float = 10.0
    faces: str = "ane"
    persons: str = "vision"
    body_head_frac: float = 0.12
    min_score: float = 0.0
    identity_device: str = "ane"
    identity_unload_after_s: float = 300.0
    look_wait_s: float = 0.4
    see_wait_s: float = 0.3

    @property
    def url(self) -> str:
        """The service as this Mac's processes reach it."""
        return f"http://127.0.0.1:{self.port}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerceptionConfig:
        """From the block's object (notes ignored); ``ValueError`` for a value no run can use."""
        defaults = cls()
        values = {
            f.name: type(getattr(defaults, f.name))(data[f.name])
            for f in fields(cls)
            if f.name in data
        }
        config = cls(**values)
        config.check()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> PerceptionConfig:
        """config/models.json's block (the defaults when the file has none)."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file("models.json")
        block = json.loads(Path(path).read_text()).get("perception", {})
        return cls.from_dict(block if isinstance(block, dict) else {})

    def check(self) -> None:
        """``ValueError`` on a value no run could use."""
        if self.source not in SOURCES:
            raise ValueError(f"perception.source is one of {SOURCES}, not {self.source!r}")
        if self.faces not in FACE_DEVICES:
            raise ValueError(f"perception.faces is one of {FACE_DEVICES}, not {self.faces!r}")
        if self.persons not in PERSON_DETECTORS:
            raise ValueError(f"perception.persons is one of {PERSON_DETECTORS}")
        if self.identity_device not in FACE_DEVICES[:-1]:
            raise ValueError(f"perception.identity_device is one of {FACE_DEVICES[:-1]}")
        if self.detect_hz <= 0 or self.ring_s <= 0:
            raise ValueError("perception needs detect_hz > 0 and ring_s > 0")
        if min(self.look_wait_s, self.see_wait_s) < 0:
            raise ValueError("perception's waits are not negative")
