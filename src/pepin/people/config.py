"""``config/people.json``: the switch's default, where the store and the models are, and every
threshold of person recognition; the file's notes give the reasons and the measurements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from pepin.overlay import read_json
from pepin.people.matcher import Thresholds

CONFIG_FILE = "people.json"
FLAG = "people"
REPO = Path(__file__).resolve().parents[3]
DEVICES = ("coreml", "mps", "cpu")


@dataclass(frozen=True)
class FaceKnobs:
    """The identity net, its thresholds and what a face must be to count."""

    model: str = "models/face/buffalo_l/w600k_r50.onnx"
    device: str = "coreml"
    accept: float = 0.45
    margin: float = 0.08
    stranger: float = 0.25
    sure: float = 0.55
    min_px: float = 34.0  # the rectified eye's pixels (fx 454.5): 40 in the raw eye's
    strong_px: float = 52.0  # 60 in the raw eye's
    max_yaw_deg: float = 45.0
    min_det: float = 0.7
    min_sharpness: float = 10.0
    min_brightness: float = 35.0
    max_brightness: float = 235.0
    dominance: float = 1.4
    fresh_s: float = 1.5
    keep: int = 30
    unload_after_s: float = 300.0

    @property
    def thresholds(self) -> Thresholds:
        """The cosines as the matcher takes them."""
        return Thresholds(self.accept, self.margin, self.stranger, self.sure)


@dataclass(frozen=True)
class VoiceKnobs:
    """The speaker-embedding net, its thresholds and how much speech counts."""

    model: str = "models/voice/voxceleb_resnet34_LM.onnx"
    threads: int = 2
    min_s: float = 1.0
    max_s: float = 6.0
    span_db: float = 35.0
    strong_s: float = 2.0
    accept: float = 0.40
    margin: float = 0.08
    stranger: float = 0.15
    sure: float = 0.50
    keep: int = 20

    @property
    def thresholds(self) -> Thresholds:
        """The cosines as the matcher takes them."""
        return Thresholds(self.accept, self.margin, self.stranger, self.sure)


@dataclass(frozen=True)
class SessionKnobs:
    """One conversation's strangers and how a name binds to one."""

    track_face: float = 0.40
    track_voice: float = 0.35
    track_keep: int = 10
    bind_looks: int = 3
    bind_look_every_s: float = 0.15
    kind: str = "guest"


@dataclass(frozen=True)
class PeopleConfig:
    """Everything person recognition reads when its switch goes on."""

    on: bool = False
    store: str = "data/people/people.db"
    face: FaceKnobs = field(default_factory=FaceKnobs)
    voice: VoiceKnobs = field(default_factory=VoiceKnobs)
    session: SessionKnobs = field(default_factory=SessionKnobs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PeopleConfig:
        """From the file's object; ``ValueError`` naming the first value no run can use."""
        config = cls(
            on=bool(data.get("flags", {}).get(FLAG, False)),
            store=str(data.get("store", cls.store)),
            face=_section(FaceKnobs, data.get("face", {})),
            voice=_section(VoiceKnobs, data.get("voice", {})),
            session=_section(SessionKnobs, data.get("session", {})),
        )
        config.check()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> PeopleConfig:
        """Read config/people.json (or ``path``)."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} holds no JSON object")
        return cls.from_dict(data)

    def check(self) -> None:
        """``ValueError`` on a value no run could use."""
        self.face.thresholds.check("people.face")
        self.voice.thresholds.check("people.voice")
        if self.face.device not in DEVICES:
            raise ValueError(f"people.face.device {self.face.device!r} is not one of {DEVICES}")
        if not 0 < self.voice.min_s <= self.voice.max_s:
            raise ValueError("people.voice needs 0 < min_s <= max_s")
        if self.face.dominance < 1.0 or self.face.min_px <= 0:
            raise ValueError("people.face needs dominance >= 1 and min_px > 0")
        if min(self.face.keep, self.voice.keep, self.session.track_keep) < 1:
            raise ValueError("people: every keep is at least 1")
        if self.session.kind not in ("guest", "family"):
            raise ValueError("people.session.kind is guest or family")

    def path(self, value: str) -> Path:
        """A configured path against the repository root (absolute paths stay)."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else REPO / p


def _section[T](kind: type[T], block: Mapping[str, Any]) -> T:
    """A block into its dataclass: unknown keys (notes) ignored, each value cast to the type of
    its field's default."""
    defaults = kind()
    values = {
        f.name: type(getattr(defaults, f.name))(block[f.name])
        for f in fields(defaults)  # type: ignore[arg-type]
        if f.name in block
    }
    return kind(**values)
