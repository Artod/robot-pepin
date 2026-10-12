"""``config/face_sensor.json``: rates, which models run, the gates, the identity and attention
thresholds, the device and every path the face sensor reads or writes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILE = "face_sensor.json"
DEVICES = ("coreml", "mps", "ane", "cpu")
EYES = ("left", "right", "full")
STAMP_MODES = ("grab", "send", "arrival")
REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelsConfig:
    """Which models run and where their weights are (paths relative to the repository root)."""

    identity: bool = True
    expression: bool = True
    blendshapes: bool = True
    detector_pack: str = "models/face/buffalo_l"
    det_size: int = 640
    det_threshold: float = 0.6
    expression_model: str = "models/face/enet_b0_8_va_mtl.onnx"
    expression_device: str = "cpu"
    landmarker_model: str = "models/face/face_landmarker.task"
    coreml_cache: str = "models/face/coreml_cache"


@dataclass(frozen=True)
class GateConfig:
    """What a face must be for each stage to believe it: its box width and its crop's quality."""

    min_identity_px: int = 18
    min_expression_px: int = 48
    min_attention_px: int = 30
    min_sharpness: float = 10.0
    min_brightness: float = 35.0
    max_brightness: float = 235.0


@dataclass(frozen=True)
class IdentityConfig:
    """Cosine thresholds against the enrolled centroids, the frame vote and the guests."""

    accept_cos: float = 0.45
    unknown_cos: float = 0.30
    margin: float = 0.08
    vote_frames: int = 5
    min_votes: int = 3
    forget_after_s: float = 3.0
    guest_cos: float = 0.45
    enrol_min_px: int = 40
    unknown_min_det: float = 0.7
    max_yaw_deg: float = 35.0


@dataclass(frozen=True)
class ExpressionConfig:
    """The smoothing window of the expression probabilities and valence/arousal."""

    window_s: float = 3.0
    min_samples: int = 2


@dataclass(frozen=True)
class AttentionConfig:
    """Head pose toward the robot (looking) and jaw movement (talking)."""

    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 20.0
    look_window_s: float = 1.0
    talk_window_s: float = 2.0
    talk_jaw_std: float = 0.04
    min_talk_samples: int = 4


@dataclass(frozen=True)
class SourceConfig:
    """How frames are taken: which eye of a side-by-side frame and which stamp they carry."""

    eye: str = "left"
    stamp: str = "grab"


@dataclass(frozen=True)
class FaceSensorConfig:
    """Everything the face sensor reads at start; the file's notes give the reasons."""

    rate_hz: float = 3.0
    publish_hz: float = 1.0
    hold_s: float = 1.0
    stale_s: float = 3.0
    device: str = "coreml"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    faces_dir: str = "data/faces"
    out_dir: str = "data/face"
    stereo_calibration: str = "config/stereo_calibration.json"
    hfov_deg: float = 94.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FaceSensorConfig:
        """From the file's object; ``ValueError`` naming the first value out of range."""
        paths = data.get("paths", {})
        bearing = data.get("bearing", {})
        d = cls()
        config = cls(
            rate_hz=float(data.get("rate_hz", d.rate_hz)),
            publish_hz=float(data.get("publish_hz", d.publish_hz)),
            hold_s=float(data.get("hold_s", d.hold_s)),
            stale_s=float(data.get("stale_s", d.stale_s)),
            device=str(data.get("device", d.device)),
            models=_section(ModelsConfig, data.get("models", {})),
            gate=_section(GateConfig, data.get("gate", {})),
            identity=_section(IdentityConfig, data.get("identity", {})),
            expression=_section(ExpressionConfig, data.get("expression", {})),
            attention=_section(AttentionConfig, data.get("attention", {})),
            source=_section(SourceConfig, data.get("source", {})),
            faces_dir=str(paths.get("faces", d.faces_dir)),
            out_dir=str(paths.get("out", d.out_dir)),
            stereo_calibration=str(bearing.get("stereo_calibration", d.stereo_calibration)),
            hfov_deg=float(bearing.get("hfov_deg", d.hfov_deg)),
        )
        config.check()
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> FaceSensorConfig:
        """Read ``config/face_sensor.json`` (or ``path``)."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        return cls.from_dict(json.loads(Path(path).read_text()))

    def check(self) -> None:
        """Raise ``ValueError`` on a value no run could use."""
        if not self.rate_hz > 0 or not self.publish_hz > 0:
            raise ValueError("face_sensor: rate_hz and publish_hz must be positive")
        for name, device in (
            ("device", self.device),
            ("expression_device", self.models.expression_device),
        ):
            if device not in DEVICES:
                raise ValueError(f"face_sensor: {name} {device!r} is not one of {DEVICES}")
        if self.source.eye not in EYES:
            raise ValueError(f"face_sensor: source.eye {self.source.eye!r} is not one of {EYES}")
        if self.source.stamp not in STAMP_MODES:
            raise ValueError(
                f"face_sensor: source.stamp {self.source.stamp!r} is not one of {STAMP_MODES}"
            )
        i = self.identity
        if not -1.0 <= i.unknown_cos <= i.accept_cos <= 1.0:
            raise ValueError("face_sensor: identity needs -1 <= unknown_cos <= accept_cos <= 1")
        if not 1 <= i.min_votes <= i.vote_frames:
            raise ValueError("face_sensor: identity needs 1 <= min_votes <= vote_frames")
        g = self.gate
        if not 0 < g.min_identity_px <= g.min_expression_px:
            raise ValueError("face_sensor: gate needs 0 < min_identity_px <= min_expression_px")
        if not g.min_brightness < g.max_brightness:
            raise ValueError("face_sensor: gate needs min_brightness < max_brightness")

    def path(self, value: str) -> Path:
        """A configured path resolved against the repository root (absolute paths stay)."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else REPO / p


def _section[T](kind: type[T], block: Mapping[str, Any]) -> T:
    """One nested block into its dataclass: keys it does not know (notes) are ignored, and each
    value is cast to the type of the field's default."""
    defaults = kind()
    values = {
        name: type(default)(block[name])
        for name, default in vars(defaults).items()
        if name in block
    }
    return kind(**values)
