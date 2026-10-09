"""The real models behind the face sensor's protocols; their libraries are imported here only
(the ``face`` dependency group: ``uv sync --group face``).

* :class:`InsightFace`: SCRFD detection and ArcFace identity from an InsightFace model pack
  (``buffalo_l``: SCRFD-10GF and ResNet50 ArcFace; weights non-commercial).
* :class:`EmotiEffOnnx`: EmotiEffLib's ``enet_b0_8_va_mtl`` (8 AffectNet classes plus valence and
  arousal) run straight through onnxruntime with the library's own preprocessing (224 px, RGB,
  ImageNet mean and std). The library's recognizer is not used because it reads weights only from
  ``~/.emotiefflib`` and pins the CPU provider; the network and its outputs are the same.
* :class:`MediaPipeLandmarker`: MediaPipe's Face Landmarker on a 256 px square crop of the face
  (its own detector is a short-range model that loses faces beyond about 1 m; on a crop it works
  down to 18 px faces); head pose from its transformation matrix, ``jawOpen`` and others from the
  52 blendshapes. CPU (XNNPACK); MediaPipe's Python package has no GPU delegate on macOS.

``device`` picks onnxruntime's providers: ``coreml`` is the CoreML provider on every compute unit
(Neural Engine, GPU, CPU), ``mps`` the CoreML provider on CPU and GPU only (Metal, no Neural
Engine), ``cpu`` the CPU provider. CoreML compiles each network at the first load and caches it
in ``coreml_cache``.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.identity import Vector, unit
from pepin.face_sensor.models import EMOTIONS, Detection, Expression, Image, Mesh

logger = logging.getLogger(__name__)

COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MESH_PX = 256


def providers(device: str, cache: Path) -> tuple[list[str], list[dict[str, str]]]:
    """onnxruntime's providers and their options for ``device`` (CPU when CoreML is absent)."""
    import onnxruntime as ort

    if device == "cpu" or COREML not in ort.get_available_providers():
        if device != "cpu":
            logger.warning("onnxruntime has no CoreML provider here; running on the CPU")
        return [CPU], [{}]
    cache.mkdir(parents=True, exist_ok=True)
    options = {"ModelCacheDirectory": str(cache)}
    if device == "mps":
        options["MLComputeUnits"] = "CPUAndGPU"
    return [COREML, CPU], [options, {}]


def require(path: Path, what: str) -> Path:
    """``path`` when it exists, else ``FileNotFoundError`` saying what belongs there."""
    if not path.exists():
        raise FileNotFoundError(f"{what} is missing: {path} (see config/face_sensor.json, models)")
    return path


class InsightFace:
    """SCRFD and ArcFace of one InsightFace pack; implements Detector and Embedder."""

    def __init__(
        self,
        pack: Path,
        *,
        device: str,
        cache: Path,
        det_size: int = 640,
        det_threshold: float = 0.5,
        identity: bool = True,
    ) -> None:
        """Load the pack's detector (and its recognizer when ``identity``)."""
        from insightface.app import FaceAnalysis

        require(pack, "the InsightFace model pack")
        self.name = pack.name
        names, options = providers(device, cache)
        modules = ["detection", "recognition"] if identity else ["detection"]
        self._app: Any = FaceAnalysis(
            name=str(pack), allowed_modules=modules, providers=names, provider_options=options
        )
        ctx = 0 if names[0] == COREML else -1
        self._app.prepare(ctx_id=ctx, det_size=(det_size, det_size), det_thresh=det_threshold)
        self._recognizer: Any = self._app.models.get("recognition")
        self.providers = names

    def detect(self, image: Image) -> list[Detection]:
        """Every face above the threshold, with its five landmarks."""
        boxes, kpss = self._app.det_model.detect(image, max_num=0)
        faces = []
        for i in range(boxes.shape[0]):
            x0, y0, x1, y1, score = (float(v) for v in boxes[i])
            kps = tuple((float(x), float(y)) for x, y in kpss[i]) if kpss is not None else ()
            faces.append(Detection((x0, y0, x1, y1), score, kps))
        return faces

    def embed(self, image: Image, face: Detection) -> Vector:
        """The unit-length ArcFace embedding of ``face``, aligned on its landmarks."""
        from insightface.utils import face_align

        if self._recognizer is None:
            raise RuntimeError("identity is off: the recognizer was not loaded")
        with warnings.catch_warnings():
            # scikit-image 0.26 deprecates the estimate() InsightFace's alignment calls.
            warnings.simplefilter("ignore", FutureWarning)
            crop = face_align.norm_crop(
                image, landmark=np.asarray(face.kps, dtype=np.float32), image_size=112
            )
        return unit(self._recognizer.get_feat(crop))


class EmotiEffOnnx:
    """EmotiEffLib's expression net on the tight face crop; implements ExpressionNet."""

    def __init__(self, path: Path, *, device: str, cache: Path, size: int = 224) -> None:
        """Load the ONNX file; ``size`` is the net's input side (224 for the enet_b0 nets)."""
        import onnxruntime as ort

        require(path, "the EmotiEffLib expression model")
        names, options = providers(device, cache)
        ort.set_default_logger_severity(3)
        self._session: Any = ort.InferenceSession(
            str(path), providers=names, provider_options=options
        )
        self._input = self._session.get_inputs()[0].name
        self._size = size
        self.providers = list(self._session.get_providers())

    def predict(self, crop: Image) -> Expression:
        """Softmax over the 8 classes; the last two outputs are valence and arousal."""
        import cv2

        rgb = cv2.cvtColor(cv2.resize(crop, (self._size, self._size)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        out = np.asarray(self._session.run(None, {self._input: x.transpose(2, 0, 1)[None]})[0])[0]
        logits = out[: len(EMOTIONS)]
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        valence, arousal = (float(v) for v in out[-2:])
        return Expression(tuple(float(p) for p in probs), valence, arousal)


class MediaPipeLandmarker:
    """MediaPipe's Face Landmarker on a square face crop; implements Landmarker."""

    def __init__(self, path: Path) -> None:
        """Load the ``.task`` bundle (CPU)."""
        from mediapipe.tasks.python import BaseOptions, vision

        require(path, "the MediaPipe face landmarker")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker: Any = vision.FaceLandmarker.create_from_options(options)

    def read(self, crop: Image) -> Mesh | None:
        """Head pose (degrees) and the jaw, smile and blink blendshapes of the face in ``crop``."""
        import cv2
        import mediapipe as mp

        square = cv2.resize(crop, (MESH_PX, MESH_PX), interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        yaw, pitch, roll = head_pose(np.asarray(result.facial_transformation_matrixes[0]))
        return Mesh(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            jaw_open=shapes.get("jawOpen", 0.0),
            smile=(shapes.get("mouthSmileLeft", 0.0) + shapes.get("mouthSmileRight", 0.0)) / 2,
            blink=(shapes.get("eyeBlinkLeft", 0.0) + shapes.get("eyeBlinkRight", 0.0)) / 2,
        )


def head_pose(matrix: np.ndarray[Any, Any]) -> tuple[float, float, float]:
    """Yaw, pitch and roll in degrees of a 4x4 face transform (0, 0, 0 = facing the camera)."""
    r = matrix[:3, :3]
    pitch = math.degrees(math.atan2(float(r[2, 1]), float(r[2, 2])))
    yaw = math.degrees(math.asin(-max(-1.0, min(1.0, float(r[2, 0])))))
    roll = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
    return yaw, pitch, roll


def load_models(cfg: FaceSensorConfig, *, identity: bool | None = None) -> dict[str, Any]:
    """The models ``cfg`` switches on: ``faces`` (InsightFace, always: it is the detector),
    ``expression`` and ``landmarker`` when on; ``identity`` overrides the config's switch."""
    m = cfg.models
    cache = cfg.path(m.coreml_cache)
    use_identity = m.identity if identity is None else identity
    models: dict[str, Any] = {
        "faces": InsightFace(
            cfg.path(m.detector_pack),
            device=cfg.device,
            cache=cache,
            det_size=m.det_size,
            det_threshold=m.det_threshold,
            identity=use_identity,
        )
    }
    if m.expression:
        models["expression"] = EmotiEffOnnx(
            cfg.path(m.expression_model), device=m.expression_device, cache=cache
        )
    if m.blendshapes:
        models["landmarker"] = MediaPipeLandmarker(cfg.path(m.landmarker_model))
    return models
