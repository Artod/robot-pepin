"""The face sensor's bench: stage times and outputs over still images or recorded clips.

Stills: each image is offered ``repeat`` times as a frame stream at the sensor's rate (the first
two are warm-up, left out of the times), and the bench prints what one frame says (faces, size,
gate, identity cosine and verdict, expression, head pose, bearing) and the line the sensor would
publish after those frames. Clips: the clip is sampled to the sensor's rate and run like a
recording; the bench counts frames with a face, keeps a crop of every detected face for review
(local files, never uploaded) and tallies the published identities. Both end with every stage's
p50/p90, the peak RAM and the rate the laptop could sustain.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.frames import (
    MJPEG_SUFFIXES,
    STILL_SUFFIXES,
    VIDEO_SUFFIXES,
    Frame,
    FrameSource,
    Image,
    MjpegFile,
    Stills,
    VideoFile,
)
from pepin.face_sensor.geometry import crop_box
from pepin.face_sensor.identity import Gallery, IdentityVote, decide
from pepin.face_sensor.models import EMOTIONS
from pepin.face_sensor.pipeline import (
    STAGES,
    FacePipeline,
    FaceState,
    FrameResult,
    percentiles,
)
from pepin.face_sensor.runner import Stats, peak_rss_mb, run_recording

WARMUP = 2
MAX_CROPS = 40


@dataclass
class BenchInputs:
    """What the bench needs besides the files: the pipeline and the gallery."""

    cfg: FaceSensorConfig
    pipeline: FacePipeline
    gallery: Gallery
    upside_down: bool = False


def frame_summary(result: FrameResult, inputs: BenchInputs) -> dict[str, Any]:
    """One frame's outputs, unsmoothed, for the bench table."""
    obs = result.best
    out: dict[str, Any] = {"faces": result.faces}
    if obs is None:
        return out
    out.update(
        face_px=round(obs.face_px),
        det=round(obs.detection.score, 3),
        gate=obs.gate.reason,
        sharpness=round(obs.quality.sharpness),
        brightness=round(obs.quality.brightness),
    )
    if obs.bearing is not None:
        out["bearing"] = [round(obs.bearing[0], 1), round(obs.bearing[1], 1)]
    if obs.embedding is not None:
        match = inputs.gallery.match(obs.embedding)
        out["identity"] = {
            "best": match.name,
            "cos": round(match.cos, 3),
            "verdict": decide(match, inputs.cfg.identity),
        }
    if obs.expression is not None:
        e = obs.expression
        ranked = sorted(zip(EMOTIONS, e.probs, strict=True), key=lambda t: -t[1])[:3]
        out["expr"] = {
            "top3": [[k, round(p, 2)] for k, p in ranked],
            "valence": round(e.valence, 2),
            "arousal": round(e.arousal, 2),
        }
    if obs.mesh is not None:
        m = obs.mesh
        out["mesh"] = {
            "yaw": round(m.yaw, 1),
            "pitch": round(m.pitch, 1),
            "jaw": round(m.jaw_open, 3),
            "smile": round(m.smile, 2),
        }
    return out


def bench_still(path: Path, inputs: BenchInputs, repeat: int) -> tuple[dict[str, Any], Stats]:
    """One image offered ``repeat`` times: its per-frame outputs and the published line."""
    cfg = inputs.cfg
    stills = Stills([path], fps=cfg.rate_hz, repeat=repeat, start=time.time(), eye=cfg.source.eye)
    state = FaceState(cfg, IdentityVote(cfg.identity, inputs.gallery))
    stats = Stats()
    last: FrameResult | None = None
    for i, frame in enumerate(stills.frames()):
        last = inputs.pipeline.process(frame)
        state.update(last)
        if i >= WARMUP or repeat <= WARMUP:
            stats.add(last)
    assert last is not None
    out = {"file": path.name, **frame_summary(last, inputs)}
    out["published"] = state.record(cfg.rate_hz)
    return out, stats


class _Remembering:
    """A source whose frames keep the image they last decoded, for the bench's crops."""

    def __init__(self, source: FrameSource) -> None:
        self.live = source.live
        self.image: Image | None = None
        self._source = source

    def frames(self) -> Iterator[Frame]:
        for frame in self._source.frames():
            yield Frame(frame.stamp, frame.index, self._loader(frame))

    def close(self) -> None:
        self._source.close()

    def _loader(self, frame: Frame) -> Callable[[], Image]:
        def load() -> Image:
            self.image = frame.image()
            return self.image

        return load


def bench_clip(
    path: Path, inputs: BenchInputs, crops_dir: Path | None
) -> tuple[dict[str, Any], Stats]:
    """One recording at the sensor's rate: faces found, crops kept, identities published."""
    cfg = inputs.cfg
    clip: FrameSource
    if path.suffix.lower() in MJPEG_SUFFIXES:
        clip = MjpegFile(
            path, eye=cfg.source.eye, upside_down=inputs.upside_down, stamp=cfg.source.stamp
        )
    else:
        clip = VideoFile(path, eye=cfg.source.eye, upside_down=inputs.upside_down)
    source = _Remembering(clip)
    state = FaceState(cfg, IdentityVote(cfg.identity, inputs.gallery))
    found: list[dict[str, Any]] = []
    first: list[float] = []
    who: Counter[str] = Counter()

    def keep(result: FrameResult) -> None:
        if not first:
            first.append(result.stamp)
        record = state.record(cfg.rate_hz)
        if record is not None and record.get("who") is not None:
            who[str(record["who"]["id"])] += 1
        if result.best is None:
            return
        row: dict[str, Any] = {"t": round(result.stamp - first[0], 2), "index": result.index}
        row.update(frame_summary(result, inputs))
        found.append(row)
        if crops_dir is not None and len(found) <= MAX_CROPS and source.image is not None:
            _save_crop(crops_dir, path, result, source.image)

    stats = run_recording(source, inputs.pipeline, state, cfg, publisher=None, on_result=keep)
    out = {
        "file": path.name,
        "frames": stats.frames,
        "frames_with_face": stats.with_face,
        "faces": found[:MAX_CROPS],
        "who_per_frame": dict(who),
    }
    return out, stats


def _save_crop(folder: Path, clip: Path, result: FrameResult, image: Image) -> None:
    """Write the best face of ``result`` grown 60 % as a JPEG named by clip, frame index, size
    and detector score."""
    import cv2

    obs = result.best
    if obs is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    crop = crop_box(image, obs.detection.box, margin=0.6, square=True)
    name = (
        f"{clip.stem[:24]}_{result.index:05d}_{round(obs.face_px)}px_{obs.detection.score:.2f}.jpg"
    )
    cv2.imwrite(str(folder / name), crop)


def run_bench(
    paths: Sequence[Path],
    inputs: BenchInputs,
    *,
    repeat: int = 12,
    crops_dir: Path | None = None,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Bench every file of ``paths`` (folders expand to their images and clips) and print the
    tables; the dictionary is what ``--out`` saves."""
    files = list(_expand(paths))
    results: list[dict[str, Any]] = []
    all_ms: dict[str, list[float]] = {stage: [] for stage in STAGES}
    frames = 0
    for path in files:
        if path.suffix.lower() in MJPEG_SUFFIXES + VIDEO_SUFFIXES:
            row, stats = bench_clip(path, inputs, crops_dir)
            report(_clip_line(row))
        else:
            row, stats = bench_still(path, inputs, repeat)
            report(_still_line(row))
        results.append(row)
        frames += stats.frames
        for stage, values in stats.ms.items():
            all_ms.setdefault(stage, []).extend(values)
    ms = {stage: percentiles(values) for stage, values in all_ms.items() if values}
    total = sum(all_ms.get("total", [])) / 1e3
    summary = {
        "frames": frames,
        "ms": ms,
        "capacity_fps": round(frames / total, 1) if total > 0 else 0.0,
        "rss_peak_mb": round(peak_rss_mb(), 1),
        "device": inputs.cfg.device,
        "rate_hz": inputs.cfg.rate_hz,
    }
    report("stage ms    " + "  ".join(f"{s:>12}" for s in ms))
    report("p50 / p90   " + "  ".join(f"{v['p50']:>5.1f} / {v['p90']:<5.1f}" for v in ms.values()))
    report(
        f"frames {frames}, the laptop sustains {summary['capacity_fps']} fps of this mix,"
        f" peak RAM {summary['rss_peak_mb']} MB, device {inputs.cfg.device}"
    )
    return {"summary": summary, "files": results}


def _expand(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.suffix.lower() in STILL_SUFFIXES + MJPEG_SUFFIXES + VIDEO_SUFFIXES
                )
            )
        else:
            out.append(path)
    return out


def _still_line(row: dict[str, Any]) -> str:
    if "face_px" not in row:
        return f"{row['file']:<24} no face"
    ident = row.get("identity")
    expr = row.get("expr")
    mesh = row.get("mesh")
    who = ((row.get("published") or {}).get("who") or {}).get("id")
    parts = [
        f"{row['face_px']:>4} px det {row['det']:.2f} {row['gate']:<6} sharp {row['sharpness']:>5}"
    ]
    parts.append(
        f"cos {ident['cos']:.3f} {ident['verdict'] or '-':<8} -> {who or '-':<8}"
        if ident
        else "identity -"
    )
    if expr:
        top, p = expr["top3"][0]
        parts.append(f"{top} {p:.2f} V{expr['valence']:+.2f} A{expr['arousal']:+.2f}")
    else:
        parts.append("expr -")
    parts.append(
        f"yaw {mesh['yaw']:+.0f} pitch {mesh['pitch']:+.0f} jaw {mesh['jaw']:.2f}"
        if mesh
        else "mesh -"
    )
    return f"{row['file']:<24} " + " | ".join(parts)


def _clip_line(row: dict[str, Any]) -> str:
    sizes = [f["face_px"] for f in row["faces"]]
    return (
        f"{row['file']:<44} {row['frames']:>5} frames, {row['frames_with_face']:>3} with a face"
        f" (px {min(sizes) if sizes else '-'}-{max(sizes) if sizes else '-'}),"
        f" published who per frame {row['who_per_frame']}"
    )


def save(result: dict[str, Any], path: Path) -> None:
    """Write the bench result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
