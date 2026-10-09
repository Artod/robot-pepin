"""The face sensor: who is in front of Pepin, where, looking and talking or not, and how their face
looks, as one JSON line a second in data/face/latest.json (pepin.face_sensor).

    uv run --group face python scripts/face_sensor.py run                     # the head camera
    uv run --group face python scripts/face_sensor.py run --source CLIP.mjpeg # a recorded drive
    uv run --group face python scripts/face_sensor.py bench FILE_OR_DIR...    # stage times, outputs
    uv run --group face python scripts/face_sensor.py enrol NAME IMAGE_OR_CLIP...
    uv run --group face python scripts/face_sensor.py list | forget NAME

Models, thresholds, rate and device: config/face_sensor.json. Weights: models/face/ (gitignored).
Enrolled people: data/faces/<name>.npz; output: data/face/ (both gitignored, never uploaded).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.frames import (
    MJPEG_SUFFIXES,
    VIDEO_SUFFIXES,
    MjpegFile,
    Stills,
    VideoFile,
    describe,
    open_source,
    sample,
)
from pepin.face_sensor.geometry import Lens
from pepin.face_sensor.identity import (
    Gallery,
    IdentityVote,
    forget_person,
    list_people,
    save_person,
)
from pepin.face_sensor.pipeline import FacePipeline, FaceState
from pepin.face_sensor.publish import Publisher

logger = logging.getLogger("face_sensor")


def camera(eye: str) -> tuple[str | None, bool]:
    """The board's stream URL and whether the head module is mounted upside down, from
    config/camera.json's active rig."""
    from pepin.audio_link import board_host
    from pepin.camera import CameraConfig
    from pepin.deployment import config_file

    try:
        cam = CameraConfig.load(config_file("camera.json"), board=board_host())
    except (OSError, ValueError, KeyError) as error:
        logger.warning("config/camera.json unreadable (%s): no board stream", error)
        return None, False
    return cam.stream, bool(cam.rig and cam.rig.upside_down and eye != "full")


def lens(cfg: FaceSensorConfig) -> Lens:
    """The eye's lens: the stereo calibration's raw eye when it is there, else the nominal
    pinhole of ``hfov_deg`` at 800x600."""
    path = cfg.path(cfg.stereo_calibration)
    if cfg.source.eye != "full" and path.exists():
        return Lens.stereo_eye(path, cfg.source.eye)
    return Lens.nominal(800, 600, cfg.hfov_deg)


def build(cfg: FaceSensorConfig) -> tuple[FacePipeline, Gallery, str]:
    """The pipeline with the models the config switches on, the gallery of its pack, the pack."""
    from pepin.face_sensor.backends import load_models

    t0 = time.perf_counter()
    models = load_models(cfg)
    faces = models["faces"]
    pipeline = FacePipeline(
        cfg,
        faces,
        embedder=faces if cfg.models.identity else None,
        expression=models.get("expression"),
        landmarker=models.get("landmarker"),
        lens=lens(cfg),
    )
    gallery = Gallery.load(cfg.path(cfg.faces_dir), faces.name)
    enrolled = ", ".join(sorted(gallery.centroids)) or "nobody"
    print(
        f"models loaded in {time.perf_counter() - t0:.1f} s: {faces.name} on {cfg.device}"
        f" ({faces.providers[0]}), expression on {cfg.models.expression_device};"
        f" enrolled: {enrolled}; bearing from {pipeline.lens.source if pipeline.lens else '-'}",
        file=sys.stderr,
    )
    return pipeline, gallery, faces.name


def cmd_run(args: argparse.Namespace, cfg: FaceSensorConfig) -> int:
    """The sensor itself, until Ctrl-C (or the end of a recording, or ``--seconds``)."""
    from pepin.face_sensor.runner import run_live, run_recording

    url, upside_down = camera(cfg.source.eye)
    source = open_source(
        args.source,
        eye=cfg.source.eye,
        upside_down=upside_down,
        stamp=cfg.source.stamp,
        still_fps=cfg.rate_hz,
        stream_url=url,
    )
    pipeline, gallery, _ = build(cfg)
    state = FaceState(cfg, IdentityVote(cfg.identity, gallery) if cfg.models.identity else None)
    out = cfg.path(cfg.out_dir)
    publisher = Publisher(out)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if args.seconds:
        threading.Timer(args.seconds, stop.set).start()
    print(f"face sensor on {describe(source)} at {cfg.rate_hz} fps -> {out / 'latest.json'}")
    if source.live:
        stats = run_live(source, pipeline, state, cfg, publisher=publisher, stop=stop)
    else:
        stats = run_recording(
            source, pipeline, state, cfg, publisher=publisher, report=print, stop=stop
        )
    summary = {"source": describe(source), "device": cfg.device, **stats.summary()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    ms = summary["ms"]
    print(
        f"{stats.frames} frames ({stats.with_face} with a face, {stats.skipped} skipped),"
        f" {summary['fps']} fps, peak RAM {summary['rss_peak_mb']} MB; ms p50/p90 "
        + ", ".join(f"{k} {v['p50']:.1f}/{v['p90']:.1f}" for k, v in ms.items() if v["n"])
        + f" -> {out / 'summary.json'}"
    )
    return 0


def cmd_bench(args: argparse.Namespace, cfg: FaceSensorConfig) -> int:
    """Stage times and outputs over stills and clips."""
    from pepin.face_sensor.bench import BenchInputs, run_bench, save

    _, upside_down = camera(cfg.source.eye)
    pipeline, gallery, _ = build(cfg)
    inputs = BenchInputs(cfg, pipeline, gallery, upside_down)
    result = run_bench(
        [Path(p) for p in args.paths], inputs, repeat=args.repeat, crops_dir=args.crops
    )
    if args.out:
        save(result, args.out)
        print(f"-> {args.out}")
    return 0


def cmd_enrol(args: argparse.Namespace, cfg: FaceSensorConfig) -> int:
    """Add a person's samples from images and clips to data/faces/<name>.npz."""
    from pepin.face_sensor.backends import InsightFace
    from pepin.face_sensor.enrol import collect

    _, upside_down = camera(cfg.source.eye)
    m = cfg.models
    faces = InsightFace(
        cfg.path(m.detector_pack),
        device=cfg.device,
        cache=cfg.path(m.coreml_cache),
        det_size=m.det_size,
        det_threshold=m.det_threshold,
    )
    embeddings = []
    for raw in args.files:
        path = Path(raw)
        suffix = path.suffix.lower()
        frames: Any
        if suffix in MJPEG_SUFFIXES:
            frames = sample(
                MjpegFile(path, eye=cfg.source.eye, upside_down=upside_down).frames(), args.fps
            )
        elif suffix in VIDEO_SUFFIXES:
            frames = sample(
                VideoFile(path, eye=cfg.source.eye, upside_down=upside_down).frames(), args.fps
            )
        else:
            frames = Stills([path], eye=cfg.source.eye).frames()
        taken = 0
        reasons: dict[str, int] = {}
        for s in collect(frames, faces, faces, cfg):
            if s.embedding is not None and taken < args.max:
                embeddings.append(s.embedding)
                taken += 1
            else:
                reasons[s.reason] = reasons.get(s.reason, 0) + 1
        print(f"{path.name}: {taken} samples; skipped {reasons or 'none'}")
    folder = cfg.path(cfg.faces_dir)
    person = save_person(folder, args.name, embeddings, faces.name, replace=args.replace)
    low, mean = person.spread()
    print(
        f"{person.name}: {person.count} samples ({faces.name}), cos to centroid min {low:.3f}"
        f" mean {mean:.3f} -> {folder / (person.name + '.npz')}"
    )
    for other in list_people(folder):
        if other.name != person.name and other.model == person.model:
            cos = float(other.centroid @ person.centroid)
            flag = "  CLOSE: check the thresholds" if cos >= cfg.identity.unknown_cos else ""
            print(f"  vs {other.name}: cos {cos:.3f}{flag}")
    return 0


def cmd_list(args: argparse.Namespace, cfg: FaceSensorConfig) -> int:
    """The enrolled people and how tight their samples are."""
    people = list_people(cfg.path(cfg.faces_dir))
    if not people:
        print(f"nobody enrolled in {cfg.path(cfg.faces_dir)}")
    for p in people:
        low, mean = p.spread()
        print(
            f"{p.name:<16} {p.count:>4} samples  {p.model:<10} {p.updated}"
            f"  cos to centroid min {low:.3f} mean {mean:.3f}"
        )
    return 0


def cmd_forget(args: argparse.Namespace, cfg: FaceSensorConfig) -> int:
    """Delete a person's file."""
    if forget_person(cfg.path(cfg.faces_dir), args.name):
        print(f"forgot {args.name}")
        return 0
    print(f"{args.name} is not enrolled", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run one subcommand."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, help="default: config/face_sensor.json")
    ap.add_argument("--device", choices=("coreml", "mps", "cpu"), help="overrides the config")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="watch a source and publish a line a second")
    run.add_argument("--source", default="board", help="board | URL | CLIP.mjpeg | VIDEO | IMAGE")
    run.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: never)")
    bench = sub.add_parser("bench", help="stage times and outputs over images and clips")
    bench.add_argument("paths", nargs="+", help="images, folders of images, .mjpeg or video clips")
    bench.add_argument("--repeat", type=int, default=12, help="frames per still (2 are warm-up)")
    bench.add_argument("--crops", type=Path, help="write each clip face crop here for review")
    bench.add_argument("--out", type=Path, help="save the result as JSON")
    enrol = sub.add_parser("enrol", help="add a person from images or a short clip")
    enrol.add_argument("name")
    enrol.add_argument("files", nargs="+")
    enrol.add_argument("--fps", type=float, default=2.0, help="clip frames sampled per second")
    enrol.add_argument("--max", type=int, default=30, help="samples kept per file")
    enrol.add_argument("--replace", action="store_true", help="drop the person's old samples")
    sub.add_parser("list", help="the enrolled people")
    forget = sub.add_parser("forget", help="delete a person's samples")
    forget.add_argument("name")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname).1s %(name)s: %(message)s",
    )
    cfg = FaceSensorConfig.load(args.config)
    if args.device:
        from dataclasses import replace

        cfg = replace(cfg, device=args.device)
    commands = {
        "run": cmd_run,
        "bench": cmd_bench,
        "enrol": cmd_enrol,
        "list": cmd_list,
        "forget": cmd_forget,
    }
    return commands[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
