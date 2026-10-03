"""Fit the SO-101 links' collision boxes (config/arm.json ``links``) to upstream's meshes.

    uv run python scripts/arm_boxes.py [--meshes DIR] [--check]

The meshes are fetched once into DIR (default ~/.cache/pepin/so101, 13 files, 16 MB) from the
commit the URDF was vendored from (``pepin/vendor/so101``) and are never committed. Each link's
visual meshes go into the link's own frame by their ``<visual><origin>``, and the link gets the
smallest of: the box along the link's own axes, the box along the vertices' principal axes, and
that better box cut in two across its longest side where two boxes save at least a quarter of
its volume. Prints the ``links`` block of config/arm.json: exact hulls, because the margin is the
run time's (``margin_m``). ``--check`` compares the printed block with the file's instead and
reports the largest corner difference, so a hand edit or an upstream change shows.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.mounts import rotation_from_rpy, rpy_from_rotation
from pepin.vendor import so101

Array = NDArray[np.float64]
REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "pepin" / "so101"
RAW = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{commit}/Simulation/SO101/{name}"
SPLIT_SAVES = 0.25  # two boxes replace one only when they save this share of its volume


def read_stl(path: Path) -> Array:
    """The vertices (n, 3) of a binary STL, metres as onshape-to-robot writes them."""
    data = path.read_bytes()
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, dtype=record, count=count, offset=84)
    vertices: Array = triangles["v"].reshape(-1, 3).astype(float)
    return vertices


def fetch(name: str, meshes: Path) -> Path:
    """The mesh ``name`` (``assets/...stl``) from the cache, fetched at the vendored commit."""
    path = meshes / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = RAW.format(commit=so101.COMMIT, name=name)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _attr(origin: ET.Element | None, name: str) -> str:
    """An ``<origin>``'s ``xyz`` or ``rpy``, zero when the element or the attribute is absent."""
    return origin.get(name, "0 0 0") if origin is not None else "0 0 0"


def link_vertices(urdf: Path, meshes: Path) -> dict[str, Array]:
    """Every link's visual mesh vertices in the link's own frame."""
    root = ET.fromstring(urdf.read_text())
    out: dict[str, Array] = {}
    for link in root.findall("link"):
        parts = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            xyz = [float(v) for v in _attr(origin, "xyz").split()]
            rpy = [float(v) for v in _attr(origin, "rpy").split()]
            vertices = read_stl(fetch(mesh.get("filename", ""), meshes))
            parts.append(vertices @ rotation_from_rpy(*rpy).T + np.array(xyz))
        if parts:
            out[link.get("name", "")] = np.vstack(parts)
    return out


def box_along(points: Array, axes: Array) -> tuple[Array, Array, Array]:
    """The tightest box with ``axes`` (columns, right-handed): centre, axes, half sides."""
    local = points @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    return axes @ ((lo + hi) / 2.0), axes, (hi - lo) / 2.0


def principal_axes(points: Array) -> Array:
    """The vertices' principal axes as a right-handed rotation (columns)."""
    _, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] = -vectors[:, 0]
    out: Array = vectors
    return out


def volume(half: Array) -> float:
    return float(np.prod(2.0 * half))


def fit(points: Array) -> list[tuple[Array, Array, Array]]:
    """One or two boxes for a link's vertices, by the rule in the module docstring."""
    candidates = [box_along(points, np.eye(3)), box_along(points, principal_axes(points))]
    best = min(candidates, key=lambda box: volume(box[2]))
    centre, axes, half = best
    longest = int(np.argmax(half))
    along = (points - centre) @ axes[:, longest]
    split: list[tuple[Array, Array, Array]] | None = None
    for cut in np.linspace(-0.6, 0.6, 13) * half[longest]:
        lo, hi = points[along <= cut], points[along > cut]
        if len(lo) < 4 or len(hi) < 4:
            continue
        pair = [box_along(lo, axes), box_along(hi, axes)]
        if split is None or sum(volume(b[2]) for b in pair) < sum(volume(b[2]) for b in split):
            split = pair
    if split is not None and sum(volume(b[2]) for b in split) <= (1 - SPLIT_SAVES) * volume(half):
        return split
    return [best]


def block(vertices: dict[str, Array]) -> list[dict[str, object]]:
    """The ``links`` entries, millimetre-rounded, in URDF order."""
    entries: list[dict[str, object]] = []
    for link, points in vertices.items():
        boxes = fit(points)
        for k, (centre, axes, half) in enumerate(boxes):
            entries.append(
                {
                    "name": link if len(boxes) == 1 else f"{link}_{k}",
                    "link": link,
                    "centre_m": [round(float(v), 4) + 0.0 for v in centre],  # never -0.0
                    "rpy_deg": [round(math.degrees(a), 2) + 0.0 for a in rpy_from_rotation(axes)],
                    "half_m": [round(float(v), 4) for v in half],
                }
            )
    return entries


def corners(entry: dict[str, object]) -> Array:
    centre = np.array(entry["centre_m"], dtype=float)
    axes = rotation_from_rpy(*(math.radians(float(v)) for v in entry["rpy_deg"]))  # type: ignore[attr-defined]
    half = np.array(entry["half_m"], dtype=float)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    out: Array = (signs * half) @ axes.T + centre
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meshes", type=Path, default=CACHE, help="the mesh cache directory")
    parser.add_argument("--check", action="store_true", help="compare with config/arm.json")
    args = parser.parse_args()
    with np.errstate(all="ignore"):  # Accelerate's matmul warns spuriously on M-series numpy
        vertices = link_vertices(so101.URDF, args.meshes)
        entries = block(vertices)
    bad = [link for link, points in vertices.items() if not np.isfinite(points).all()]
    if bad:
        raise SystemExit(f"non-finite vertices in {bad}")
    for link, points in vertices.items():
        hull = np.ptp(points, axis=0).round(4).tolist()
        print(f"# {link}: {len(points)} vertices, hull {hull} m")
    if not args.check:
        print(json.dumps(entries, indent=2))
        return
    shipped = json.loads((REPO / "config" / "arm.json").read_text())["links"]
    if [e["name"] for e in shipped] != [e["name"] for e in entries]:
        print(f"MISMATCH: names {[e['name'] for e in shipped]} vs {[e['name'] for e in entries]}")
        return
    worst = max(
        float(np.abs(corners(a) - corners(b)).max()) for a, b in zip(shipped, entries, strict=True)
    )
    print(f"config/arm.json links match the meshes within {worst * 1000:.1f} mm (corners)")


if __name__ == "__main__":
    main()
