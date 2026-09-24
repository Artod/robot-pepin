#!/usr/bin/env python3
"""One place descriptor for EVERY node of an RTAB-Map database, in RTAB-Map's own storage.

WHY EVERY NODE. RTAB-Map compares two nodes' global descriptors only when both carry the same
number of them and ABORTS otherwise (Signature::compareTo, Signature.cpp:252; the whole story is
pepin.global_descriptor). The snapshots carry one each since sensor_pack's global_descriptor, so
the nodes the database already holds must carry one too before RTAB-Map may compare places by
descriptor (rtabmap_frame's place_recognition): a node with a picture gets its place vector from
the localisation service (``POST /place``), a node without one (a lidar-only snapshot) the null
descriptor. The same tag goes into every descriptor's info, so a census can tell which weights
described the nodes.

WHERE. RTAB-Map's ``GlobalDescriptor`` table (node_id, type, info, data; DatabaseSchema.sql.in),
which it reads for every node it loads (DBDriverSqlite3.cpp:3556-3610) — type 1, info and data in
``rtabmap::compressData``'s bytes (pepin.global_descriptor.PlaceDescriptor.blobs). A node's
picture is ``Data.image``, the JPEG RTAB-Map stored.

SAFELY. A database a running RTAB-Map holds is refused (a running container that mounts its
directory, or a journal beside it: a writer mid-transaction); a timestamped copy is made before
the first write (``<db>.backup-YYYYmmdd_HHMMSS``); all descriptors are replaced in ONE transaction,
so a failure leaves the database as it was; a re-run replaces what the last one wrote; a vector
that is not a finite unit vector is refused, never stored (one NaN in the database would abort
RTAB-Map at every comparison with that node). ``--check`` of a database a container holds reads a
COPY: RTAB-Map sets no sqlite busy timeout, so even a reader's lock during its commit aborts it.
The pictures are read one node at a time, so a database of tens of thousands of nodes is never in
memory at once.

Usage (the laptop, from the repository, the service running — ros/models.sh):
    uv run python ros/tools/place_backfill.py ros/maps/rtabmap.db       write every node's
    uv run python ros/tools/place_backfill.py --check DB               the census: nodes, how
                                                                        many carry one, tags
    uv run python ros/tools/place_backfill.py --check --json DB        the census as one JSON line
                                                                        on stdout
The vslam launch takes the same census itself (pepin.global_descriptor.database_census) of the file
RTAB-Map is given, on every start, before RTAB-Map opens it.
``--url`` names the service (default PEPIN_MODELS_URL, else http://127.0.0.1:8791). A missing
database's census is zero nodes: an empty room takes any descriptor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pepin.global_descriptor import (
    TYPE_PY_DESCRIPTOR,
    Census,
    PlaceDescriptor,
    database_census,
    unit_or_none,
)

HOST_URL = "http://127.0.0.1:8791"  # the service as the laptop itself sees it
Describe = Callable[[npt.NDArray[np.uint8]], tuple[npt.NDArray[np.float32], str] | None]


class RefusedError(RuntimeError):
    """The database may not be written now; the message says why."""


def census(path: Path) -> Census:
    """What the database's nodes carry (pepin.global_descriptor.database_census). Read-only, and
    only on a database no running RTAB-Map holds: the caller copies a held one first
    (:func:`census_of_copy`)."""
    return database_census(path)


def census_of_copy(path: Path) -> Census:
    """The census of a COPY of the database (and of its journal, when there is one): plain file
    reads take no sqlite lock, so the RTAB-Map writing the original can never fail a commit on
    them. A copy taken mid-transaction may be a moment old or torn; the census then says so or
    fails, and RTAB-Map is untouched either way."""
    with tempfile.TemporaryDirectory(prefix="place_census_") as tmp:
        copy = Path(tmp) / path.name
        shutil.copy2(path, copy)
        journal = Path(f"{path}-journal")
        if journal.exists():
            shutil.copy2(journal, Path(f"{copy}-journal"))
        return database_census(copy)


def holders(path: Path) -> list[str]:
    """The running containers that mount the database's directory (or a parent of it), by
    name: a running RTAB-Map there may hold the file. Empty when Docker is not here."""
    try:
        ids = subprocess.run(
            ["docker", "ps", "-q"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.split()
        if not ids:
            return []
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.Name}}|{{range .Mounts}}{{.Source}};{{end}}", *ids],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    here = path.resolve().parent
    held = []
    for line in out.splitlines():
        name, _, sources = line.partition("|")
        for source in filter(None, sources.split(";")):
            try:
                mounted = Path(source).resolve()
            except OSError:
                continue
            if here == mounted or mounted in here.parents:
                held.append(name.lstrip("/"))
                break
    return held


def refusal(path: Path) -> str | None:
    """Why this database may not be written now, or ``None``."""
    if not path.is_file():
        return f"{path} is not a database"
    journal = Path(f"{path}-journal")
    if journal.exists() and journal.stat().st_size > 0:
        return f"{journal} is not empty: a writer is mid-transaction (or crashed in one)"
    held = holders(path)
    if held:
        return f"running containers mount {path.parent}: {', '.join(held)} (stop the vslam first)"
    return None


def backup(path: Path, clock: Callable[[], float] = time.time) -> Path:
    """A copy of the database beside it, named by the time: what a bad backfill is undone from."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(clock()))
    copy = path.with_name(f"{path.name}.backup-{stamp}")
    shutil.copy2(path, copy)
    return copy


def picture(blob: bytes | None) -> npt.NDArray[np.uint8] | None:
    """A node's stored JPEG as an RGB picture, or ``None`` when it has none (or it is not one)."""
    if not blob:
        return None
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])


def backfill(
    path: Path, describe: Describe, dim: int, progress: Callable[[str], None] = print
) -> Counter[str]:
    """Replace every node's descriptors with exactly one: its picture's place vector, or the null
    descriptor of length ``dim``. One transaction; any failure rolls all of it back. Counts of
    ``described`` and of each null's reason. A picture the service does not describe, or
    describes with anything but a finite unit vector of length ``dim``, stops it: nothing is
    written."""
    counts: Counter[str] = Counter()
    db = sqlite3.connect(str(path), isolation_level=None)
    try:
        db.execute("BEGIN IMMEDIATE")
        ids = [int(r[0]) for r in db.execute("SELECT id FROM Node ORDER BY id")]
        db.execute("DELETE FROM GlobalDescriptor")
        tags: set[str] = set()
        for i, node_id in enumerate(ids, start=1):
            row = db.execute("SELECT image FROM Data WHERE id = ?", (node_id,)).fetchone()
            rgb = picture(row[0] if row is not None else None)
            descriptor = PlaceDescriptor.null(dim)
            if rgb is None:
                counts["null: no picture"] += 1
            else:
                answer = describe(rgb)
                if answer is None:
                    raise RuntimeError(f"node {node_id}: the service did not describe its picture")
                vector, tag = answer
                if vector.size != dim:
                    raise RuntimeError(
                        f"node {node_id}: {vector.size} values from {tag}, not {dim}"
                    )
                unit = unit_or_none(vector)
                if unit is None:
                    raise RuntimeError(
                        f"node {node_id}: {tag} answered no finite unit vector; nothing written"
                    )
                descriptor = PlaceDescriptor(tag, unit)
                tags.add(tag)
                counts["described"] += 1
            info, data = descriptor.blobs()
            db.execute(
                "INSERT INTO GlobalDescriptor(node_id, type, info, data) VALUES (?, ?, ?, ?)",
                (node_id, TYPE_PY_DESCRIPTOR, info, data),
            )
            if i % 25 == 0 or i == len(ids):
                progress(f"  {i}/{len(ids)} nodes")
        if len(tags) > 1:
            raise RuntimeError(f"the service changed weights during the backfill: {sorted(tags)}")
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return counts


def _describer(url: str) -> tuple[Describe, int, str]:
    """The service's /place as a describe function, its vector length and its tag — asked once
    on a grey test picture, so a service that is down is found before anything is written."""
    from pepin.localization_service import LocalizationClient

    client = LocalizationClient(url, timeout_s=30.0)
    probe = client.place(np.full((60, 80, 3), 128, dtype=np.uint8), encoding="raw")
    if probe is None:
        why = client.remotes["place"].last_error
        raise RefusedError(f"the service at {url} does not describe a picture: {why}")
    vector, tag = probe

    def describe(rgb: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], str] | None:
        return client.place(rgb, encoding="raw")  # lossless: the stored JPEG is compressed once

    return describe, int(vector.size), tag


def main(argv: list[str] | None = None) -> int:
    """Backfill, or with --check print the census; the exit status says whether it worked."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("database", type=Path)
    parser.add_argument("--check", action="store_true", help="print the census, write nothing")
    parser.add_argument("--json", action="store_true", help="with --check: the census as JSON")
    parser.add_argument("--url", default=os.environ.get("PEPIN_MODELS_URL") or HOST_URL)
    args = parser.parse_args(argv)
    path: Path = args.database
    if args.check:
        held = holders(path) if path.is_file() else []
        found = census_of_copy(path) if held else census(path)
        read = f" (read from a copy: {', '.join(held)} hold it)" if held else ""
        print(
            f"{path}{read}: {found.text()}; {found.refusal() or 'every node carries one'}",
            file=sys.stderr,
        )
        if args.json:
            print(found.to_json())
        return 0
    why = refusal(path)
    if why is not None:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    try:
        describe, dim, tag = _describer(args.url)
    except RefusedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"{path}: before, {census(path).text()}")
    copy = backup(path)
    print(f"backup: {copy}")
    t0 = time.perf_counter()
    counts = backfill(path, describe, dim)
    after = census(path)
    print(
        f"{path}: {counts['described']} described by {tag}, {counts['null: no picture']} null"
        f" (no picture), in {time.perf_counter() - t0:.1f} s; now {after.text()};"
        f" {after.refusal(tag, dim) or 'every node carries one'}"
    )
    print(json.dumps({"backup": str(copy), "census": json.loads(after.to_json())}))
    return 0 if after.refusal(tag, dim) is None else 1


if __name__ == "__main__":
    sys.exit(main())
