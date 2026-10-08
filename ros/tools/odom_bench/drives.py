"""The frozen drive set: which recorded drives the bench reads, and what each must hash to."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
REC = REPO / "ros" / "maps" / "rec"
SETS = Path(__file__).resolve().parent / "sets"


def sha256(path: Path) -> str:
    """The file's sha256 in hex."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(rec: Path, n: str) -> tuple[Path, Path]:
    """Drive n's ring-bag slice and its tape: the newest NNNN_<utc>Z_<goal>/ that has both."""
    for d in sorted(rec.glob(f"{n}_*Z_*"), reverse=True):
        tape = d.parent / (d.name + ".jsonl")
        if d.is_dir() and "_arm_" not in d.name and tape.exists():
            return d, tape
    raise SystemExit(f"no {rec}/{n}_<utc>Z_<goal>/ with its .jsonl tape")


def goal_span(tape: Path) -> tuple[float, float]:
    """The tape's navigate_to_pose rows: the first with a goal executing (status 2) .. the first
    after it with none executing."""
    t0 = t1 = None
    with tape.open() as f:
        for line in f:
            if '"nav"' not in line[:80]:
                continue
            r = json.loads(line)
            if r.get("topic") != "nav" or r.get("action") != "navigate_to_pose":
                continue
            running = 2 in r["status"]
            if t0 is None and running:
                t0 = r["t"]
            elif t0 is not None and not running:
                t1 = r["t"]
                break
    if t0 is None or t1 is None:
        raise SystemExit(f"{tape.name}: no complete navigate_to_pose goal in the tape's nav rows")
    return float(t0), float(t1)


@dataclass(frozen=True)
class Drive:
    """One drive of the set: its bag slice's name, camera-rate group and frozen hashes.

    ``reference``: the recorded VIO claims are known to be the base configuration's (the
    group's median ratio is the yardstick for the others); ``sqrt_rule``: recorded with the
    relay's sqrt(rate / 10) factor on both VIO sigmas.
    """

    n: str
    bag: str
    group: str
    reference: bool
    sqrt_rule: bool
    mcap_sha256: str
    tape_sha256: str
    truth_sha256: str | None

    @property
    def goal(self) -> str:
        """The goal's place name (the bag name's last part)."""
        return self.bag.split("_")[-1]

    @property
    def train(self) -> bool:
        """TRAIN = odd drive numbers, TEST = even."""
        return int(self.n) % 2 == 1

    def bag_dir(self, rec: Path) -> Path:
        """The bag slice's directory under ``rec``."""
        return rec / self.bag

    def tape(self, rec: Path) -> Path:
        """The tape beside the bag slice."""
        return rec / (self.bag + ".jsonl")


@dataclass(frozen=True)
class DriveSet:
    """A frozen set of drives and how their truth is made (one JSON file under ``sets/``).

    Exactly two camera-rate groups: the older one and ``live_group``, today's live rate, on which
    the verdict is decided.
    """

    path: Path
    truth_map: Path
    heading_deg: float
    lidar_yaw_offset_deg: float
    groups: dict[str, int]
    live_group: str
    replay_check: str
    drives: dict[str, Drive]

    @property
    def name(self) -> str:
        """The set's file stem (the work directory's name)."""
        return self.path.stem

    @property
    def old_group(self) -> str:
        """The group that is not the live one."""
        return next(g for g in self.groups if g != self.live_group)

    def fps(self, n: str) -> int:
        """Drive n's camera rate."""
        return self.groups[self.drives[n].group]

    def group(self, n: str) -> str:
        """Drive n's camera-rate group."""
        return self.drives[n].group

    @property
    def numbers(self) -> list[str]:
        """Every drive number, in order."""
        return sorted(self.drives)

    @classmethod
    def load(cls, path: Path) -> DriveSet:
        """Read a set file; paths in it are relative to the repository."""
        raw: dict[str, Any] = json.loads(path.read_text())
        truth = raw["truth"]
        groups = {str(k): int(v) for k, v in raw["groups"].items()}
        if len(groups) != 2 or raw["live_group"] not in groups:
            raise SystemExit(f"{path}: two camera-rate groups and the live one among them")
        drives = {
            n: Drive(
                n=n,
                bag=d["bag"],
                group=d["group"],
                reference=bool(d.get("reference", False)),
                sqrt_rule=bool(d.get("sqrt_rule", False)),
                mcap_sha256=d["mcap_sha256"],
                tape_sha256=d["tape_sha256"],
                truth_sha256=d.get("truth_sha256"),
            )
            for n, d in raw["drives"].items()
        }
        bad = [n for n, d in drives.items() if d.group not in groups]
        if bad:
            raise SystemExit(f"{path}: drives in no group: {bad}")
        return cls(
            path=path,
            truth_map=REPO / truth["map"],
            heading_deg=float(truth["heading_deg"]),
            lidar_yaw_offset_deg=float(truth["lidar_yaw_offset_deg"]),
            groups=groups,
            live_group=raw["live_group"],
            replay_check=raw["replay_check"],
            drives=drives,
        )


def entry(rec: Path, n: str, group: str) -> dict[str, Any]:
    """A new drive's set entry (bag and tape hashed), for ``freeze``."""
    bag, tape = find(rec, n)
    mcap = sorted(bag.glob("*.mcap"))
    if len(mcap) != 1:
        raise SystemExit(f"{bag}: {len(mcap)} .mcap files, expected one")
    goal_span(tape)  # a tape without a complete goal cannot be scored
    return {
        "bag": bag.name,
        "group": group,
        "mcap_sha256": sha256(mcap[0]),
        "tape_sha256": sha256(tape),
    }
