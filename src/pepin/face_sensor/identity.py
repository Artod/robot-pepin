"""Who a face is: the enrolled people, the per-frame match and the vote over frames.

Each enrolled person is one file, ``data/faces/<name>.npz``: the unit-length ArcFace embeddings of
their enrolment crops, the centroid (their normalised mean), the sample count and the model pack
that made them (embeddings of different packs are not comparable, so a gallery loads only its
own pack's files).

One frame's embedding is matched to every centroid by cosine. It votes for the best person when
the cosine reaches ``accept_cos`` and beats the runner-up by ``margin``; it votes ``unknown``
below ``unknown_cos``, but only for a clear face (detector score at least ``unknown_min_det`` and
not known to be turned more than ``max_yaw_deg`` away: a profile or a doubtful box matches nobody
and is no evidence of a stranger); otherwise it abstains. The published identity is the label
with at least ``min_votes`` of the last ``vote_frames`` ballots. Unknown faces become session guests
(``guest-1``, ``guest-2``, ...): once ``min_votes`` unknown ballots of faces at least
``enrol_min_px`` wide agree, their mean embedding is a new guest, matched like a person
afterwards and forgotten when the process ends.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.config import IdentityConfig

logger = logging.getLogger(__name__)

Vector = NDArray[np.float32]
UNKNOWN = "unknown"
GUEST = "guest"
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def unit(vector: NDArray[np.floating]) -> Vector:
    """``vector`` scaled to length 1 (a zero vector stays zero)."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def centroid(embeddings: NDArray[np.floating]) -> Vector:
    """The normalised mean of unit-length embeddings (one per row)."""
    rows = np.asarray(embeddings, dtype=np.float32)
    return unit(np.mean([unit(r) for r in rows], axis=0))


def valid_name(name: str) -> bool:
    """A person's name as a file name: lower-case, starting with a letter, never a guest's."""
    return bool(NAME.match(name)) and not name.startswith(GUEST) and name != UNKNOWN


@dataclass(frozen=True)
class Person:
    """One enrolled person: the embeddings, their centroid and the pack that made them."""

    name: str
    embeddings: NDArray[np.float32]
    model: str
    updated: str

    @property
    def centroid(self) -> Vector:
        """The normalised mean embedding."""
        return centroid(self.embeddings)

    @property
    def count(self) -> int:
        """How many enrolment samples."""
        return int(self.embeddings.shape[0])

    def spread(self) -> tuple[float, float]:
        """The lowest and the mean cosine of the samples to the centroid."""
        c = self.centroid
        cos = self.embeddings @ c
        return float(np.min(cos)), float(np.mean(cos))


def load_person(path: Path) -> Person:
    """One ``<name>.npz``; ``ValueError`` when it is not a gallery file."""
    with np.load(path, allow_pickle=False) as data:
        try:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            model = str(data["model"])
            updated = str(data["updated"])
        except KeyError as missing:
            raise ValueError(f"{path}: not a face gallery file ({missing} missing)") from None
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"{path}: no embeddings")
    return Person(path.stem, embeddings, model, updated)


def save_person(
    folder: Path, name: str, embeddings: Sequence[Vector], model: str, *, replace: bool = False
) -> Person:
    """Write ``<folder>/<name>.npz`` with ``embeddings`` added to what is there (or instead of it
    with ``replace``); ``ValueError`` on a bad name, no samples or another pack's file."""
    if not valid_name(name):
        raise ValueError(f"{name!r}: a name is lower-case letters, digits, - and _ (not a guest)")
    if not embeddings:
        raise ValueError(f"no usable face for {name}")
    rows = np.stack([unit(e) for e in embeddings]).astype(np.float32)
    path = folder / f"{name}.npz"
    if path.exists() and not replace:
        old = load_person(path)
        if old.model != model:
            raise ValueError(f"{path} holds {old.model} embeddings, not {model}: use --replace")
        rows = np.concatenate([old.embeddings, rows])
    folder.mkdir(parents=True, exist_ok=True)
    updated = time.strftime("%Y-%m-%d")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        embeddings=rows,
        centroid=centroid(rows),
        count=np.int64(rows.shape[0]),
        model=np.str_(model),
        updated=np.str_(updated),
    )
    tmp.replace(path)
    return Person(name, rows, model, updated)


def forget_person(folder: Path, name: str) -> bool:
    """Delete ``<folder>/<name>.npz``; whether there was one."""
    path = folder / f"{name}.npz"
    if not path.exists():
        return False
    path.unlink()
    return True


def list_people(folder: Path) -> list[Person]:
    """Every readable gallery file in ``folder``, by name."""
    people = []
    for path in sorted(folder.glob("*.npz")):
        try:
            people.append(load_person(path))
        except (ValueError, OSError) as error:
            logger.warning("skipping %s: %s", path, error)
    return people


@dataclass(frozen=True)
class Match:
    """The best and the runner-up cosine of one embedding against a gallery."""

    name: str | None
    cos: float
    second: float


def decide(match: Match, cfg: IdentityConfig) -> str | None:
    """One frame's verdict: the matched name (cosine at least ``accept_cos`` and ``margin`` above
    the runner-up), ``unknown`` (below ``unknown_cos``), or ``None`` (abstain, in between)."""
    if (
        match.name is not None
        and match.cos >= cfg.accept_cos
        and match.cos - match.second >= cfg.margin
    ):
        return match.name
    if match.cos < cfg.unknown_cos:
        return UNKNOWN
    return None


class Gallery:
    """Named centroids to match embeddings against."""

    def __init__(self, centroids: dict[str, Vector] | None = None) -> None:
        """Start from ``centroids`` (name to unit vector)."""
        self.centroids: dict[str, Vector] = dict(centroids or {})

    @classmethod
    def load(cls, folder: Path, model: str) -> Gallery:
        """The people of ``folder`` enrolled with ``model``; files of another pack are skipped
        with a warning."""
        centroids = {}
        for person in list_people(folder):
            if person.model != model:
                logger.warning(
                    "skipping %s: enrolled with %s, not %s", person.name, person.model, model
                )
                continue
            centroids[person.name] = person.centroid
        return cls(centroids)

    def match(self, embedding: Vector) -> Match:
        """The closest centroid by cosine, and the runner-up's cosine (-1 when there is none)."""
        if not self.centroids:
            return Match(None, -1.0, -1.0)
        e = unit(embedding)
        scored = sorted(((float(c @ e), n) for n, c in self.centroids.items()), reverse=True)
        second = scored[1][0] if len(scored) > 1 else -1.0
        return Match(scored[0][1], scored[0][0], second)


@dataclass(frozen=True)
class Ballot:
    """One frame's vote: a name, ``unknown``, or ``None`` (abstained)."""

    stamp: float
    label: str | None
    cos: float
    face_px: float
    embedding: Vector


@dataclass(frozen=True)
class Who:
    """The vote's answer: the winning label (``None`` while undecided), its mean cosine and the
    ballots it won out of those in the window."""

    id: str | None
    cos: float
    votes: int
    ballots: int

    def to_json(self) -> dict[str, object]:
        """The published form: ``{"id", "cos", "votes": "k/n"}``."""
        return {"id": self.id, "cos": round(self.cos, 3), "votes": f"{self.votes}/{self.ballots}"}


class IdentityVote:
    """The vote over the last frames, and the session's guests."""

    def __init__(self, cfg: IdentityConfig, gallery: Gallery) -> None:
        """Vote against ``gallery`` (family) with an empty guest list."""
        self.cfg = cfg
        self.gallery = gallery
        self.guests = Gallery()
        self._window: deque[Ballot] = deque(maxlen=cfg.vote_frames)
        self._last = -np.inf

    def add(
        self,
        stamp: float,
        embedding: Vector,
        face_px: float,
        *,
        det: float = 1.0,
        frontal: bool | None = None,
    ) -> Ballot:
        """One frame's embedding of the tracked face, with its detector score and whether the
        head faces the camera (``None``: unknown); the ballot it cast."""
        self._expire(stamp)
        clear = det >= self.cfg.unknown_min_det and frontal is not False
        ballot = self._ballot(stamp, unit(embedding), face_px, clear)
        self._window.append(ballot)
        self._last = stamp
        self._promote_guest()
        return self._window[-1]

    def current(self, now: float) -> Who | None:
        """The vote's answer at ``now``; ``None`` when no face voted in ``forget_after_s``."""
        self._expire(now)
        if not self._window:
            return None
        counts = Counter(b.label for b in self._window if b.label is not None)
        ballots = len(self._window)
        if not counts:
            return Who(None, max(b.cos for b in self._window), 0, ballots)
        label, votes = counts.most_common(1)[0]
        mean_cos = float(np.mean([b.cos for b in self._window if b.label == label]))
        if votes < self.cfg.min_votes:
            return Who(None, mean_cos, votes, ballots)
        return Who(label, mean_cos, votes, ballots)

    def reset(self) -> None:
        """Forget the window (a new face is being tracked)."""
        self._window.clear()

    def _expire(self, now: float) -> None:
        if now - self._last > self.cfg.forget_after_s:
            self._window.clear()

    def _ballot(self, stamp: float, e: Vector, face_px: float, clear: bool) -> Ballot:
        cfg = self.cfg
        family = self.gallery.match(e)
        label = decide(family, cfg)
        if label != UNKNOWN:
            return Ballot(stamp, label, family.cos, face_px, e)
        guest = self.guests.match(e)
        if guest.name is not None and guest.cos >= cfg.guest_cos:
            return Ballot(stamp, guest.name, guest.cos, face_px, e)
        return Ballot(stamp, UNKNOWN if clear else None, family.cos, face_px, e)

    def _promote_guest(self) -> None:
        """Turn agreeing unknown ballots of big enough faces into a new guest."""
        unknown = [
            b for b in self._window if b.label == UNKNOWN and b.face_px >= self.cfg.enrol_min_px
        ]
        if len(unknown) < self.cfg.min_votes:
            return
        mean = centroid(np.stack([b.embedding for b in unknown]))
        agree = [b for b in unknown if float(b.embedding @ mean) >= self.cfg.guest_cos]
        if len(agree) < self.cfg.min_votes:
            return
        name = f"{GUEST}-{len(self.guests.centroids) + 1}"
        self.guests.centroids[name] = centroid(np.stack([b.embedding for b in agree]))
        relabelled = [
            Ballot(
                b.stamp,
                name,
                float(b.embedding @ self.guests.centroids[name]),
                b.face_px,
                b.embedding,
            )
            if any(b is a for a in agree)
            else b
            for b in self._window
        ]
        self._window.clear()
        self._window.extend(relabelled)
        logger.info("new session guest %s from %d frames", name, len(agree))
