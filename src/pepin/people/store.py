"""Where the known people live: a name, and face and voice samples (embeddings) per person.

:class:`PersonStore` is the seam: recognition reads the samples of one modality and model and
writes new people and samples through it, nothing else. :class:`SqliteStore` keeps them on this
Mac (``data/people/people.db``, gitignored, the file 0600 in a 0700 folder; embeddings as
float32 blobs; ``secure_delete`` on, so a forgotten person's rows are overwritten, not just
unlinked); :class:`MemoryStore` is the same in a dict, for tests. The companion's profile
database implements the same protocol, and swapping it in is one constructor.

Samples carry the model that made them (``w600k_r50``, ``voxceleb_resnet34_LM``): embeddings of
different networks never compare, so a model swap starts its galleries empty instead of matching
nonsense. Each person keeps at most ``keep`` samples per modality and model, the newest.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Modality = Literal["face", "voice"]
MODALITIES: tuple[Modality, ...] = ("face", "voice")
Kind = Literal["family", "guest"]
Matrix = NDArray[np.float32]


@dataclass(frozen=True)
class Person:
    """One known person: an id that never changes, the name the robot calls them by, whether
    they are family or a guest, and when they were met and last seen (Unix seconds)."""

    id: str
    name: str
    kind: Kind
    created_at: float
    last_seen_at: float | None = None


class PersonStore(Protocol):
    """People and their samples; every method is safe to call from any thread."""

    def people(self) -> list[Person]:
        """Everyone, oldest first."""
        ...

    def get(self, person_id: str) -> Person | None:
        """One person by id."""
        ...

    def named(self, name: str) -> list[Person]:
        """Everyone called ``name`` (case and surrounding blanks ignored)."""
        ...

    def add_person(self, name: str, *, kind: Kind, now: float) -> Person:
        """A new person (a new id even when the name exists)."""
        ...

    def add_samples(
        self,
        person_id: str,
        modality: Modality,
        model: str,
        vectors: Sequence[NDArray[np.floating]],
        *,
        now: float,
        keep: int,
    ) -> int:
        """Append ``vectors`` (unit length) to the person's samples of ``modality`` made by
        ``model``, keep the newest ``keep``; how many there are now. ``KeyError`` for an
        unknown person."""
        ...

    def samples(self, modality: Modality, model: str) -> dict[str, Matrix]:
        """Every person's samples of ``modality`` by ``model``: id to an ``(n, d)`` matrix."""
        ...

    def counts(self, person_id: str) -> dict[str, int]:
        """How many samples a person has per modality (any model)."""
        ...

    def seen(self, person_id: str, now: float) -> None:
        """The person was recognised at ``now``."""
        ...

    def forget(self, person_id: str) -> bool:
        """Delete the person and every sample of theirs; whether there was one."""
        ...


def normal_name(name: str) -> str:
    """A name as compared: blanks collapsed, case folded."""
    return " ".join(name.split()).casefold()


def new_id() -> str:
    """A person's id: random, never reused."""
    return uuid.uuid4().hex


@dataclass
class _Row:
    person: Person
    samples: dict[tuple[str, str], list[tuple[float, NDArray[np.float32]]]] = field(
        default_factory=dict
    )


class MemoryStore:
    """:class:`PersonStore` in memory (tests, and a run that must leave nothing on disk)."""

    def __init__(self) -> None:
        """Nobody known."""
        self._rows: dict[str, _Row] = {}
        self._lock = threading.Lock()

    def people(self) -> list[Person]:
        """Everyone, oldest first."""
        with self._lock:
            return sorted((r.person for r in self._rows.values()), key=lambda p: p.created_at)

    def get(self, person_id: str) -> Person | None:
        """One person by id."""
        with self._lock:
            row = self._rows.get(person_id)
            return row.person if row is not None else None

    def named(self, name: str) -> list[Person]:
        """Everyone called ``name``."""
        key = normal_name(name)
        return [p for p in self.people() if normal_name(p.name) == key]

    def add_person(self, name: str, *, kind: Kind, now: float) -> Person:
        """A new person."""
        person = Person(new_id(), " ".join(name.split()), kind, now)
        with self._lock:
            self._rows[person.id] = _Row(person)
        return person

    def add_samples(
        self,
        person_id: str,
        modality: Modality,
        model: str,
        vectors: Sequence[NDArray[np.floating]],
        *,
        now: float,
        keep: int,
    ) -> int:
        """Append and keep the newest ``keep``."""
        with self._lock:
            row = self._rows[person_id]
            kept = row.samples.setdefault((modality, model), [])
            kept.extend((now, np.asarray(v, dtype=np.float32).ravel().copy()) for v in vectors)
            del kept[: max(0, len(kept) - keep)]
            return len(kept)

    def samples(self, modality: Modality, model: str) -> dict[str, Matrix]:
        """Every person's samples of ``modality`` by ``model``."""
        with self._lock:
            out = {}
            for pid, row in self._rows.items():
                kept = row.samples.get((modality, model))
                if kept:
                    out[pid] = np.stack([v for _, v in kept]).astype(np.float32)
            return out

    def counts(self, person_id: str) -> dict[str, int]:
        """Samples per modality."""
        with self._lock:
            row = self._rows.get(person_id)
            out: dict[str, int] = dict.fromkeys(MODALITIES, 0)
            if row is not None:
                for (modality, _), kept in row.samples.items():
                    out[modality] += len(kept)
            return out

    def seen(self, person_id: str, now: float) -> None:
        """Recognised at ``now``."""
        with self._lock:
            row = self._rows.get(person_id)
            if row is not None:
                row.person = replace(row.person, last_seen_at=now)

    def forget(self, person_id: str) -> bool:
        """Delete the person and their samples."""
        with self._lock:
            return self._rows.pop(person_id, None) is not None


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('family', 'guest')),
    created_at REAL NOT NULL,
    last_seen_at REAL
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK (modality IN ('face', 'voice')),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_of ON samples (person_id, modality, model);
"""
VERSION = 1


class SqliteStore:
    """:class:`PersonStore` in one SQLite file on this Mac; one connection behind a lock."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the store at ``path``: the folder 0700, the file 0600."""
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        fresh = not path.exists()
        self.path = path
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if fresh:
            os.chmod(path, 0o600)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA secure_delete = ON")
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = DELETE")
            self._db.executescript(SCHEMA)
            version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                self._db.execute(f"PRAGMA user_version = {VERSION}")
            elif version != VERSION:
                raise ValueError(f"{path}: store version {version}, this code reads {VERSION}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    @staticmethod
    def _person(row: tuple[str, str, str, float, float | None]) -> Person:
        kind: Kind = "family" if row[2] == "family" else "guest"
        return Person(row[0], row[1], kind, float(row[3]), row[4])

    def people(self) -> list[Person]:
        """Everyone, oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, kind, created_at, last_seen_at FROM people ORDER BY created_at"
            ).fetchall()
        return [self._person(r) for r in rows]

    def get(self, person_id: str) -> Person | None:
        """One person by id."""
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, kind, created_at, last_seen_at FROM people WHERE id = ?",
                (person_id,),
            ).fetchone()
        return self._person(row) if row is not None else None

    def named(self, name: str) -> list[Person]:
        """Everyone called ``name`` (compared in Python: SQLite folds only ASCII case)."""
        key = normal_name(name)
        return [p for p in self.people() if normal_name(p.name) == key]

    def add_person(self, name: str, *, kind: Kind, now: float) -> Person:
        """A new person."""
        person = Person(new_id(), " ".join(name.split()), kind, now)
        with self._tx() as db:
            db.execute(
                "INSERT INTO people (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
                (person.id, person.name, person.kind, person.created_at),
            )
        return person

    def add_samples(
        self,
        person_id: str,
        modality: Modality,
        model: str,
        vectors: Sequence[NDArray[np.floating]],
        *,
        now: float,
        keep: int,
    ) -> int:
        """Append and keep the newest ``keep``."""
        rows = [np.asarray(v, dtype="<f4").ravel() for v in vectors]
        with self._tx() as db:
            if db.execute("SELECT 1 FROM people WHERE id = ?", (person_id,)).fetchone() is None:
                raise KeyError(person_id)
            db.executemany(
                "INSERT INTO samples (person_id, modality, model, dim, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(person_id, modality, model, len(r), r.tobytes(), now) for r in rows],
            )
            db.execute(
                "DELETE FROM samples WHERE person_id = ? AND modality = ? AND model = ? AND id NOT"
                " IN (SELECT id FROM samples WHERE person_id = ? AND modality = ? AND model = ?"
                " ORDER BY id DESC LIMIT ?)",
                (person_id, modality, model, person_id, modality, model, keep),
            )
            (count,) = db.execute(
                "SELECT COUNT(*) FROM samples WHERE person_id = ? AND modality = ? AND model = ?",
                (person_id, modality, model),
            ).fetchone()
        return int(count)

    def samples(self, modality: Modality, model: str) -> dict[str, Matrix]:
        """Every person's samples of ``modality`` by ``model``."""
        with self._lock:
            rows = self._db.execute(
                "SELECT person_id, dim, vector FROM samples WHERE modality = ? AND model = ?"
                " ORDER BY id",
                (modality, model),
            ).fetchall()
        grouped: dict[str, list[NDArray[np.float32]]] = {}
        for pid, dim, blob in rows:
            vector = np.frombuffer(blob, dtype="<f4")
            if len(vector) == dim:
                grouped.setdefault(pid, []).append(vector.astype(np.float32))
        return {pid: np.stack(vs) for pid, vs in grouped.items()}

    def counts(self, person_id: str) -> dict[str, int]:
        """Samples per modality."""
        with self._lock:
            rows = self._db.execute(
                "SELECT modality, COUNT(*) FROM samples WHERE person_id = ? GROUP BY modality",
                (person_id,),
            ).fetchall()
        out: dict[str, int] = dict.fromkeys(MODALITIES, 0)
        out.update({str(m): int(n) for m, n in rows})
        return out

    def seen(self, person_id: str, now: float) -> None:
        """Recognised at ``now``."""
        with self._tx() as db:
            db.execute("UPDATE people SET last_seen_at = ? WHERE id = ?", (now, person_id))

    def forget(self, person_id: str) -> bool:
        """Delete the person and their samples (overwritten on disk: secure_delete)."""
        with self._tx() as db:
            db.execute("DELETE FROM samples WHERE person_id = ?", (person_id,))
            gone = db.execute("DELETE FROM people WHERE id = ?", (person_id,)).rowcount
        return gone > 0

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            self._db.close()
