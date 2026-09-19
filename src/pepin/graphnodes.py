"""Where OUR map says each of the database's nodes is: a per-node table, and a word hung on one.

ONE GLOBAL TRANSFORM CANNOT SERVE THIS DATABASE, and that is measured, not argued. Fitted over 238
pairs from 19 tapes (scratch/graph_tie_fit.py, 2026-09-18) the pieces of ``ros/maps/rtabmap.db``
are each rigid against our map to 3.6-5.9 cm rms — and they sit in DIFFERENT frames: RTAB-Map
sessions 1 and 38 at (-10.33, +1.43, +52.1 deg), sessions 0 and 26 1.6 m away, sessions 31 and 32
129 degrees away, with 56 % of the pairs outside the chi-square gate of the best single fit. Two
more facts settle it. The live graph frame is not even the frame the saved optimised poses are in
(the ties implied live on 2026-09-17 were about 80 degrees from the offline fit), and it MOVES when
the graph is re-rooted or re-optimised: before the 2026-09-16 prune the stored anchor was (-9.17,
-0.29, -147.5 deg), after it (-11.10, +5.71, -27.8 deg). A global tie is stable within a run and
across a plain restart, and worthless the moment RTAB-Map re-optimises.

So the "90 degrees wrong at home" was never a false recognition. It was a TRUE recognition of a
node the optimised graph misplaces, read through a tie fitted to another piece.

WHAT IS HERE is the frame-free answer. For a database node ``N`` the table holds the cart's pose in
OUR map at that node's own moment (:class:`NodePose`), measured by OUR tracker off a sharp
lidar-held seating and never by reading RTAB-Map's optimised poses. When RTAB-Map localises against
``N`` the word is::

    rel  = inverse(P_N) . P_current      (both in the LIVE graph frame of that same moment)
    word = Table[N] . rel

``rel`` is a RELATIVE pose between two nodes of one piece, so however the optimiser places the
pieces — a re-root, a prune, a new closure across sessions — it cancels. Nothing in this module
ever touches the global frame, and a table entry written months ago is still true.

THE TABLE IS A LOG, ``<map id>.graph_nodes.jsonl`` beside the map, append-only for the same reason
the pairs log is (:mod:`pepin.graphtie`): a calibration is worth what it was measured over. On load
the best entry per node wins, "best" being the smaller covariance — an entry is replaced by a
sharper measurement and never by a disagreement.

WHEN ``N`` IS NOT IN THE TABLE the word may still be hung on the nearest tabled node of the SAME
SESSION (:meth:`NodeTable.hang`), because a session is what the measurement found rigid; the
substitution costs the piece's own measured rigidity, added to the covariance. When the session has
no tabled node at all there is no word: nothing on our map knows that part of the database, and
that is a thing to say out loud rather than a pose to invent.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pepin.anchors import ANCHOR_MAX_SIGMA_DEG, ANCHOR_MAX_SIGMA_M, map_slug
from pepin.fusion import COV_RIDGE, Matrix
from pepin.graphtie import TiePair, fit_tie
from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D

__all__ = [
    "ALWAYS_LOCALISE",
    "ALWAYS_MAP",
    "BY_TRUST",
    "LOCALISING",
    "MAPPING",
    "NODES_SUFFIX",
    "ModeRule",
    "ModeVerdict",
    "NodePose",
    "NodeTable",
    "NodeWord",
    "append_node",
    "entry_from_match",
    "load_nodes",
    "node_covariance",
    "nodes_path",
    "read_sessions",
    "write_nodes",
]

NODES_SUFFIX = ".graph_nodes.jsonl"
# The sentinel session of a node the database file does not know: one this run has just created.
# Every node of one run is placed by one odometry, so they are one piece by construction — which is
# the fallback when the database cannot be read at all.
THIS_RUN = -1


@dataclass(frozen=True)
class NodePose:
    """Where OUR map says one database node is: the node's id, the RTAB-Map session it belongs to,
    the moment it was created, the cart's pose in the map frame at that moment, and how sharply the
    tracker said it (x, y in metres, heading in radians)."""

    node_id: int
    session: int
    stamp: float
    cart: Pose2D
    sigma: tuple[float, float, float] = (
        ANCHOR_MAX_SIGMA_M,
        ANCHOR_MAX_SIGMA_M,
        math.radians(ANCHOR_MAX_SIGMA_DEG),
    )

    @property
    def covariance(self) -> Matrix:
        """The entry's own 3x3 over (x, y, yaw): the seating the tracker published, as it published
        it, plus the ridge that keeps a matrix invertible (:data:`pepin.fusion.COV_RIDGE`).

        No floor beyond that: the seating gate says how SOFT a seating may be and still enter the
        table, and reading it as a floor would call a 0.5 cm seating and a 3 cm one equally good —
        which is exactly the comparison :attr:`volume` has to get right.
        """
        return np.asarray(
            np.diag([s**2 + COV_RIDGE for s in self.sigma]),
            dtype=np.float64,
        )

    @property
    def volume(self) -> float:
        """The generalised variance of this entry: what "a sharper measurement" compares."""
        return float(abs(np.linalg.det(self.covariance)))

    def to_json(self) -> str:
        """One line of the node table."""
        return json.dumps(
            {
                "node": self.node_id,
                "session": self.session,
                "stamp": round(self.stamp, 4),
                "map": [round(self.cart.x, 4), round(self.cart.y, 4), round(self.cart.theta, 5)],
                "sigma": [round(float(s), 6) for s in self.sigma],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> NodePose:
        """One line back; ``ValueError``, ``KeyError`` or ``TypeError`` for anything else."""
        raw: dict[str, Any] = json.loads(text)
        cart, sigma = raw["map"], raw["sigma"]
        return cls(
            node_id=int(raw["node"]),
            session=int(raw["session"]),
            stamp=float(raw["stamp"]),
            cart=Pose2D(float(cart[0]), float(cart[1]), float(cart[2])),
            sigma=(float(sigma[0]), float(sigma[1]), float(sigma[2])),
        )


def nodes_path(directory: Path | str, map_id: str) -> Path:
    """Where the node table of ``map_id`` lives: ``<directory>/<slug>.graph_nodes.jsonl``."""
    return Path(directory) / f"{map_slug(map_id)}{NODES_SUFFIX}"


def append_node(directory: Path | str, map_id: str, entry: NodePose) -> Path:
    """Append one node's pose to the table and answer with the path."""
    path = nodes_path(directory, map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log:
        log.write(entry.to_json() + "\n")
    return path


def load_nodes(directory: Path | str, map_id: str) -> dict[int, NodePose]:
    """The node table of ``map_id``: the SHARPEST entry logged for each node, by id.

    A damaged line is skipped rather than raised on — the log is appended to by a live node and a
    power cut leaves half a line, which is not a reason to refuse the other thousand entries.
    """
    path = nodes_path(directory, map_id)
    if not path.exists():
        return {}
    table: dict[int, NodePose] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = NodePose.from_json(line)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
            continue
        held = table.get(entry.node_id)
        if held is None or entry.volume <= held.volume:
            table[entry.node_id] = entry
    return table


def write_nodes(directory: Path | str, map_id: str, entries: Iterable[NodePose]) -> Path:
    """Write a whole table at once (an offline build), oldest node first."""
    path = nodes_path(directory, map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [entry.to_json() for entry in sorted(entries, key=lambda e: e.node_id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def read_sessions(database: Path | str) -> dict[int, int]:
    """Every database node's session (``Node.map_id``), read READ-ONLY from the database file.

    An empty mapping when the file cannot be read — which is the normal case while RTAB-Map holds a
    hot journal on it — and the caller then falls back on "one run is one piece" and says so. A
    read that raises is never allowed to take a node down over a table it can live without.
    """
    try:
        connection = sqlite3.connect(f"file:{Path(database)}?mode=ro", uri=True)
        try:
            return {int(i): int(m) for i, m in connection.execute("select id, map_id from Node")}
        finally:
            connection.close()
    except sqlite3.Error:
        return {}


def node_covariance(
    entry: NodePose,
    rel: Pose2D,
    rel_covariance: Matrix | None = None,
    rigidity_m: float | None = None,
) -> Matrix:
    """What a word hung on ``entry`` is uncertain by, as a 3x3 in the MAP frame.

    Three terms, and each is a measurement:

    * the entry's OWN covariance, carried through the composition ``entry.cart . rel`` — the same
      ``[[1, 0, -dy], [0, 1, dx], [0, 0, 1]]`` Jacobian a carry uses
      (:func:`pepin.fusion.carry_pose`), so a heading error in the entry costs the word the arm
      ``|rel|`` buys;
    * RTAB-Map's own covariance of the localisation, rotated into the map by the entry's heading:
      that is what the registration between the two nodes is worth;
    * the piece's own measured RIGIDITY when the word had to be hung on a SUBSTITUTE node — the
      residual rms of the fit over that session's tabled nodes, never a constant — because the
      substitution assumes the session is rigid and that assumption has a measured error.
    """
    cos, sin = math.cos(entry.cart.theta), math.sin(entry.cart.theta)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    dx, dy = cos * rel.x - sin * rel.y, sin * rel.x + cos * rel.y
    jacobian = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
    covariance = jacobian @ entry.covariance @ jacobian.T
    if rel_covariance is not None:
        covariance = covariance + rotation @ np.asarray(rel_covariance, dtype=float) @ rotation.T
    if rigidity_m:
        covariance = covariance + np.diag([rigidity_m**2, rigidity_m**2, 0.0])
    return np.asarray(covariance, dtype=np.float64)


MAPPING, LOCALISING = "mapping", "localising"
# The three settings of the override: let the rule decide, or pin one mode.
BY_TRUST, ALWAYS_MAP, ALWAYS_LOCALISE = "trust", "map", "localise"


@dataclass(frozen=True)
class ModeVerdict:
    """Which mode RTAB-Map should be in, and the one phrase that says who decided and why."""

    mapping: bool
    why: str

    @property
    def mode(self) -> str:
        """``mapping`` or ``localising``."""
        return MAPPING if self.mapping else LOCALISING

    def text(self) -> str:
        """``mapping: lidar-held seating 1.2/0.4 cm`` for a report line."""
        return f"{self.mode}: {self.why}"


class ModeRule:
    """When the database may LEARN, and when it may only RECOGNISE — decided by trust in the pose
    and never by a sensor's name.

    The database may learn only while a SHARP pose exists that does not come from the database
    itself. Two conditions, and the same pair the volume's painting already follows:

    * SHARPNESS — the seating test every table entry has to pass (the tracker's own published
      covariance under ``anchor_max_sigma_m`` / ``anchor_max_sigma_deg``, fresh, of this moment).
      A lidar-held pose passes at 1-2 cm; a mono camera-only pose held by graph words at a sigma
      around 20 cm fails by itself, with nothing naming it; a stereo matcher good to a few cm will
      pass the day it exists, with no code change here;
    * THE PUPIL IS NOT THE TEACHER — if the source HOLDING the pose is the graph, nothing may be
      taught however sharp the number looks. Who holds the pose is read off the board's own report
      (:func:`pepin.watch.source_words`, :meth:`pepin.watch.Preflight.holding`: the enabled source
      the cart has driven the least since its word), so no rule here spells "lidar".

    A verdict must HOLD before it is acted on, and for a length that is derived and not chosen: as
    long as the evidence it rests on takes to refresh. The seating's own freshness window is that
    length — one missed ``/tracker_pose`` then cannot flap the mode, and a real change is acted on
    as soon as it is a change and not a gap.

    Pure: seating, holder and a clock in; a verdict out, and only when it is worth a service call.
    """

    def __init__(self, hold_s: float, override: str = BY_TRUST, graph: str = "graph") -> None:
        self.hold_s = hold_s
        self.override = override
        self.graph = graph
        self._applied: bool | None = None  # the mode the service has been told, once it has been
        self._wanted: ModeVerdict | None = None  # ...and the one the rule has been asking for
        self._since = 0.0  # since when, on the caller's clock
        self._switches = 0

    @property
    def mode(self) -> str:
        """The mode this rule has asked for, or ``unknown`` before it has asked for anything."""
        return "unknown" if self._applied is None else (MAPPING if self._applied else LOCALISING)

    @property
    def switches(self) -> int:
        """How many times the rule has changed its mind and said so."""
        return self._switches

    @property
    def wanted(self) -> ModeVerdict | None:
        """The verdict the rule is asking for right now, whether or not it has been acted on."""
        return self._wanted

    def verdict(self, refusal: str | None, holder: str | None, seating: str = "") -> ModeVerdict:
        """What the rule says about this instant, with no clock and no memory: the override when
        there is one, then the two conditions in the order a person would ask them."""
        if self.override == ALWAYS_MAP:
            return ModeVerdict(True, "told to map whatever the pose is worth")
        if self.override == ALWAYS_LOCALISE:
            return ModeVerdict(False, "told to localise whatever the pose is worth")
        if refusal is not None:
            return ModeVerdict(False, f"the pose is not worth learning from: {refusal}")
        if holder is None:
            return ModeVerdict(False, "nobody is holding the pose: no source has spoken")
        if holder == self.graph:
            return ModeVerdict(False, "the pose is held by the graph: the pupil is not the teacher")
        return ModeVerdict(
            True, f"the pose is held by {holder}" + (f", seated to {seating}" if seating else "")
        )

    def update(
        self, now: float, refusal: str | None, holder: str | None, seating: str = ""
    ) -> ModeVerdict | None:
        """One instant in; the verdict to ACT on, or ``None``.

        A verdict is returned only when it differs from the mode already asked for AND has been the
        answer for :attr:`hold_s` without a break — except the very first one, which is the initial
        mode and is asked for at once. Returning it counts a switch and records it as applied, so a
        caller whose service call fails must ask again by feeding the rule the next instant.
        """
        verdict = self.verdict(refusal, holder, seating)
        if self._wanted is None or verdict.mapping != self._wanted.mapping:
            self._since = now
        self._wanted = verdict
        if self._applied is None:
            self._applied = verdict.mapping
            self._switches += 1
            return verdict
        if verdict.mapping == self._applied or now - self._since < self.hold_s:
            return None
        self._applied = verdict.mapping
        self._switches += 1
        return verdict

    def text(self) -> str:
        """The mode, who decided it and why, for a report line: ``localising (the pose is held by
        the graph: the pupil is not the teacher, 2 switches, by trust)`` — and, while a verdict is
        waiting out its hold, ``mapping (asking localising: …)``."""
        wanted = self._wanted
        if wanted is None:
            said = "nothing decided yet"
        elif wanted.mapping == self._applied:
            said = wanted.why
        else:
            said = f"asking {wanted.text()}"
        return f"{self.mode} ({said}, {self._switches} switches, by {self.override})"


def entry_from_match(
    node_id: int,
    session: int,
    stamp: float,
    cart: Pose2D,
    seating: tuple[float, float, float],
    rel: Pose2D,
    rel_covariance: Matrix | None = None,
) -> NodePose:
    """Where node ``N`` is on OUR map, measured BACKWARDS from a recognition of it:
    ``Table[N] = cart . inverse(rel)``.

    This is how the table grows where no new node is ever created — in localisation mode RTAB-Map
    keeps nothing in the database, so ``ref_id`` tables nothing — and it is the only way the 43
    sessions of ``ros/maps/rtabmap.db`` that no lidar-held drive ever visited become placeable:
    drive past a place with the lidar holding the pose, let the camera recognise a node of any
    session, and that node's place on our map is known from then on.

    The entry is ONE REGISTRATION NOISIER than one measured at the node itself, and says so: its
    sigma is the tracker's seating, RTAB-Map's own registration sigma, and the arm ``|rel|`` times
    the registration's heading sigma, added in quadrature — the same lever arm every other error
    here is propagated through.
    """
    arm = math.hypot(rel.x, rel.y)
    registration = (0.0, 0.0)
    if rel_covariance is not None:
        matrix = np.asarray(rel_covariance, dtype=float)
        registration = (
            math.sqrt(max(float(matrix[0, 0]), float(matrix[1, 1]), 0.0)),
            math.sqrt(max(float(matrix[2, 2]), 0.0)),
        )
    return NodePose(
        node_id=node_id,
        session=session,
        stamp=stamp,
        cart=compose(cart, inverse(rel)),
        sigma=(
            math.hypot(seating[0], registration[0], arm * registration[1]),
            math.hypot(seating[1], registration[0], arm * registration[1]),
            math.hypot(seating[2], registration[1]),
        ),
    )


@dataclass(frozen=True)
class NodeWord:
    """One word hung on one node: the relative pose that carries the cart from that node, the entry
    it was hung on, the node RTAB-Map actually recognised, the covariance of the result in the map
    frame, and how far the hung-on node is from the cart (``lever_m``, the arm the entry's heading
    error is multiplied by)."""

    rel: Pose2D
    entry: NodePose
    matched: int
    covariance: Matrix
    lever_m: float
    rigidity_m: float | None = None

    @property
    def substituted(self) -> bool:
        """Whether the word had to be hung on a neighbour because the matched node is not tabled."""
        return self.entry.node_id != self.matched

    @property
    def pose(self) -> Pose2D:
        """Where the word puts the cart on our map: ``Table[N] . rel``."""
        return compose(self.entry.cart, self.rel)

    def text(self) -> str:
        """``node 2841 (session 38) 1.42 m away`` for a report line."""
        return (
            f"node {self.entry.node_id}"
            + (f" for {self.matched}" if self.substituted else "")
            + f" (session {self.entry.session}) {self.lever_m:.2f} m away"
            + (f", piece rigid to {self.rigidity_m * 100:.0f} cm" if self.rigidity_m else "")
        )


class NodeTable:
    """The per-node table in memory, with the sessions its nodes belong to, and the one operation
    a word needs: hang a localisation on a node this map knows.

    Pure: poses in, a :class:`NodeWord` or a refusal out. The node feeds it entries, the live graph
    poses and the database's session mapping; nothing here reads ROS or a clock.
    """

    def __init__(
        self,
        entries: Mapping[int, NodePose] | None = None,
        sessions: Mapping[int, int] | None = None,
    ) -> None:
        self._entries: dict[int, NodePose] = dict(entries or {})
        self._sessions: dict[int, int] = dict(sessions or {})
        self._rigidity: dict[tuple[int, int], float | None] = {}  # (node set, graph) -> rms

    def __len__(self) -> int:
        """How many database nodes this map has a pose for."""
        return len(self._entries)

    def __contains__(self, node_id: int) -> bool:
        """Whether this node has an entry."""
        return node_id in self._entries

    @property
    def entries(self) -> dict[int, NodePose]:
        """The table itself, by node id."""
        return self._entries

    @property
    def sessions_known(self) -> bool:
        """Whether the database file could be read for its ``id -> session`` mapping at all. False
        means every unknown id reads as :data:`THIS_RUN` — one run is one piece — which is the
        honest fallback while RTAB-Map holds a hot journal on the file, and the node says so."""
        return bool(self._sessions)

    def session_of(self, node_id: int) -> int:
        """Which RTAB-Map session a node belongs to: the database's ``Node.map_id`` when the file
        could be read, then the session the node's own TABLE ENTRY recorded on the day it was
        measured, and :data:`THIS_RUN` for a node neither knows — one this run has just created,
        which is one piece with every other node of the run."""
        known = self._sessions.get(node_id)
        if known is not None:
            return known
        entry = self._entries.get(node_id)
        return entry.session if entry is not None else THIS_RUN

    def learn_sessions(self, sessions: Mapping[int, int]) -> None:
        """Take a freshly read ``id -> session`` mapping from the database file."""
        self._sessions.update(sessions)

    def offer(self, entry: NodePose) -> bool:
        """Put one node's measured pose in the table if it is the SHARPEST one yet for that node;
        answer whether it was taken. An entry is replaced by a better measurement and by nothing
        else — never because something disagreed with it."""
        held = self._entries.get(entry.node_id)
        if held is not None and entry.volume > held.volume:
            return False
        self._entries[entry.node_id] = entry
        self._rigidity.clear()
        return True

    def coverage(self) -> dict[int, int]:
        """How many tabled nodes each session has, for a report line."""
        counts: dict[int, int] = {}
        for entry in self._entries.values():
            counts[entry.session] = counts.get(entry.session, 0) + 1
        return counts

    def hang(
        self,
        matched: int,
        current: Pose2D,
        graph_poses: Mapping[int, Pose2D],
        rel_covariance: Matrix | None = None,
        graphs: int = 0,
    ) -> NodeWord | str:
        """The word for a localisation that recognised node ``matched``, or one phrase saying why
        there is none.

        ``current`` is RTAB-Map's localisation pose and ``graph_poses`` the LIVE graph's own poses
        of its nodes — both in whatever frame this optimisation happens to use, which is exactly why
        only the RELATIVE pose between two of its nodes is taken from them.

        The node hung on is ``matched`` when it is tabled, and otherwise the tabled node of the SAME
        SESSION that is nearest to the cart in the live graph — a session is the piece the
        measurement found rigid, so a neighbour of the same session carries the word at the cost of
        that piece's rigidity and no more. ``graphs`` only keys the rigidity cache: it is recomputed
        whenever the graph moves.
        """
        session = self.session_of(matched)
        blind = not self.sessions_known and matched not in self._entries
        candidates = [
            node_id
            for node_id, entry in self._entries.items()
            if node_id in graph_poses and (blind or entry.session == session)
        ]
        if not candidates:
            tabled = sum(1 for entry in self._entries.values() if entry.session == session)
            if not tabled:
                return f"nothing on our map knows session {session} (node {matched})"
            return (
                f"session {session} has {tabled} tabled nodes and the live graph carries none of"
                f" them ({len(graph_poses)} nodes in the graph): in localisation mode the ids this"
                " run creates are temporary and never enter the graph, so nothing they were tabled"
                " under can be hung on"
            )
        if matched in self._entries and matched in graph_poses:
            chosen = matched
        else:
            chosen = min(
                candidates,
                key=lambda node_id: math.hypot(
                    graph_poses[node_id].x - current.x, graph_poses[node_id].y - current.y
                ),
            )
        rel = compose(inverse(graph_poses[chosen]), current)
        rigidity = None if chosen == matched else self.rigidity(candidates, graph_poses, graphs)
        entry = self._entries[chosen]
        return NodeWord(
            rel=rel,
            entry=entry,
            matched=matched,
            covariance=node_covariance(entry, rel, rel_covariance, rigidity),
            lever_m=math.hypot(rel.x, rel.y),
            rigidity_m=rigidity,
        )

    def rigidity(
        self, nodes: Sequence[int], graph_poses: Mapping[int, Pose2D], graphs: int = 0
    ) -> float | None:
        """How rigid this piece of the database is against our map, in metres: the residual rms of
        one rigid fit between those nodes' tabled poses and the LIVE graph's poses of the same nodes
        (:func:`pepin.graphtie.fit_tie`).

        ``None`` while fewer than two of them are in the live graph — a rigid fit over one point has
        no residual, so there is nothing measured to charge and the word's own floor stands alone.
        Measured offline on this database the sessions come out at 3.6-5.9 cm; nothing here assumes
        that, it is read off the data in hand — and a set spanning two pieces of the database comes
        out metres wide, which is exactly the answer such a word deserves.
        """
        key = (hash(tuple(sorted(nodes))), graphs)
        if key in self._rigidity:
            return self._rigidity[key]
        pairs = [
            TiePair(
                stamp=self._entries[node_id].stamp,
                cart=self._entries[node_id].cart,
                place=graph_poses[node_id],
                sigma=self._entries[node_id].sigma,
            )
            for node_id in nodes
            if node_id in self._entries and node_id in graph_poses
        ]
        answer = None
        if len(pairs) >= 2:
            fitted = fit_tie(pairs)
            if fitted is not None and fitted.inliers >= 2:
                # Over EVERY pair of the piece and not only the fit's inliers: the substitution
                # assumes one rigid transform explains this whole piece, so what it must be charged
                # is how far that assumption is from the truth — including the pairs the robust fit
                # threw out, which are precisely the places where it is worst.
                answer = math.sqrt(
                    sum(
                        (compose(fitted.pose, pair.place).x - pair.cart.x) ** 2
                        + (compose(fitted.pose, pair.place).y - pair.cart.y) ** 2
                        for pair in pairs
                    )
                    / len(pairs)
                )
        self._rigidity[key] = answer
        return answer
