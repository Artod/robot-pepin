"""The static transforms a node needs, and a loud warning when they do not arrive.

A node that looks a transform up through frames joined by ``/tf_static`` edges
(``camera_link -> camera_optical``, ``camera_optical -> head_imu``, ``base_link -> laser``) cannot
work until those edges are in its buffer. Under rmw_zenoh a TRANSIENT_LOCAL subscription (tf2's
static listener) gets nothing from a publisher it has not heard yet until its history query has
finalized, and a query finalizes only when every publisher cache it reached has answered: one
peer that holds its connection open but answers nothing holds every late joiner's static facts
(scratch/zenoh_tf_static, 2026-10-05: 50 s, up to the peer's lease). :class:`StaticWait` turns
that silence into a WARN after ``after_s`` that names the missing edges, the usual cause and the
one environment variable that makes zenoh name the silent peer; :func:`waiting_nodes` reads those
lines back out of container logs for ``ros/preflight.sh``'s ``tf_static`` line.

Run as ``python3 -m pepin.static_facts < logs``: one ``OK``/``FAIL`` line about the nodes whose
last word on ``/tf_static`` is that they are still waiting. With ``--silent``, over the log of a
process started with :data:`DEBUG_RUST_LOG` (a node, or a router under PEPIN_ZROUTER_LOG): the
zenoh sessions a query was propagated to that never sent their final reply, with the ROS nodes
each one declared — the silent peer by name.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_WAIT_S = 15.0  # knob tf_static_wait_s: a static edge normally lands within ~1 s of a start
REPEAT_FACTOR = 4.0  # while still waiting the warning repeats every REPEAT_FACTOR * after_s
# What a node's container gets under PEPIN_ZENOH_DEBUG=1 (ros/laptop.sh; a test keeps the two
# equal): zenoh-ext's history queries, every query's propagation to a session and that session's
# final reply, and the liveliness tokens that tie a session's zid to its ROS node names.
DEBUG_RUST_LOG = (
    "zenoh_ext=debug,zenoh::net::routing::dispatcher::queries=trace,"
    "zenoh::net::routing::dispatcher::token=debug"
)
WAITING = "WAITING"
COMPLETE = "complete"
_LINE = re.compile(
    r"\[(?:INFO|WARN|WARNING)\] \[[0-9.]+\] \[(?P<node>[^\]]+)\]: tf_static: "
    rf"(?P<state>{WAITING}|{COMPLETE})(?P<rest>.*)$"
)

Edge = tuple[str, str]
Present = Callable[[str, str], bool]


def edge_text(edges: Iterable[Edge]) -> str:
    """``a->b, c->d``."""
    return ", ".join(f"{parent}->{child}" for parent, child in edges)


@dataclass
class StaticWait:
    """The static edges one node needs since ``started`` (seconds on any monotonic clock).

    :meth:`update` is called about once a second with the clock and a ``present(parent, child)``
    test (the node's TF buffer); it answers a ``(level, text)`` log line when one is due: a WARN
    at ``after_s`` and then every :data:`REPEAT_FACTOR` x ``after_s`` while an edge is missing,
    and one INFO when the last edge lands. :meth:`clause` is the state for a report line.
    """

    edges: tuple[Edge, ...]
    started: float
    after_s: float = DEFAULT_WAIT_S
    _have: set[Edge] = field(default_factory=set)
    _next_warn: float | None = None
    _done_at: float | None = None

    @property
    def done(self) -> bool:
        """Every edge has been seen."""
        return self._done_at is not None

    def missing(self) -> list[Edge]:
        """The edges not seen yet, in the order given."""
        return [edge for edge in self.edges if edge not in self._have]

    def update(
        self, now: float, present: Present, after_s: float | None = None
    ) -> tuple[str, str] | None:
        """Look again; ``after_s`` (a live knob) replaces the patience given at the start."""
        if after_s is not None and after_s > 0:
            self.after_s = after_s
        if self.done:
            return None
        for edge in self.missing():
            if present(*edge):
                self._have.add(edge)
        waited = now - self.started
        if not self.missing():
            self._done_at = now
            return (
                "info",
                f"tf_static: {COMPLETE}, {len(self.edges)} static edge(s) after {waited:.1f} s",
            )
        if self._next_warn is None:
            self._next_warn = self.started + self.after_s
        if now < self._next_warn:
            return None
        self._next_warn = now + REPEAT_FACTOR * self.after_s
        return "warn", self.warning(waited)

    def warning(self, waited: float) -> str:
        """The WARN text: what is missing, the usual cause, the recipe."""
        missing = self.missing()
        return (
            f"tf_static: {WAITING} {waited:.0f} s for {len(missing)} static edge(s): "
            f"{edge_text(missing)}. Usually a silent zenoh peer: some /tf_static publisher's "
            "cache has not answered this subscriber's history query, and zenoh-ext holds every "
            "publisher's samples until it does. To name the peer, restart this node's container "
            f"with RUST_LOG={DEBUG_RUST_LOG} (ros/laptop.sh: PEPIN_ZENOH_DEBUG=1) and read its "
            "log with python3 -m pepin.static_facts --silent"
        )

    def clause(self, now: float) -> str:
        """``tf_static ok`` or ``tf_static WAITING 23 s for a->b`` for a report line."""
        if self.done:
            return "tf_static ok"
        return f"tf_static {WAITING} {now - self.started:.0f} s for {edge_text(self.missing())}"


def waiting_nodes(lines: Iterable[str]) -> dict[str, str]:
    """Nodes whose LAST ``tf_static:`` line in ``lines`` (container logs, oldest first) says they
    are still waiting, with the rest of that line; a later ``complete`` clears a node."""
    last: dict[str, tuple[str, str]] = {}
    for line in lines:
        match = _LINE.search(line.rstrip())
        if match:
            last[match["node"]] = (match["state"], match["rest"].strip())
    return {node: rest for node, (state, rest) in last.items() if state == WAITING}


def verdict(waiting: dict[str, str]) -> str:
    """One preflight line body: ``OK ...`` or ``FAIL ...`` naming the waiting nodes."""
    if not waiting:
        return "OK no node waiting for /tf_static"
    names = ", ".join(_short(node, rest) for node, rest in sorted(waiting.items()))
    return (
        f"FAIL {len(waiting)} node(s) waiting for /tf_static: {names}; a silent zenoh peer is "
        "the usual cause (PEPIN_ZENOH_DEBUG=1 on the container's restart names it)"
    )


_REST = re.compile(r"^(?P<secs>\d+) s for \d+ static edge\(s\): (?P<edges>[^ ]+(?:, [^ ]+)*)\.")


def _short(node: str, rest: str) -> str:
    match = _REST.match(rest)
    return f"{node} {match['secs']} s ({match['edges']})" if match else node


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROPAGATE = re.compile(r"Propagate query to Face\{\d+, (?P<zid>[0-9a-f]+)\}:(?P<qid>\d+)")
_FINAL = re.compile(r"Face\{\d+, (?P<zid>[0-9a-f]+)\}:(?P<qid>\d+) Received final reply")
_NODE_TOKEN = re.compile(
    r"Declare token \d+ \(@ros2_lv/\d+/(?P<zid>[0-9a-f]+)/\d+/\d+/NN/[^/]*/[^/]*/"
    r"(?P<name>[^/)\s]+)"
)


SLOW_REPLY_S = 2.0  # a final reply later than this counts as silence too (a peer that thawed)
_STAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")


def _stamp(line: str) -> float | None:
    """The line's last ISO stamp (tracing's own; docker logs -t puts its own first)."""
    found = _STAMP.findall(line)
    if not found:
        return None
    return datetime.fromisoformat(found[-1][:26] + "+00:00").timestamp()


Silent = tuple[str, int, float | None, list[str]]


def silent_peers(lines: Iterable[str], slow_s: float = SLOW_REPLY_S) -> list[Silent]:
    """``(zid, queries, worst reply delay s or None when one never came, its ROS node names)``
    per zenoh session that a query was propagated to and that answered it late (``slow_s``) or
    never, from a log written under :data:`DEBUG_RUST_LOG`; most queries first."""
    sent: dict[tuple[str, str], float | None] = {}
    late: dict[str, list[float | None]] = {}
    names: dict[str, set[str]] = {}
    for raw in lines:
        line = _ANSI.sub("", raw)
        if (match := _PROPAGATE.search(line)) is not None:
            sent[(match["zid"], match["qid"])] = _stamp(line)
        elif (match := _FINAL.search(line)) is not None:
            key = (match["zid"], match["qid"])
            if key in sent:
                t0, t1 = sent.pop(key), _stamp(line)
                if t0 is not None and t1 is not None and t1 - t0 > slow_s:
                    late.setdefault(key[0], []).append(t1 - t0)
        elif (match := _NODE_TOKEN.search(line)) is not None:
            names.setdefault(match["zid"], set()).add(match["name"])
    for zid, _qid in sent:
        late.setdefault(zid, []).append(None)
    rows: list[Silent] = []
    for zid, delays in late.items():
        known = [d for d in delays if d is not None]
        worst = None if len(known) < len(delays) else max(known)
        rows.append((zid, len(delays), worst, sorted(names.get(zid, ()))))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def silent_text(rows: list[Silent]) -> str:
    """``OK ...`` when every propagated query was answered in time, else one ``SILENT`` line per
    zid."""
    if not rows:
        return "OK every query this log saw propagated was answered in time"
    out = []
    for zid, n, worst, nodes in rows:
        who = ", ".join(nodes) or "no ROS node token in this log: a router?"
        when = "never answered" if worst is None else f"answered {worst:.1f} s late at worst"
        out.append(f"SILENT {zid} ({who}): {n} quer{'y' if n == 1 else 'ies'} {when}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Read logs on stdin, print the verdict (``--silent``: the silent peers); exit 1 on FAIL."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--silent" in args:
        text = silent_text(silent_peers(sys.stdin))
        print(text)
        return 0 if text.startswith("OK") else 1
    line = verdict(waiting_nodes(sys.stdin))
    print(line)
    return 0 if line.startswith("OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
