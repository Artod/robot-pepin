"""May a goal start: the age of ``map -> base_link``, and whether this start of RTAB-Map has
been placed.

The decision is pure — the readings and the answer, no ROS — so the goal server and the goal
client (ros/tools/goto_ros.py) ask the same rule and a test can hold it. RTAB-Map on the laptop
owns ``map -> odom``. The board tracker's half of this module (the whole-map search, the fused
sigma, the source roster and the three-question preflight) is on the tag alt/tracker-2026-09-22.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Which rule reached a :class:`Readiness`, so a refusal says what judged it.
BY_TF = "tf"
BY_PLACEMENT = "placement"

# WHETHER RTAB-MAP'S POSE RESTS ON ANYTHING SINCE ITS START. RTAB-Map
# owns map -> odom and publishes it from the moment it starts: the pose it SAVED at its last
# shutdown, before it has recognised anything. That transform is milliseconds fresh and says
# nothing about where the cart is — on 2026-09-23 it put the cart "at home" while it stood at the
# bookshelf (0 of 198 updates recognised a node) and, after the next restart, 76 cm from its base
# inside the table, and each time the preflight read "map -> base_link 8 ms old" and let the
# drive start: the startup-zero trap a third time (the tracker's default pose, then the graph's
# lone first node). pepin_bringup.rtabmap_frame counts RTAB-Map's updates against the ones that
# recognised a node of the database it LOADED and hears the operator's seeds, and says on this
# latched topic whether the present start has been PLACED (:class:`Placement`).
PLACEMENT_TOPIC = "/localization/placement"
# The switch back (rule 19), on the board's side of the word: the goal server's flag, which
# ros/tools/goto_ros.py reads too, so it lifts a refusal of silence as well as of "not placed".
PLACEMENT_SWITCH = (
    "ros/flags.sh set goal_server start_needs_placement false takes the pose as it is"
)
PLACEMENT_REMEDY = (
    "the pose is the one RTAB-Map saved at its last shutdown, not a localisation. Either seed it"
    " where the cart stands (ros/goto.sh seed X Y YAW) or move the cart where the camera sees a"
    f" mapped place until RTAB-Map recognises one, then send the goal again ({PLACEMENT_SWITCH})"
)

# How old the map -> base_link edge may be and still be a pose to start a drive on. RTAB-Map
# broadcasts map -> odom at 20 Hz over odometry published at 50 Hz, so a whole second without one
# is twenty missed broadcasts: the BROADCASTER or the odometry has stopped, not jittered. Nav2's
# own tolerance is 0.3 s.
TF_FRESH_S = 1.0


@dataclass(frozen=True)
class Readiness:
    """Whether a goal may start now, and why not when it may not — the phrase the operator reads
    on a refusal, and which rule (:data:`BY_TF`, :data:`BY_PLACEMENT`) reached it."""

    ready: bool
    reason: str = ""
    rule: str = ""


@dataclass(frozen=True)
class Placement:
    """What RTAB-Map's pose rests on since RTAB-Map's own start, as pepin_bringup.rtabmap_frame
    publishes it (latched) on :data:`PLACEMENT_TOPIC`.

    ``updates`` is the updates of this start, ``recognised`` how many of them named a node of
    the database it LOADED, ``seeds`` the operator seeds (``/rtabmap/initialpose``) heard since
    its first update. ``loaded`` is ``False`` for a start that loaded an empty database (its
    first node is 1: the start pose IS the map's origin) and ``None`` before its first update.
    ``required`` is rtabmap_frame's flag ``start_needs_placement``; off, every pose counts, which
    is the behaviour before 2026-09-23.
    """

    updates: int
    recognised: int
    seeds: int
    loaded: bool | None
    required: bool = True

    @property
    def placed(self) -> bool:
        """Whether a goal may start on this pose: recognised, seeded, a map of its own, or not
        asked at all."""
        return not self.required or self.loaded is False or self.recognised > 0 or self.seeds > 0

    def how(self) -> str:
        """One phrase for the operator: what the pose rests on, with the counts behind it."""
        counts = (
            f"{self.recognised} of {self.updates} updates since RTAB-Map's start recognised the"
            f" loaded map, {self.seeds} seed message{'' if self.seeds == 1 else 's'}"
            " (/rtabmap/initialpose)"
        )
        if not self.required:
            return f"not asked (rtabmap_frame start_needs_placement is off): {counts}"
        if self.loaded is False:
            return "RTAB-Map loaded an empty database: its start pose is this map's origin"
        if self.loaded is None:
            return "RTAB-Map has not made one update since its start"
        if self.recognised > 0 or self.seeds > 0:
            return f"placed: {counts}"
        return f"not placed: {counts}"

    def to_json(self, stamp: float) -> str:
        """The latched message rtabmap_frame publishes; ``stamp`` is its own clock."""
        return json.dumps(
            {
                "placed": self.placed,
                "updates": self.updates,
                "recognised": self.recognised,
                "seeds": self.seeds,
                "loaded": self.loaded,
                "required": self.required,
                "stamp": round(stamp, 3),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Placement | None:
        """One message of :data:`PLACEMENT_TOPIC`; ``None`` for anything that does not parse —
        which the preflight reads as "nobody said", never as placed."""
        try:
            heard = json.loads(text)
            loaded = heard["loaded"]
            return cls(
                int(heard["updates"]),
                int(heard["recognised"]),
                int(heard["seeds"]),
                None if loaded is None else bool(loaded),
                bool(heard.get("required", True)),
            )
        except (TypeError, ValueError, KeyError):
            return None


@dataclass(frozen=True)
class GoalGate:
    """May the cart be sent to a goal: on the age of ``map -> base_link``. Younger than
    ``fresh_s`` it is a pose to drive on; missing or stale, the refusal says which and how
    stale. Whether the pose RESTS on anything since RTAB-Map's start is the question of
    :meth:`Preflight.placement`, asked after this one."""

    fresh_s: float = TF_FRESH_S

    def verdict(self, tf_age_s: float | None) -> Readiness:
        """One goal's answer; ``tf_age_s`` is how many seconds ago ``map -> base_link`` was
        stamped (``None``: nothing publishes it)."""
        if tf_age_s is None:
            return Readiness(
                False,
                rule=BY_TF,
                reason="nothing publishes map -> base_link: the SLAM half of the stack is not up",
            )
        if tf_age_s > self.fresh_s:
            return Readiness(
                False,
                rule=BY_TF,
                reason=f"map -> base_link is {tf_age_s:.1f} s old: a drive needs it fresher than"
                f" {self.fresh_s:.1f} s",
            )
        return Readiness(True, rule=BY_TF)


@dataclass(frozen=True)
class Check:
    """One preflight question and its answer: the name the operator reads, whether it passed,
    and the reading that decided it."""

    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        """The one line printed per check: ``preflight certainty  ok       sigma 0.06 m ...``."""
        return f"preflight {self.name:9s} {'ok     ' if self.ok else 'REFUSED'}  {self.detail}"


class Preflight:
    """The question asked before a goal is sent where RTAB-Map owns ``map -> odom``, answered
    out loud: has this start of RTAB-Map recognised the loaded map, or been seeded, or is its
    pose still the one it saved at its last shutdown (2026-09-23)."""

    @staticmethod
    def placement(placement: Placement | None, asked: bool = True) -> Check:
        """Where RTAB-Map owns ``map -> odom``: has this start of RTAB-Map been PLACED — a node of
        the loaded map recognised, or an operator's seed — or is its pose still the one it saved
        at its last shutdown (:data:`PLACEMENT_TOPIC`)? ``None`` is nothing heard at all, which
        says nothing either way and is refused the same, with where to look. A refusal always
        says what to do: seed, or let the camera see a mapped place. ``asked`` is the goal
        server's flag ``start_needs_placement``; off, this passes whatever was heard."""
        if not asked:
            return Check(
                "placed",
                True,
                "not asked (goal_server start_needs_placement is off): RTAB-Map's pose is taken"
                " as it is, "
                + (f"nothing on {PLACEMENT_TOPIC}" if placement is None else placement.how()),
            )
        if placement is None:
            return Check(
                "placed",
                False,
                f"nothing on {PLACEMENT_TOPIC}: rtabmap_frame (the laptop's pepin-vslam) is not"
                " saying whether RTAB-Map has placed the cart since its start, so the pose may"
                " be the one it saved at its last shutdown. Either the node is down or"
                " respawning (docker logs pepin-vslam | grep 'rtabmap frame'), or it runs code"
                " from before this word existed, which ros/laptop.sh vslam restarts on the"
                f" checkout. {PLACEMENT_SWITCH}",
            )
        if placement.placed:
            return Check("placed", True, placement.how())
        return Check("placed", False, f"{placement.how()}: {PLACEMENT_REMEDY}")
