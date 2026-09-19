"""A pose measured on one machine and fused on another: the message, and the gate that takes it.

No ROS and no robot — the odometry trail is a real :class:`pepin.timeline.OdomHistory` — so
every rule the board applies to the laptop's camera poses is decided here, in milliseconds:
what travels, what a carry does to it, what is too old to believe, and what happens when the
camera is the only sense left.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.fusion import ODOM_XY_FLOOR_M, ODOM_YAW_FLOOR_RAD, PoseMeasurement
from pepin.measurements import (
    GRAPH_FLOOR_XY_M,
    GRAPH_FLOOR_YAW_DEG,
    MEASUREMENT_MAX_AGE_S,
    REMOTE_FLOOR_XY_M,
    REMOTE_FLOOR_YAW_DEG,
    MeasurementGate,
    RemoteMeasurement,
    compose,
    graph_measurement,
    inverse,
)
from pepin.odometry import Pose2D
from pepin.sources import CAMERA, CONTACT, DEPTH, GRAPH, LIDAR, SourceRegistry
from pepin.timeline import OdomHistory

SOMEWHERE = Pose2D(1.0, 2.0, 0.5)  # a place with a heading, so a carry rotates as it must
SURE = np.diag([0.04**2, 0.04**2, math.radians(2.0) ** 2])
VAGUE = np.diag([0.30**2, 0.30**2, math.radians(20.0) ** 2])


def trail(*poses: tuple[float, Pose2D]) -> OdomHistory:
    """An odometry history with exactly these (time, pose) samples."""
    history = OdomHistory(horizon_s=5.0)
    for t, pose in poses:
        history.add(t, pose)
    return history


def rolling(start: float = 100.0, steps: int = 11, step_m: float = 0.02) -> OdomHistory:
    """The cart rolling straight ahead at 0.2 m/s, a sample every 0.1 s."""
    return trail(*((start + 0.1 * k, Pose2D(step_m * k, 0.0, 0.0)) for k in range(steps)))


def remote(
    pose: Pose2D = SOMEWHERE,
    source: str = DEPTH,
    stamp: float = 100.0,
    fit: float = 0.6,
    covariance: np.ndarray = SURE,
    map_id: str = "map1",
) -> RemoteMeasurement:
    """One measurement as the laptop makes it."""
    return RemoteMeasurement(
        x=pose.x,
        y=pose.y,
        yaw=pose.theta,
        covariance=covariance,
        source=source,
        stamp=stamp,
        fit=fit,
        map_id=map_id,
    )


def test_a_measurement_travels_as_one_message_and_comes_back_whole() -> None:
    """Everything the receiver needs to judge and fuse it is in the one JSON message: the
    place, how sure it is per direction, who measured it, when, how well it fitted and the map
    it means something on. The sender's own notes ride along and a reader ignores them."""
    sent = remote().to_json(belief_age_ms=40.0, matched_on="/map_tracked")
    back = RemoteMeasurement.from_json(sent)
    assert (back.x, back.y) == (1.0, 2.0) and abs(back.yaw - 0.5) < 1e-9
    assert back.source == DEPTH and back.stamp == 100.0 and back.map_id == "map1"
    assert back.fit == 0.6 and back.edge is False
    assert np.allclose(back.covariance, SURE)
    assert back.measurement().sigmas[0] == pytest.approx(0.04, abs=1e-3)
    assert back.text() == "depth (+1.00, +2.00, +29 deg) fit 0.60"
    assert '"matched_on": "/map_tracked"' in sent


def test_a_message_that_is_not_a_measurement_says_so() -> None:
    with pytest.raises(ValueError, match="3x3"):
        RemoteMeasurement.from_json(
            '{"x": 0, "y": 0, "yaw": 0, "covariance": [[1, 0], [0, 1]],'
            ' "source": "depth", "stamp": 1, "fit": 0.5, "map": "m"}'
        )
    with pytest.raises(KeyError):
        RemoteMeasurement.from_json('{"x": 0}')
    with pytest.raises(ValueError):
        RemoteMeasurement.from_json("not json at all")


def test_of_a_measurement_made_here_is_the_same_numbers_plus_the_map() -> None:
    """What the laptop does with what its matcher answered: nothing but name the map."""
    measured = PoseMeasurement(0.5, -0.25, 0.1, SURE, CONTACT, 12.0, 0.55, edge=True)
    travelling = RemoteMeasurement.of(measured, "map7")
    assert travelling.map_id == "map7" and travelling.edge is True
    back = travelling.measurement()
    assert (back.x, back.y, back.yaw) == (measured.x, measured.y, measured.yaw)
    assert (back.source, back.stamp, back.fit) == (CONTACT, 12.0, 0.55)
    assert np.allclose(back.covariance, measured.covariance)


def test_a_measurement_is_carried_to_the_moment_of_the_update_that_takes_it() -> None:
    """A pose measured at 100.0 and an update at 100.4: the cart rolled 8 cm in between, so the
    place it measured has moved 8 cm too. Uncarried it would pull the tracker backwards along
    the drive every time, which is a bias and not noise."""
    gate = MeasurementGate()
    gate.offer(remote(Pose2D(1.0, 2.0, 0.0), stamp=100.0), "map1")
    taken = gate.take(100.4, rolling())
    assert len(taken) == 1
    assert taken[0].x == pytest.approx(1.08, abs=1e-6) and taken[0].y == pytest.approx(2.0)
    assert taken[0].source == CAMERA, "one word from the camera, whatever measured it"
    assert taken[0].stamp == 100.4 and taken[0].fit == pytest.approx(0.6)
    assert taken[0].covariance[1, 1] > SURE[1, 1], "the carry adds the heading's lever arm"
    assert gate.take(100.4, rolling()) == [], "used once"


def test_a_measurement_too_old_or_unreachable_is_dropped_and_counted() -> None:
    """The failure of 2026-09-13: a camera pose fused as if it spoke for the moment it was used
    at. What the odometry cannot reach is dropped, and with ``carry_stale_words`` off the age does
    it as it did before 2026-09-18 — the report line says which of the two it was."""
    gate = MeasurementGate(carry_stale_words=False)
    assert gate.measurement_max_age_s == MEASUREMENT_MAX_AGE_S
    gate.offer(remote(stamp=100.0), "map1")
    assert gate.take(100.8, rolling(steps=11)) == [], "0.8 s old: past the gate"
    gate.offer(remote(stamp=99.0), "map1")  # before the trail begins
    assert gate.take(99.2, rolling()) == []
    gate.offer(remote(map_id="another"), "map1")
    assert gate.pending == (), "a pose measured on another map is not kept at all"
    gate.malformed("not json")
    line = gate.report()
    assert "stale 1" in line and "uncovered 1" in line and "elsewhere 1" in line
    assert "malformed 1 (not json)" in line and "per source: none" in line


def test_the_sources_of_one_update_are_fused_into_one_camera_word() -> None:
    """The depth band and the floor-contact line are two measurements of one moment: they are
    carried to the update's instant and fused by their information before the tracker sees
    them, so the roster has one name to switch and one word to weigh."""
    gate = MeasurementGate()
    gate.offer(remote(Pose2D(1.0, 2.0, 0.0), DEPTH, stamp=100.0), "map1")
    gate.offer(remote(Pose2D(1.1, 2.0, 0.0), CONTACT, stamp=100.1, covariance=VAGUE), "map1")
    taken = gate.take(100.2, rolling())
    assert len(taken) == 1 and taken[0].source == CAMERA
    # the sure one carries the fusion; the vague one moves it by millimetres
    assert taken[0].x == pytest.approx(1.04, abs=0.01)
    assert gate.status()["used"] == [DEPTH, CONTACT] and gate.status()["rejected"] == []
    assert gate.status()["age_ms"] == pytest.approx(100.0, abs=1.0)
    assert "per source: contact 1, depth 1" in gate.report()


def test_a_source_that_disagrees_with_the_surest_is_left_out_and_named() -> None:
    """Two camera readings of one instant that describe different places cannot both be true:
    the information filter's own gate drops the far one and the gate says which."""
    gate = MeasurementGate()
    gate.offer(remote(Pose2D(1.0, 2.0, 0.0), DEPTH, stamp=100.0), "map1")
    gate.offer(remote(Pose2D(3.0, 2.0, 0.0), CONTACT, stamp=100.0), "map1")
    taken = gate.take(100.1, rolling())
    assert len(taken) == 1 and taken[0].rejected == (CONTACT,)
    assert gate.status()["rejected"] == [CONTACT]
    assert "disagreed 1" in gate.report()


def test_the_roster_switches_the_whole_camera_off_and_keeps_its_health() -> None:
    """The gate asks the tracker's own roster: with the camera off nothing is taken (the
    measurements simply wait and are replaced), and every measurement that arrives is still
    counted as the camera's heartbeat, so the report line can say it is alive."""
    registry = SourceRegistry(enabled=(LIDAR,))
    gate = MeasurementGate(registry)
    gate.offer(remote(stamp=100.0), "map1")
    assert gate.take(100.1, rolling()) == [] and not gate.enabled()
    assert registry.health(CAMERA).verdict(100.2) == "fresh", "heard from, even while off"
    registry.enable((LIDAR, CAMERA))
    assert gate.enabled() and len(gate.take(100.1, rolling())) == 1


def test_the_camera_drives_an_update_only_when_no_scan_source_does() -> None:
    """What a dead lidar leaves. With a scan source anchoring, the measurement waits for that
    update (and rides it); with nothing anchoring, it drives an update at its own stamp, where
    the odometry is the pose it is predicted from and nothing has to be carried forward."""
    gate = MeasurementGate()
    gate.offer(remote(stamp=100.2), "map1")
    assert gate.drive(LIDAR, rolling()) is None, "the lidar drives: the measurement rides it"
    plan = gate.drive(None, rolling())
    assert plan is not None and plan.stamp == 100.2
    assert plan.odom.x == pytest.approx(0.04, abs=1e-6)
    assert plan.measurement.source == CAMERA and plan.measurement.stamp == 100.2
    assert gate.drive(None, rolling()) is None, "nothing waiting any more"
    gate.offer(remote(stamp=200.0), "map1")
    assert gate.drive(None, rolling()) is None, "and none the odometry cannot reach"


def test_a_new_map_forgets_what_was_measured_against_the_old_one() -> None:
    gate = MeasurementGate()
    gate.offer(remote(stamp=100.0), "map1")
    gate.forget()
    assert gate.pending == () and gate.take(100.1, rolling()) == []


def test_the_age_the_gate_refuses_past_is_a_live_switch() -> None:
    gate = MeasurementGate()
    assert gate.switches == (
        "measurement_max_age_s",
        "carry_stale_words",
        "self_check",
        "remote_floor_xy_m",
        "remote_floor_yaw_deg",
    )
    gate.switch("carry_stale_words", False)
    gate.switch("measurement_max_age_s", 1.0)
    gate.offer(remote(stamp=100.0), "map1")
    assert len(gate.take(100.8, rolling(steps=11))) == 1, "0.8 s is inside the new age"
    with pytest.raises(ValueError, match="not a switch"):
        gate.switch("rest_lock", True)


def standing(seconds: float = 3.0) -> OdomHistory:
    """The cart standing still, a sample every 0.1 s: a carry that adds nothing, so what is
    left between two measurements is the source's own scatter."""
    return trail(*((100.0 + 0.1 * k, Pose2D()) for k in range(int(seconds / 0.1) + 1)))


def jumpy(gate: MeasurementGate, steps: int = 20, jump_m: float = 0.24) -> list[PoseMeasurement]:
    """``steps`` measurements of a source that alternates ``jump_m`` while the cart stands and
    claims SURE (4 cm) throughout; the last update's taken measurements."""
    still, taken = standing(), []
    for k in range(steps):
        stamp = 100.0 + 0.1 * k
        gate.offer(remote(Pose2D(1.0 + (jump_m if k % 2 else 0.0), 2.0, 0.5), stamp=stamp), "map1")
        taken = gate.take(stamp, still)
    return taken


def test_a_remote_source_that_does_not_repeat_loses_its_weight() -> None:
    """The day of 2026-09-13 in one test: the camera claimed a sigma a formula gave it while
    nobody had measured how far apart two of its answers fall. Here it claims 4 cm and jumps 24
    at rest, which is a variance ratio of 24^2 / (2 * 4^2) over three degrees of freedom — 6,
    less the tenth of a percent the standing carry's own floors take off the denominator
    (:func:`pepin.fusion.odometry_covariance`: 2 mm and 0.05 deg when the odometry reports no
    motion at all) — and its covariance is multiplied by exactly that before anything is fused
    with it."""
    gate = MeasurementGate()
    taken = jumpy(gate)
    assert gate.self_check.ratio(DEPTH) == pytest.approx(5.9925, abs=0.001)
    # Widened at its own moment, then carried: the check's factor rides the claim, and the
    # carry's own cost is added after it, never multiplied by it.
    assert taken[0].covariance[0, 0] == pytest.approx(
        SURE[0, 0] * gate.self_check.ratio(DEPTH) + ODOM_XY_FLOOR_M**2
    )
    assert gate.status()["self_check"][DEPTH] == [5.993, 5.993]
    assert "self_check on: depth r 5.99 x5.99" in gate.report()


def test_the_self_check_is_a_live_switch_and_off_is_the_old_behaviour() -> None:
    """Off, the check widens nothing: what the covariance picks up on the way is the carry's
    own cost and nothing else (:func:`pepin.fusion.odometry_covariance` — here the standing
    floors alone). The ratio is still measured, so the report line shows what the switch would
    do before it is moved."""
    gate = MeasurementGate(
        remote_floor_xy_m=0.0, remote_floor_yaw_deg=0.0
    )  # the floor is its own switch
    gate.switch("self_check", False)
    taken = jumpy(gate)
    assert taken[0].covariance[0, 0] == pytest.approx(SURE[0, 0] + ODOM_XY_FLOOR_M**2)
    assert taken[0].covariance[2, 2] == pytest.approx(SURE[2, 2] + ODOM_YAW_FLOOR_RAD**2)
    assert gate.self_check.ratio(DEPTH) == pytest.approx(5.9925, abs=0.001)
    assert gate.status()["self_check"][DEPTH] == [5.993, 1.0]


def test_each_remote_source_is_judged_on_its_own_record() -> None:
    """A wild depth and a steady contact line arrive together: the contact line's covariance is
    untouched, and its record is the same whether depth is there or not."""
    gate, alone = MeasurementGate(), MeasurementGate()
    still = standing()
    for k in range(20):
        stamp = 100.0 + 0.1 * k
        wild = Pose2D(1.0 + (0.24 if k % 2 else 0.0), 2.0, 0.5)
        gate.offer(remote(wild, source=DEPTH, stamp=stamp), "map1")
        for g in (gate, alone):
            g.offer(remote(SOMEWHERE, source=CONTACT, stamp=stamp), "map1")
            g.take(stamp, still)
    assert gate.self_check.inflation(CONTACT) == 1.0
    assert gate.self_check.inflation(DEPTH) == pytest.approx(5.9925, abs=0.001)
    assert gate.self_check.ratio(CONTACT) == alone.self_check.ratio(CONTACT)


def test_a_new_map_forgets_the_self_check_too() -> None:
    """A source's repeatability was measured against poses on the old map; it starts over."""
    gate = MeasurementGate()
    jumpy(gate)
    gate.forget()
    assert gate.self_check.ratio(DEPTH) == 1.0 and gate.status()["self_check"] == {}


def test_a_remote_word_is_never_fused_tighter_than_the_measured_floor() -> None:
    """2026-09-13: the camera's word was 10 cm and 2-5 deg off the lidar's truth while a fan on a
    wall claimed 1 deg; fused on that claim the pose spun. The gate raises a claim under the
    floor to the floor and leaves a wider claim alone; zero floors change nothing."""
    import math

    import numpy as np

    from pepin.fusion import PoseMeasurement
    from pepin.measurements import MeasurementGate

    gate = MeasurementGate(remote_floor_xy_m=0.08, remote_floor_yaw_deg=5.0)
    tight = PoseMeasurement(
        0.0, 0.0, 0.0, np.diag([0.01**2, 0.01**2, math.radians(1.0) ** 2]), "depth", 0.0, 0.9
    )
    wide = PoseMeasurement(
        0.0, 0.0, 0.0, np.diag([0.3**2, 0.3**2, math.radians(20.0) ** 2]), "depth", 0.0, 0.9
    )
    floored = gate._floored(tight)
    assert math.isclose(math.sqrt(floored.covariance[0, 0]), 0.08) and math.isclose(
        math.sqrt(floored.covariance[1, 1]), 0.08
    )
    assert math.isclose(math.degrees(math.sqrt(floored.covariance[2, 2])), 5.0)
    assert gate._floored(wide) is wide, "a wider claim is left alone"
    gate.switch("remote_floor_xy_m", 0.0)
    gate.switch("remote_floor_yaw_deg", 0.0)
    assert gate._floored(tight) is tight, "zero floors change nothing"


def test_a_graph_word_is_the_localisation_itself_in_the_one_map_frame() -> None:
    """RTAB-Map's optimised map frame IS `map`, so a localisation it publishes is already a place on
    our map: the word carries it through untouched. Everything that used to sit between the two —
    an anchor learned from one seating, a tie fitted over pairs, a per-node table — was a cure for
    two frames, and there is one frame now."""
    place = Pose2D(2.0, 1.0, math.pi / 2)
    word = graph_measurement(place, 10.0, "map-a")
    assert word.source == GRAPH
    assert (word.x, word.y) == pytest.approx((2.0, 1.0))
    assert word.yaw == pytest.approx(math.pi / 2)
    assert word.map_id == "map-a" and word.stamp == 10.0


def test_rtabmap_s_own_covariance_rides_on_the_word_and_is_floored_and_never_replaced() -> None:
    """What the registration against the recognised node was worth is RTAB-Map's to say, and it is
    the one number here that is a measurement — but its own arithmetic is useless in both
    directions (706 m of sigma with no closure, 8 mm right after one), so the floor is raised under
    it and a claim already wider than the floor is left exactly as it came."""
    tight = graph_measurement(
        Pose2D(1.0, 0.0, 0.0), 1.0, "map-a", measured=np.diag([1e-4, 1e-4, 1e-6])
    )
    cov = np.asarray(tight.covariance, dtype=float)
    assert cov[0, 0] == pytest.approx(GRAPH_FLOOR_XY_M**2), "a graph sure of itself is not"
    assert cov[2, 2] == pytest.approx(math.radians(GRAPH_FLOOR_YAW_DEG) ** 2)

    measured = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.5]])
    wide = graph_measurement(Pose2D(1.0, 0.0, 0.0), 1.0, "map-a", measured=measured)
    kept = np.asarray(wide.covariance, dtype=float)
    assert kept == pytest.approx(measured), "a registration measured as wide stays wide, arms too"


def test_compose_and_inverse_are_each_other_s_undoing() -> None:
    """The two planar helpers the whole graph path is built of: a pose put through a transform
    and then through its inverse is the pose again, whatever the angles."""
    frame, pose = Pose2D(1.5, -2.5, 2.0), Pose2D(-0.25, 0.75, -1.25)
    back = compose(inverse(frame), compose(frame, pose))
    assert (back.x, back.y, back.theta) == pytest.approx((pose.x, pose.y, pose.theta))


def test_a_graph_gate_takes_nothing_until_the_roster_switches_it_on() -> None:
    """The graph rides the same gate the camera does, under a name of its own: with `graph`
    missing from the sources flag its word is never taken, and naming it is the whole switch."""
    registry = SourceRegistry(enabled=(LIDAR,))
    gate = MeasurementGate(registry, name=GRAPH)
    history = trail((1.0, Pose2D()), (2.0, Pose2D()))
    word = graph_measurement(Pose2D(1.0, 0.0, 0.0), 1.5, "map-a")
    assert gate.offer(word, "map-a") and gate.pending == (GRAPH,)
    assert gate.take(1.6, history) == [], "the roster has not switched it on"

    registry.enable((LIDAR, GRAPH))
    assert gate.offer(word, "map-a")
    (taken,) = gate.take(1.6, history)
    assert taken.source == GRAPH, "one word, under the gate's own name"
    assert (taken.x, taken.y) == pytest.approx((1.0, 0.0))
    assert not gate.offer(word, "another-map"), "a word about another map is evidence here"


def test_a_graph_measurement_claims_no_more_than_the_remote_floor() -> None:
    """RTAB-Map's own numbers are useless in both directions (706 m of standard deviation with no
    closure, 8 mm right after one), so the word is worth the floor MEASURED FOR THE GRAPH and no
    more -- 0.20 m / 8 deg since 2026-09-16, not the camera's 0.08 m it used to borrow, because a
    graph word sat 22-23 cm and 12 deg off the lidar's pose in motion."""
    m = graph_measurement(Pose2D(1.0, 0.0, 0.0), 1.0, "map-a")
    cov = np.asarray(m.covariance, dtype=float)
    assert GRAPH_FLOOR_XY_M > REMOTE_FLOOR_XY_M, "the graph is not as good as the camera"
    assert GRAPH_FLOOR_YAW_DEG > REMOTE_FLOOR_YAW_DEG
    assert cov[0, 0] == pytest.approx(GRAPH_FLOOR_XY_M**2)
    assert cov[1, 1] == pytest.approx(GRAPH_FLOOR_XY_M**2)
    assert cov[2, 2] == pytest.approx(math.radians(GRAPH_FLOOR_YAW_DEG) ** 2)
    assert np.count_nonzero(cov - np.diag(np.diagonal(cov))) == 0
    # and it travels: the board reads back exactly what was measured
    back = RemoteMeasurement.from_json(m.to_json())
    assert back.source == GRAPH and back.x == pytest.approx(1.0)


def test_a_late_word_is_carried_and_widened_instead_of_dropped() -> None:
    """2026-09-17, measured on the tapes (scratch/word_age.py): the camera's words arrive 272-364 ms
    old at the median and p90 607-802 ms against a 500 ms budget, so 892 of 6037 `depth` and 916 of
    5694 `contact` words on tape 0374 alone were thrown away on arrival. A word the odometry can
    still reach is evidence; being late costs it covariance, which the carry already adds."""
    gate = MeasurementGate()
    gate.offer(remote(stamp=100.0, source=DEPTH), "map1")
    taken = gate.take(100.8, rolling(steps=11))  # 0.8 s: the tape's own worst decile
    assert len(taken) == 1, "0.8 s late and reachable: carried, not dropped"
    assert "stale" not in gate.report()
    # The pose moved with the cart: 0.8 s of the trail is 16 cm of travel.
    fresh = MeasurementGate()
    fresh.offer(remote(stamp=100.0, source=DEPTH), "map1")
    now = fresh.take(100.0, rolling(steps=11))
    # (the step is the cart's OWN forward motion, so in the map it lands along the word's heading)
    assert len(now) == 1
    assert math.hypot(taken[0].x - now[0].x, taken[0].y - now[0].y) == pytest.approx(0.16, abs=0.01)
    # ...and it arrives WIDER than it was sent, by what the odometry over that carry costs. Shown on
    # a word that claims more than the remote floor (the graph's 0.20 m), because under the floor
    # the carry's own cost is invisible — which is the whole argument for carrying: 0.8 s of this
    # cart's odometry is worth less than the floor every remote word already carries.
    wide = np.diag([0.2**2, 0.2**2, math.radians(8.0) ** 2])
    for gate_, at in ((MeasurementGate(), 100.8), (MeasurementGate(), 100.0)):
        gate_.offer(remote(stamp=100.0, source=DEPTH, covariance=wide), "map1")
        sigmas = np.sqrt(np.diag(gate_.take(at, rolling(steps=11))[0].covariance))
        if at == 100.8:
            late = sigmas
        else:
            prompt = sigmas
    assert (late >= prompt).all() and late[0] > prompt[0], (late, prompt)
    assert late[0] - prompt[0] < 0.02, "and the carry costs centimetres, not decimetres"


def test_the_odometry_the_board_holds_is_the_only_bound_on_the_age() -> None:
    """The age budget is gone as a tunable: what the trail cannot reach is still refused, and that
    refusal is a fact about the board's own memory rather than a constant."""
    gate = MeasurementGate()
    gate.offer(remote(stamp=99.0), "map1")  # before the trail begins
    assert gate.take(99.2, rolling()) == []
    assert "uncovered 1" in gate.report()
    gate.offer(remote(stamp=100.0), "map1")
    assert len(gate.take(101.0, rolling(steps=11))) == 1, "inside the trail: taken at any age"


def test_the_old_age_cut_is_one_switch_away() -> None:
    """CLAUDE.md rule 19: the behaviour of before 2026-09-18 is reachable without a restart."""
    gate = MeasurementGate()
    gate.switch("carry_stale_words", False)
    gate.offer(remote(stamp=100.0), "map1")
    assert gate.take(100.8, rolling(steps=11)) == [] and "stale 1" in gate.report()
    gate.switch("carry_stale_words", True)
    gate.offer(remote(stamp=100.0), "map1")
    assert len(gate.take(100.8, rolling(steps=11))) == 1
