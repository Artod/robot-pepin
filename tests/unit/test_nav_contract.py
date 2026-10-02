"""Regression guards for the navigation configuration and the operator scripts.

Each assertion is a lesson paid for on the robot: the value it pins was the cause of a failed run.
Python sources are read through their syntax trees (``source_facts``): a contract holds on what a
file calls, imports and assigns, never on a substring a comment could satisfy.
"""

import ast
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import source_facts as sf
import yaml

from pepin.flags import FlagSet, load_knobs, load_table, with_knobs

REPO = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
ROBOT_LAUNCH = "ros/pepin_bringup/launch/robot.launch.py"
NAV_LAUNCH = "ros/pepin_bringup/launch/nav.launch.py"
VSLAM_LAUNCH = "ros/pepin_bringup/launch/vslam.launch.py"
NODES = "ros/pepin_bringup/pepin_bringup"


def _node_named(launch: ast.Module, name: str) -> ast.Call:
    """The launch's ``Node(...)`` call whose ``name`` keyword is the literal ``name``."""
    return next(
        c
        for c in sf.calls_to(launch, "Node")
        if ast.unparse(sf.keywords(c).get("name", ast.Constant(None))) == repr(name)
    )


def _live_table(node: str) -> FlagSet:
    """A node's whole live table: its FLAGS, then its config knobs (config/knobs.json)."""
    return with_knobs(load_table(REPO / NODES / f"{node}.py"), load_knobs(node))


def _started_by_describe(launch: ast.Module) -> set[str]:
    """The names in the list the launch's ``_describe`` returns: what it starts."""
    describe = next(
        n for n in launch.body if isinstance(n, ast.FunctionDef) and n.name == "_describe"
    )
    returned = [n.value for n in ast.walk(describe) if isinstance(n, ast.Return)][-1]
    assert isinstance(returned, ast.List)
    return {ast.unparse(e) for e in returned.elts if isinstance(e, ast.Name)}


def _rtabmap(table: str) -> dict[str, object]:
    """One of vslam.launch.py's RTAB-Map tables as a dict. Since 2026-09-19 there is ONE — the
    ``RTABMAP`` table that serves every situation — beside ``LOCALIZE`` and
    ``TF_ODOMETRY_VARIANCE``, so a contract names the one it means instead of the union of every
    dict literal in the file."""
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))[table]))


def _case_blocks(script: str) -> dict[str, str]:
    """A shell script's top-level ``case`` branches by label: ``on)`` up to the next label."""
    labels = list(re.finditer(r"^    ([\w|]+|\*)\)", script, re.M))
    ends = [m.start() for m in labels[1:]] + [len(script)]
    return {m.group(1): script[m.end() : end] for m, end in zip(labels, ends, strict=True)}


def _p(node: str) -> dict:  # type: ignore[type-arg]
    block = PARAMS[node]
    if (
        "ros__parameters" not in block
    ):  # the costmaps nest one level deeper: node/node/ros__parameters
        block = block[node]
    return block["ros__parameters"]  # type: ignore[no-any-return]


def test_the_controller_runs_in_a_frame_that_does_not_cross_the_wifi() -> None:
    """Both of these read ``map`` from 2026-09-06 (a slipping wheel fed odom a metre of motion and
    Nav2 saw progress for 58 s) until 2026-09-22, when ``map -> odom`` became the laptop's and the
    board's radio spikes (0.4-1.2 s) started leaving the controller's costmap without its own
    frame. The slip is answered inside odom now — rf2o is the EKF's odom3 — and the two values
    must stay EQUAL: the behaviours build their poses in ``local_frame`` and hand them to a
    collision checker that reads them as cells of the local costmap."""
    assert _p("local_costmap")["global_frame"] == "odom"
    assert _p("behavior_server")["local_frame"] == _p("local_costmap")["global_frame"]
    assert "odom3: odom_laser" in (REPO / "ros/params/ekf.yaml").read_text(), (
        "what makes odom trustworthy on carpet: the lidar's own scan-to-scan twist in the filter"
    )


def test_stuck_is_declared_within_seconds_and_short_goals_stay_reachable() -> None:
    checker = _p("controller_server")["progress_checker"]
    assert checker["required_movement_radius"] <= 0.15
    assert checker["movement_time_allowance"] <= 6.0


def test_turning_in_place_counts_as_progress() -> None:
    """2026-09-08: a 70 deg pivot at the base was called 'no progress' and answered by reversing."""
    checker = _p("controller_server")["progress_checker"]
    assert checker["plugin"].endswith("PoseProgressChecker")
    assert 0.0 < checker["required_movement_angle"] <= 0.6


def test_the_planner_can_always_plan_out_of_where_the_cart_stands() -> None:
    """A planning disc made the START cell impassable beside the sofa and the planner failed four
    times in a row (2026-09-08). Clearance is bought with cost, never with a body the cart has not
    got: no disc, no negative padding, the real polygon in both costmaps."""
    for costmap in ("global_costmap", "local_costmap"):
        parameters = _p(costmap)
        assert "robot_radius" not in parameters, (
            f"{costmap}: a disc makes a parked cart unplannable"
        )
        assert "footprint" in parameters, f"{costmap} must carry the real shape"
        assert parameters["footprint_padding"] == 0.0, (
            f"{costmap}: padding is a phantom hull — nav2's 0.01 default put a centimetre in front "
            "of a bumper that is 0.0625 m from base_link, and a pose touching a table read as a "
            "collision"
        )


def test_arriving_means_arriving() -> None:
    """A manipulator will have to reach an object on the table, so 'reached' must mean reached:
    the checker used to accept 0.15 m, which is 2.4x the whole bumper offset."""
    checker = _p("controller_server")["general_goal_checker"]
    assert checker["xy_goal_tolerance"] <= 0.10
    assert checker["yaw_goal_tolerance"] <= 0.20
    # NavFn's own tolerance cannot go below the inflation's inscribed band or a mark standing
    # against furniture becomes unplannable ("Failed to create plan with tolerance of 0.050000").
    assert 0.15 <= _p("planner_server")["GridBased"]["tolerance"] <= 0.30


def test_a_tof_return_is_marked_across_its_whole_cone() -> None:
    """One cell is a mark the planner squeezes past; the sensor cannot say where in the cone.

    The fan that feeds the ObstacleLayers says it by putting the one measured distance on EVERY
    beam of the cone, and by having enough beams that no costmap cell inside the arc is skipped
    at the sensor's own ceiling.
    """
    from pepin.tof_horizon import cone_beams

    facts = sf.assignments(sf.tree(f"{NODES}/tof_bridge.py"))
    fov, cell = float(facts["_FIELD_OF_VIEW_RAD"]), float(facts["_COSTMAP_CELL_M"])
    assert cell == _p("local_costmap")["resolution"], "the fan is drawn on that costmap's cells"
    for ceiling in (0.9586, 0.5680, 0.5858):  # the three mounts of config/tof.json
        beams = cone_beams(ceiling, fov, cell)
        assert ceiling * fov / (beams - 1) <= cell, "a gap in the cone at its widest"


def test_each_tof_owns_its_own_layer() -> None:
    """2026-09-08: one shared layer let the dead front sensor clear the side sensors' marks.

    Still one layer per sensor after 2026-09-21, and now it is an ObstacleLayer with a grid of
    its own per whisker — the same property by a different mechanism, and the one that made the
    change safe: a shared ObstacleLayer would have let the front sensor's clearing rays scrub
    the side sensors' marks exactly as the shared probability grid did.
    """
    layers = [name for name in _p("local_costmap")["plugins"] if name.startswith("tof_")]
    assert len(layers) == 3, "the three ToF must not share a grid"
    for sensor, layer in zip(("front", "left", "right"), layers, strict=True):
        block = _p("local_costmap")[layer]
        assert layer == f"tof_{sensor}_scan_layer", layers
        assert block["plugin"].endswith("ObstacleLayer")
        assert block["observation_sources"].split() == [f"tof_{sensor}_scan"]
        assert block[f"tof_{sensor}_scan"]["topic"] == f"/tof/{sensor}/scan"


def test_the_whiskers_do_not_run_in_the_plugin_with_the_unbounded_loop() -> None:
    """2026-09-21, the second RangeSensorLayer defect and the reason the ToF left it.

    ``range_sensor_layer.cpp:362-369`` clamps ``bx0``/``by0`` at zero and ``bx1``/``by1`` at the
    grid's size, then walks ``for (unsigned int x = bx0; x <= (unsigned int)bx1; x++)``: a cone
    entirely off the grid's left or bottom edge leaves the upper bounds NEGATIVE, the cast makes
    them about 4e9, and one thread spins for ever holding the costmap's mutex. It needs one jump
    of the pose in ``map`` between a reading's stamp and the update — a tracker restart, a
    relocalisation, the cart lifted — which happens after the reading has left, so no publisher
    can hold it off. Reproduced by restarting the relocaliser (tid 191 of the Nav2 container:
    415 s of CPU in 700 s). So: no costmap may LIST a RangeSensorLayer, and the
    whiskers are ObstacleLayers fed by a fan.
    """
    for costmap in ("local_costmap", "global_costmap"):
        parameters = _p(costmap)
        for name in parameters["plugins"]:
            assert not parameters[name]["plugin"].endswith("RangeSensorLayer"), (
                f"{costmap}.{name}: the unbounded loop is one pose jump away"
            )


def test_a_whisker_clears_its_cone_without_marking_a_ring_at_the_ceiling() -> None:
    """The lesson /depth_scan paid for, applied to the fan. "Nothing within my trusted range" is
    +inf on every beam; Nav2's laserScanValidInfCallback turns that into a point at the scan's
    ``range_max`` minus a tenth of a millimetre, so if ``obstacle_max_range`` reached the fan's
    own range_max the CLEARING point would be marked instead — a lethal ring at the ceiling. The
    fan's range_max is the sensor's ceiling (tof_bridge), so every marking range here must sit
    below it while the raytrace range may reach it."""
    for sensor, ceiling in (("front", 0.9586), ("left", 0.5680), ("right", 0.5858)):
        source = _p("local_costmap")[f"tof_{sensor}_scan_layer"][f"tof_{sensor}_scan"]
        assert source["inf_is_valid"] is True, "an inf is how a whisker says 'clear'"
        assert source["marking"] is True and source["clearing"] is True
        assert source["obstacle_max_range"] < ceiling - 0.04, sensor
        assert ceiling - 0.01 <= source["raytrace_max_range"] <= ceiling + 0.01, sensor
        assert source["data_type"] == "LaserScan"
        # No sensor_frame: the cone's origin is the sensor, and the fan carries that frame.
        assert "sensor_frame" not in source, sensor


def test_a_whisker_never_erases_what_the_lidar_or_the_camera_saw() -> None:
    """Each whisker clears in a grid of its own, and that grid reaches the master by MAXIMUM
    (combination_method 1, every layer here) — so a cone that says "free" can never lower a
    lethal cell another sensor wrote. The order in the list is the other half of it: the ToF
    stand after the three sensor layers, so they are the last to write and still cannot."""
    plugins = _p("local_costmap")["plugins"]
    for sensor in ("front", "left", "right"):
        assert _p("local_costmap")[f"tof_{sensor}_scan_layer"]["combination_method"] == 1
    tof = [name for name in plugins if name.startswith("tof_")]
    assert plugins.index("lidar_layer") < plugins.index(tof[0])
    assert plugins.index("camera_layer") < plugins.index(tof[0])


def test_the_camera_grid_layer_ships_off_in_both_costmaps_and_draws_by_maximum() -> None:
    """The camera grid (2026-09-24): a StaticLayer in each costmap that only DRAWS the volume's
    current columns (depth_fusion's grid_out). Off as shipped, just before inflation, on its own
    topics, the global one with Nav2's updates. use_maximum is costmap-wide and the static_layer
    reads it too, which changes nothing only while that layer is the FIRST of a costmap that does
    not track unknown space (it writes into a master reset to 0 and max(0, c) is c) — both held
    here."""
    for costmap, topic in (
        ("local_costmap", "/camera_grid"),
        ("global_costmap", "/camera_grid_map"),
    ):
        params = _p(costmap)
        layer, plugins = params["camera_grid_layer"], params["plugins"]
        assert layer["plugin"] == "nav2_costmap_2d::StaticLayer", costmap
        assert layer["map_topic"] == topic and layer["map_subscribe_transient_local"] is True
        assert plugins.index("camera_grid_layer") == plugins.index("inflation_layer") - 1
        assert plugins[-1] == "inflation_layer"
        assert params["use_maximum"] is True, f"{costmap}: FREE cells must not erase other marks"
        assert params.get("track_unknown_space", False) is False, costmap
    assert _p("global_costmap")["plugins"][0] == "static_layer"
    assert "static_layer" not in _p("local_costmap")["plugins"]


def test_the_tof_whiskers_serve_the_local_costmap_only() -> None:
    """2026-09-21: the ToF are short whiskers for the controller's map. In the global costmap
    they bought a room-scale plan nothing and were the worst amplifier of the RangeSensorLayer
    wedge (transform_tolerance 1.0 s x 15 Hz = 15x per update cycle), which hung planner_server's
    activation on 4 of 7 board starts."""
    plugins = _p("global_costmap")["plugins"]
    assert not [name for name in plugins if name.startswith("tof_")], plugins


def test_a_pivot_the_cart_cannot_make_is_preferred_less_than_an_arc() -> None:
    """The rear corners sweep 0.407 m; RPP checks 7.03 deg per step, so the speed sets the sweep."""
    follow = _p("controller_server")["FollowPath"]
    assert follow["rotate_to_heading_angular_vel"] <= 0.5
    assert follow["rotate_to_heading_min_angle"] >= 1.0
    # A Spin judged over 2 s of yaw needs the whole swing circle and is refused before it starts.
    assert _p("behavior_server")["simulate_ahead_time"] <= 1.0


def test_the_operator_scripts_parse_and_keep_their_safety_lines() -> None:
    for script in (
        "stop.sh",
        "goto.sh",
        "clip.sh",
        "lib.sh",
        "map.sh",
        "feature.sh",
        "laptop.sh",
        "board.sh",
        "flags.sh",
        "preflight.sh",
        "ready.sh",
        "reset_world.sh",
        "restart.sh",
    ):
        subprocess.run(["bash", "-n", str(REPO / "ros" / script)], check=True)
    stop = (REPO / "ros/stop.sh").read_text()
    # The red button (tests/unit/test_stop.py drives it against fakes): the goal server's
    # cancel, the base's own stop believed from its state stream, and when either is not
    # confirmed the cmd_vel producers killed BEFORE the stop that is believed. The only kills
    # are those producers: Nav2's composed container and the board's measured motion. Never a
    # board restart: it zeroes the odometry the map is tied to.
    assert 'pepin.goal_link --port "$GOAL_PORT" --timeout 3 cancel' in stop
    assert "pepin.red_button" in stop
    acted = "\n".join(ln for ln in stop.splitlines() if not ln.lstrip().startswith("#"))
    kills = [ln.strip() for ln in acted.splitlines() if "pkill" in ln or "kill -KILL" in ln]
    assert any("pkill -9 -f '__node:=nav2_container'" in k for k in kills), kills
    assert all(
        "__node:=nav2_container" in k or "pepin_motion.pid" in k or '"$pid"' in k for k in kills
    ), kills
    assert acted.index("__node:=nav2_container") < acted.rindex("base && STILL=1")
    assert "systemctl" not in acted and "goto_ros" not in acted
    goto = (REPO / "ros/goto.sh").read_text()
    assert re.search(r"trap .*EXIT", goto)
    # Every signal cancels and never restarts the board (2026-09-29): the red button is typed.
    for sig, status in (("INT", 130), ("HUP", 129), ("TERM", 143)):
        assert f"trap 'on_signal {status}' {sig}" in goto, sig
    cancel = goto[goto.index("cancel() {") : goto.index("\n}\n", goto.index("cancel() {"))]
    assert cancel.count('goal_link --timeout "$CANCEL_S" cancel && return 0') == 2
    assert not re.search(r'/stop\.sh"', goto), "Ctrl-C must not run stop.sh"
    run = (REPO / "ros/run.sh").read_text()
    assert "--rm" not in run  # a stopped container must keep its log for the next start to save


def test_the_recoveries_are_not_gated_by_a_condition_that_refuses_the_real_codes() -> None:
    """START_OCCUPIED and a boxed-in controller are exactly what the round robin is for, and
    WouldA*RecoveryHelp answers FAILURE for them."""
    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    recovery = tree.find(".//ReactiveFallback[@name='RecoveryFallback']")
    assert recovery is not None
    parent = next(p for p in tree.iter() if recovery in list(p))
    assert not [c for c in parent if c.tag == "Fallback"], (
        "a gate stands in front of the recoveries"
    )
    planner_branch = tree.find(".//RecoveryNode[@name='ComputePathToPose']")
    assert planner_branch is not None
    assert planner_branch.find(".//WouldAPlannerRecoveryHelp") is None


def test_the_chosen_planner_is_the_only_planner() -> None:
    """No NavFn fallback: when the footprint planner says the cart does not fit, a point planner
    used to squeeze a 12 cm disc through the gap over the toes (run 0113). A refusal goes to the
    recovery round robin and waits for the world to change."""
    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    branch = tree.find(".//RecoveryNode[@name='ComputePathToPose']")
    assert branch is not None
    first = next(iter(branch))
    assert first.tag == "ComputePathToPose" and first.get("planner_id") == "{selected_planner}"
    assert not [n for n in tree.iter("ComputePathToPose") if n.get("planner_id") == "GridBased"]
    assert int(branch.get("number_of_retries")) >= 5, "a blocked path is waited out, not given up"


def test_the_planners_buy_a_berth_with_cost_not_with_walls() -> None:
    """NavFn and Smac2D plan a point robot the size of the inscribed band (6 cm: base_link sits at
    the front edge of a 55 cm cart) and passed a person's shins at 6 cm (2026-09-09). A wall-sized
    band would strand every docked start, so the berth is cost: expensive to cross, still
    crossable when there is no other way. nav2's NavFn has no cost weight, so Smac2D carries it.
    The global costmap follows a moving person at 2 Hz. Its inflation falls off steeply since
    2026-09-23 (10.0, was 2.0): the drives run on Hybrid-A*, which checks the whole footprint,
    and the shallow 2.0 kept half a metre around every piece of furniture expensive,
    so its plans hunted troughs across the flat (Artem's call); the point planners' berth is
    narrower for it."""
    assert _p("planner_server")["Smac2D"]["cost_travel_multiplier"] >= 5.0
    assert "cost_factor" not in _p("planner_server")["GridBased"], "nav2's NavFn has no such knob"
    inflation = _p("global_costmap")["inflation_layer"]
    assert (inflation["inflation_radius"], inflation["cost_scaling_factor"]) == (0.55, 5.0)
    assert _p("global_costmap")["update_frequency"] >= 2.0
    follow = _p("controller_server")["FollowPath"]
    assert follow["max_allowed_time_to_collision_up_to_carrot"] >= 0.7
    local = _p("local_costmap")["inflation_layer"]
    assert follow["inflation_cost_scaling_factor"] == local["cost_scaling_factor"]
    assert follow["cost_scaling_dist"] <= local["inflation_radius"]


def test_the_tof_layers_never_stall_either_costmap() -> None:
    """A whisker that goes quiet must never be able to stop the robot: a sensor can leave the
    bus, a cone can be dropped by the layer's own message filter while TF is catching up, and
    tof_bridge is restarted on its own (ros/board.sh kick). So the scan sources carry no
    expected_update_rate (a buffer given a rate calls itself stale and Nav2 answers every goal
    with "Costmap timed out waiting for update", 2026-09-07)."""
    for sensor in ("front", "left", "right"):
        layer = _p("local_costmap")[f"tof_{sensor}_scan_layer"]
        assert layer[f"tof_{sensor}_scan"]["expected_update_rate"] == 0.0


def test_the_whiskers_are_fed_to_a_layer_that_drops_what_it_cannot_place() -> None:
    """The Nav2 wedge of 2026-09-21. tf2's canTransform blocks the WHOLE timeout on any failure
    and RangeSensorLayer calls it once per message with the message's own stamp, so three layers
    at 15 Hz against a 0.3 s (local) and 1.0 s (global) tolerance amplify 4.5x and 15x: the
    backlog outgrows the drain, every message ages past the 10 s TF cache, the first costmap
    update never ends and planner_server hangs in Activating (scratch/nav2_hang/wedge_gain.py).
    The tolerances stay where they are — stability would need one under 67 ms, below this
    robot's own TF latency — and the answer is the CONSUMER: an ObstacleLayer's message filter
    drops what it cannot place instead of blocking on it. For one day the publisher guarded it
    too (a tf_gate and dynamic mounts); both came out on 2026-09-22 with the range layers they
    were written for."""
    assert _p("local_costmap")["transform_tolerance"] == 0.3
    assert _p("global_costmap")["transform_tolerance"] == 1.0
    source = (REPO / NODES / "tof_bridge.py").read_text()
    assert "from tf2_ros import StaticTransformBroadcaster\n" in source, (
        "the mounts leave on /tf_static alone: the gate's TF listener cost 20-37 % of an A53 core,"
        " and a mount on /tf with the reading's stamp made the ObstacleLayer's filter drop the fan"
    )


def test_the_board_s_stack_starts_behind_a_router_that_accepts() -> None:
    """A cold start of the board: `docker run` returns before rmw_zenohd accepts, and the stack
    started behind a router that was still binding is what delayed /tf_static by 157 s."""
    router = (REPO / "board/pepin-zrouter.service").read_text()
    assert "ExecStartPost=" in router and "/dev/tcp/127.0.0.1/7447" in router


def test_two_controllers_and_only_the_footprint_planner_may_plan_a_reverse() -> None:
    """FollowPath never reverses and may pivot; FollowPathRS follows the cusps of a Hybrid-A*
    plan and, as RPP demands, gives up rotate-to-heading for it. The lattice stays forward-only."""
    cs = _p("controller_server")
    assert cs["controller_plugins"] == [
        "FollowPath",
        "FollowPathRS",
        "FollowPathMPPI",
        "FollowPathShim",
        "FollowPathGraceful",
        "FollowPathDWB",
    ]
    assert cs["FollowPath"].get("allow_reversing", False) is False
    assert cs["FollowPathRS"]["allow_reversing"] is True
    assert cs["FollowPathRS"]["use_rotate_to_heading"] is False
    same = {
        k: v
        for k, v in cs["FollowPathRS"].items()
        if k not in ("allow_reversing", "use_rotate_to_heading")
    }
    base = {
        k: v
        for k, v in cs["FollowPath"].items()
        if k not in ("allow_reversing", "use_rotate_to_heading")
    }
    assert same == base, (
        "the reversing twin drifted from FollowPath (no YAML anchors: rcl cannot parse them)"
    )
    assert _p("planner_server")["Lattice"]["allow_reverse_expansion"] is False


def test_mppi_follows_every_planner_within_the_base_caps_and_is_held_to_the_heading() -> None:
    """MPPI (the goal server's `controller` flag) samples only commands the velocity smoother
    passes, steps its model at the controller's own period, checks the true rectangle, rides out
    the same WiFi stall as the RPP pair, and ends the drive on the yaw-checking goal checker."""
    import ast

    cs = _p("controller_server")
    m = cs["FollowPathMPPI"]
    vs = _p("velocity_smoother")
    assert m["vx_max"] == vs["max_velocity"][0]
    assert m["vx_min"] == vs["min_velocity"][0]
    assert m["wz_max"] == vs["max_velocity"][2]
    assert m["ax_max"] == vs["max_accel"][0] and m["ax_min"] == vs["max_decel"][0]
    assert m["model_dt"] == pytest.approx(1.0 / cs["controller_frequency"])
    assert m["CostCritic"]["consider_footprint"] is True
    assert m["transform_tolerance"] == cs["FollowPath"]["transform_tolerance"]
    src = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    followers = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FOLLOWERS"
    )
    controller, checker = followers["mppi"]
    assert controller in cs["controller_plugins"]
    assert checker in cs["goal_checker_plugins"] and cs[checker]["yaw_goal_tolerance"] <= 0.20


def test_the_shim_wraps_the_reversing_rpp_and_turns_only_at_the_goal() -> None:
    """FollowPathShim is FollowPathRS verbatim inside Nav2's RotationShimController: it turns the
    cart to the mark's heading once inside the goal tolerance, never to the path at the start
    (a Hybrid plan may begin with a reverse cusp), and ends on the yaw-checking goal checker."""
    import ast

    cs = _p("controller_server")
    shim, rs = cs["FollowPathShim"], cs["FollowPathRS"]
    assert shim["plugin"] == "nav2_rotation_shim_controller::RotationShimController"
    assert shim["primary_controller"] == rs["plugin"]
    assert shim["rotate_to_goal_heading"] is True
    assert shim["angular_dist_threshold"] > math.pi
    assert {k: v for k, v in shim.items() if k in rs and k != "plugin"} == {
        k: v for k, v in rs.items() if k != "plugin"
    }, "the shim's RPP drifted from FollowPathRS (no YAML anchors: rcl cannot parse them)"
    src = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    followers = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FOLLOWERS"
    )
    assert followers["rpp_shim"] == ("FollowPathShim", "general_goal_checker")


def test_shim_mppi_drives_on_the_shim_and_parks_on_mppi_both_held_to_the_heading() -> None:
    """The flag value that hands a goal over: it starts on the shim pair, parks on the MPPI pair,
    both controllers exist, and both end on the yaw-checking goal checker (so the goal server's
    pivot, which follows only the position-only checker, never runs after it)."""
    import ast

    cs = _p("controller_server")
    tree = ast.parse((REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text())
    tables = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("FOLLOWERS", "PARKERS")
    }
    followers, parkers = tables["FOLLOWERS"], tables["PARKERS"]
    assert followers["shim_mppi"] == followers["rpp_shim"]
    assert parkers["shim_mppi"] == followers["mppi"]
    flags = load_table(REPO / NODES / "goal_server.py")
    assert set(followers) | set(parkers) <= set(flags.flag("controller").choices)
    for controller, checker in (*followers.values(), *parkers.values()):
        assert controller in cs["controller_plugins"]
        assert checker in cs["goal_checker_plugins"]
    for _controller, checker in parkers.values():
        assert cs[checker]["yaw_goal_tolerance"] <= 0.20


def test_graceful_and_dwb_stay_inside_the_base_caps_check_the_footprint_and_hold_the_heading() -> (
    None
):
    """The A/B controllers beside MPPI: the velocity smoother's caps, DWB's obstacle critic on the
    true rectangle (BaseObstacle would check base_link's cell, the bumper), Graceful forward-only
    (its allow_backward reverses towards any target behind the bumper, blind), and both ended by
    the yaw-checking goal checker, since both turn to the heading themselves."""
    import ast

    cs = _p("controller_server")
    vs = _p("velocity_smoother")
    g, d = cs["FollowPathGraceful"], cs["FollowPathDWB"]
    assert (
        g["v_linear_max"] == vs["max_velocity"][0] and g["v_angular_max"] == vs["max_velocity"][2]
    )
    assert g["allow_backward"] is False and g["initial_rotation"] is True
    assert g["slowdown_radius"] <= g["max_lookahead"]
    assert d["max_vel_x"] == vs["max_velocity"][0] and d["min_vel_x"] == vs["min_velocity"][0]
    assert d["max_vel_theta"] == vs["max_velocity"][2]
    assert d["acc_lim_x"] == vs["max_accel"][0] and d["decel_lim_x"] == vs["max_decel"][0]
    assert "ObstacleFootprint" in d["critics"] and "BaseObstacle" not in d["critics"]
    assert d["xy_goal_tolerance"] == cs["general_goal_checker"]["xy_goal_tolerance"]
    src = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    followers = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FOLLOWERS"
    )
    assert followers["graceful"] == ("FollowPathGraceful", "general_goal_checker")
    assert followers["dwb"] == ("FollowPathDWB", "general_goal_checker")


def test_every_planner_the_goal_server_offers_exists_with_its_controller() -> None:
    import ast

    src = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    tree = ast.parse(src)
    catalogue = next(
        ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PLANNERS"
    )
    planners = _p("planner_server")["planner_plugins"]
    controllers = _p("controller_server")["controller_plugins"]
    for name, (planner, controller) in catalogue.items():
        assert planner in planners, (name, planner)
        assert controller in controllers, (name, controller)


def test_the_footprint_planner_plans_the_cart_and_backs_out_only_briefly() -> None:
    """Hybrid-A* checks the true polygon at every heading (the point planners' 6 cm band is not
    the cart). Reeds-Shepp lets it leave a dock in reverse inside the plan; the analytic
    expansion is kept short so no reverse arc ever crosses a room (the lattice, run 0027)."""
    h = _p("planner_server")["Hybrid"]
    assert h["plugin"].endswith("SmacPlannerHybrid")
    assert h["motion_model_for_search"] == "REEDS_SHEPP"
    assert h["reverse_penalty"] >= 2.0
    assert h["analytic_expansion_max_length"] <= 1.0
    assert 0.2 <= h["minimum_turning_radius"] <= 0.4
    assert h["allow_unknown"] is True


def test_the_cart_drives_at_its_one_speed_and_nothing_clips_it() -> None:
    """ONE speed, config/base.json's max_wheel_speed_m_s: the base server's wheel ceiling, and
    what Nav2 is given at launch over every controller's and the smoother's linear limit
    (pepin.speed.NAV2_SPEED, nav.launch.py), under the axis caps the base and the C++ bridge
    clamp at. Every tape until 2026-09-09 sat at 0.20 m/s because two Nav2 numbers said so; a
    controller left out of the table would be such a number again."""
    from pepin.deployment import BASE_MAX_ANGULAR_RAD_S, BASE_MAX_LINEAR_M_S
    from pepin.geometry import BaseConfig
    from pepin.speed import NAV2_SPEED, SPEED_RANGE_M_S, check_speed, file_value, nav2_overrides

    cfg = json.loads((REPO / "config/base.json").read_text())
    axes = (cfg["max_speed_m_s"], cfg["max_yaw_rate_rad_s"])
    assert axes == (BASE_MAX_LINEAR_M_S, BASE_MAX_ANGULAR_RAD_S)
    robot = sf.dict_items(sf.tree(ROBOT_LAUNCH))
    assert robot["max_linear_m_s"] == {"BASE_MAX_LINEAR_M_S"}, "the C++ bridge keeps its 0.25 cap"
    assert SPEED_RANGE_M_S[1] == BASE_MAX_LINEAR_M_S, "ros/speed.sh never asks past the bridge"
    speed = check_speed(BaseConfig.from_json(REPO / "config/base.json").max_wheel_speed_m_s)
    for plugin in _p("controller_server")["controller_plugins"]:
        assert any(p.name.startswith(f"{plugin}.") for p in NAV2_SPEED), f"{plugin}: own speed"
    for param in NAV2_SPEED:
        file_value(PARAMS, param)  # KeyError: a name Nav2 would ignore in silence
    run = nav2_overrides(speed, PARAMS)  # what Nav2 runs: the file under the launch's overrides
    for param in NAV2_SPEED:
        assert param.held(run[param.node][param.name]) == pytest.approx(param.sign * speed)
    smoother = _p("velocity_smoother")
    for array in ("max_velocity", "min_velocity"):
        assert run["velocity_smoother"][array][1:] == smoother[array][1:], "the rest is the file's"
    loads = {
        ast.unparse(sf.keywords(c)["name"]): ast.unparse(sf.keywords(c)["parameters"])
        for c in sf.calls_to(sf.tree(NAV_LAUNCH), "ComposableNode")
        if "parameters" in sf.keywords(c)
    }
    for node in {p.node for p in NAV2_SPEED}:
        assert loads[repr(node)] == f"[params, speed[{node!r}]]", "the overrides after the file"
    follow = _p("controller_server")["FollowPath"]
    assert follow["max_lookahead_dist"] >= follow["lookahead_time"] * speed
    assert smoother["max_accel"][0] >= 0.5 and smoother["max_decel"][0] <= -1.0


def test_the_controller_stops_for_what_stands_in_its_path_and_never_for_what_it_touches() -> None:
    """RPP's collision check is on — with it off the cart drove into a person (runs 0090-0091).
    Its current-pose rule deadlocked the cart parked against the printer (run 0087); that is
    removed at the source: an 8 cm contact band around the hull that neither the lidar's hull
    filter nor the ToF ranges may mark, so nothing the cart is parked against ever lands in the
    outline's own costmap cells."""
    from pepin.footprint import CONTACT_BAND_M, HULL, hull_box

    assert _p("controller_server")["FollowPath"]["use_collision_detection"] is True
    resolution = _p("local_costmap")["resolution"]
    assert 1.5 * resolution <= CONTACT_BAND_M, "a mark just outside the band must not share a cell"
    robot = sf.assignments(sf.tree(ROBOT_LAUNCH))
    assert robot["HULL"] == "hull_box()", "the lidar hull filter must use the contact band"
    tof = sf.assignments(sf.tree(f"{NODES}/tof_bridge.py"))
    assert tof["_MIN_RANGE_M"] == "CONTACT_BAND_M", "ToF readings inside the band must be dropped"
    box = hull_box()
    assert box["max_x"] == pytest.approx(HULL.front_m + CONTACT_BAND_M)
    # A controller that cannot move hands over at once; the tree replans (2026-10-02, was 3 s).
    assert _p("controller_server")["failure_tolerance"] == 0.0


def test_a_blocked_retreat_gives_up_within_seconds() -> None:
    """The rear is unsensed: a BackUp that has not covered its distance in about twice its
    nominal time is pushing something and must fail, not push for the default 10 s (run 0087:
    10-20 s per push into a box)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    smoother = _p("velocity_smoother")
    reverse_cap = abs(smoother["min_velocity"][0])  # every BackUp is clamped to this
    moves = [(n, "backup_dist", "backup_speed", reverse_cap) for n in tree.iter("BackUp")]
    moves += [
        (n, "dist_to_travel", "speed", smoother["max_velocity"][0])
        for n in tree.iter("DriveOnHeading")
    ]
    assert moves
    for node, dist_key, speed_key, cap in moves:
        speed = min(float(node.get(speed_key)), cap)  # what the wheels really get
        nominal = float(node.get(dist_key)) / speed
        allowance = node.get("time_allowance")
        assert allowance is not None, f"{node.tag} {node.attrib} relies on the 10 s default"
        assert nominal < float(allowance) <= 2.0 * nominal + 2.0, (node.attrib, nominal)
    for spin in tree.iter("Spin"):
        assert spin.get("time_allowance") is not None, f"Spin {spin.attrib} relies on the default"
    assert reverse_cap >= 0.15, "a retreat at 0.075 m/s timed out every progress check"


def test_a_new_object_is_the_cell_the_beam_found_and_nothing_more() -> None:
    """No ring, no second observation source: what the static map cannot explain is a lidar
    return, and the lidar's own layer marks it at the cell it came from and clears it with its
    own rays. The rings that stood here until 2026-09-14 were an overfit to a standing person's
    feet (a 70 cm tape gap read 58 cm lethal-to-lethal, four minutes of recoveries against 36 s
    with them off) and they were redundant besides: every mark they published landed in this
    same layer, at a cell `scan` had already marked from the same return."""
    from pepin.footprint import COSTMAP_CELL_M

    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        assert params["resolution"] == COSTMAP_CELL_M
        layer = params["lidar_layer"]
        assert layer["observation_sources"].split() == ["scan"]
        assert "dynamic" not in layer, f"{costmap}: the ring's source is gone"
        assert layer["scan"]["marking"] is True and layer["scan"]["clearing"] is True


def test_the_nav2_footprint_is_the_hull() -> None:
    """Nav2 carries the polygon as a string in two costmaps; both are pepin.footprint.HULL."""
    import ast

    from pepin.footprint import HULL

    for costmap in ("local_costmap", "global_costmap"):
        polygon = [tuple(p) for p in ast.literal_eval(_p(costmap)["footprint"])]
        assert polygon == HULL.polygon(), costmap


def test_the_stop_reflex_has_a_budget() -> None:
    """A person who appeared 0.51 m ahead at 0.30 m/s was passed to the controller 0.3 s later,
    refused only 0.20 m before contact and stopped 0.14 m short (run 0142). The reflex lives on
    the board: local costmap tick, RPP's projection, the smoother's braking, and the tree must
    not wipe a fresh mark every second."""
    local = _p("local_costmap")
    assert local["update_frequency"] >= 5.0
    follow = _p("controller_server")["FollowPath"]
    v = follow["desired_linear_vel"]
    assert follow["max_allowed_time_to_collision_up_to_carrot"] * v >= 0.40, "look 0.4 m ahead"
    assert _p("velocity_smoother")["max_decel"][0] <= -1.5
    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    forget = next(
        n
        for n in tree.iter("RateController")
        if any(c.get("name") == "ForgetStaleObstacles" for c in n)
    )
    assert float(forget.get("hz")) <= 0.25, "a fresh ToF mark was wiped 0.1 s after it appeared"


def test_the_whole_nav2_starts_by_itself_with_the_goal_server_beside_it() -> None:
    """One machine, one manager: Nav2 on the Mac activates its own nodes, and nothing is left
    of the split's laptop-driven bring-up of a board half."""
    nav = sf.dict_items(sf.tree(NAV_LAUNCH))
    assert nav["autostart"] == {"True"}
    server = sf.tree(f"{NODES}/goal_server.py")
    assert "next_transition" not in sf.calls(server) and "ChangeState" not in sf.imported(server)


def test_a_goal_is_judged_on_the_transform_rtabmap_s_correction_composes() -> None:
    """The rule lives in pepin.watch.GoalGate; the node supplies one reading — the age of
    map -> base_link — beside the placement word (test_one_localiser holds that rule)."""
    server = sf.tree(f"{NODES}/goal_server.py")
    assert {"GoalGate", "Readiness"} <= sf.imported(server), "the rule is pepin's"
    assert "TfLookup" in sf.imported(server) and "self._tf.transform" in sf.calls(server)
    assert sf.assignments(server)["MAP_FRAME"] == "'map'"
    assert sf.assignments(server)["BASE_FRAME"] == "'base_link'"
    assert "self._gate.verdict" in sf.calls(server)
    flags = load_table(REPO / NODES / "goal_server.py")
    assert all(flags.flag(name).live for name in flags.names)
    assert "self._switches.state" in sf.calls(server), "and it is printed in the node's own line"


def test_the_recorder_is_its_own_node_beside_the_goal_server() -> None:
    """The recorder is a node of its own, started with Nav2, and reads the pose the goal server
    republishes on /pose; the goal server only sends it a command."""
    nav = sf.tree(NAV_LAUNCH)
    assert "pepin_bringup.run_recorder" in sf.strings(nav)
    assert "TfLookup" not in sf.imported(sf.tree(f"{NODES}/run_recorder.py")), "one listener"
    server = sf.tree(f"{NODES}/goal_server.py")
    assert "RunRecorder" not in sf.calls(server) and "RunRecorder" not in sf.imported(server)
    assert not any("curl" in s for s in sf.strings(server))
    assert {"RUN_COMMAND_TOPIC", "RUN_STATUS_TOPIC"} <= sf.names(server)


def test_the_static_layers_read_the_map_rtabmap_frame_relays_and_no_pgm_is_served() -> None:
    """One map: rtabmap_frame relays RTAB-Map's grid onto /map and both costmaps' static layers
    read THAT. The topic is written as a literal in both places and held equal here; no pgm is
    served at all."""
    params = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
    layers = [
        params[costmap][costmap]["ros__parameters"]["static_layer"]
        for costmap in ("local_costmap", "global_costmap")
    ]
    assert [layer["map_topic"] for layer in layers] == ["/map", "/map"]
    assert all(layer["map_subscribe_transient_local"] for layer in layers), "latched, or it waits"
    node = sf.tree("ros/pepin_bringup/pepin_bringup/rtabmap_frame.py")
    assert "/map" in {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }, "rtabmap_frame must carry the same literal"
    launch = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    assert "nav2_map_server" not in launch and '"map_server"' not in launch


def test_one_recorder_writes_a_drive_not_two() -> None:
    """ros/goto.sh started ros/tools/session_logger.py for every drive while the run recorder was
    already subscribed to the same topics: two rclpy processes turning the same 10 Hz LaserScan
    into Python objects. The recorder beside Nav2 tapes every goal; goto starts no recorder."""
    script = (REPO / "ros/goto.sh").read_text()
    assert "session_logger" not in script


def test_goto_ends_every_helper_it_started_whatever_ends_it() -> None:
    """bash defers a trap until the running foreground command returns, and three goto.sh
    survived their SIGTERM on 2026-09-13, each holding a watcher. So nothing of a drive is in
    the foreground: the goal is a background job waited on (a trapped signal interrupts the
    wait), the traps are set before any helper starts, the film and the streams run in process
    groups of their own and end by themselves when goto.sh is gone, and `finish` runs exactly
    once (tests/unit/test_goal_link.py sends INT, HUP and TERM against fakes)."""
    script = (REPO / "ros/goto.sh").read_text()
    assert "trap finish EXIT\n" in script and "trap 'on_signal 143' TERM" in script
    assert script.index("trap finish EXIT") < script.index('bash "$HERE/clip.sh"')
    assert 'goal_link --log "$LOG" go "$@" &\nGOAL=$!\nwait "$GOAL"' in script
    assert 'if [ "$FINISHED" = 1 ]; then return 0; fi' in script, "finish must not run twice"
    assert "os.setpgrp()" in script and 'kill -TERM -- "-$STREAMS"' in script
    assert 'kill -TERM "$CLIP"' in script and 'wait "$CLIP"' in script


def test_the_numbered_tape_says_which_clock_named_it() -> None:
    """The recorder runs in a container on UTC while the board's shell, the laptop and every
    ros/goto.sh file are on local time: a bare 220039 was read as a drive four hours later than
    it was (2026-09-13). The stamp is UTC and carries the letter that says so."""
    recorder = sf.tree(f"{NODES}/run_recorder.py")
    assert "time.gmtime" in sf.calls(recorder), "the stamp is UTC on purpose, not by accident"
    assert any(text == "Z" for text in sf.strings(recorder)), "and the name says which clock"


def test_the_camera_slam_owns_the_one_map_frame() -> None:
    """World R's one frame: RTAB-Map's map frame IS ``map`` and RTAB-Map owns ``map -> odom``."""
    vslam = sf.tree(VSLAM_LAUNCH)
    params = sf.dict_items(vslam)
    assert params["publish_tf"] == {"False", "True"}, (
        "the SLAM node broadcasts map -> odom; the visual odometry node does not"
    )
    assert _rtabmap("RTABMAP")["map_frame_id"] == "map", "one frame, since 2026-09-19"
    rtabmap = sf.keywords(_node_named(vslam, "rtabmap"))
    assert ast.unparse(rtabmap["namespace"]) == "'rtabmap'", (
        "its relative 'map' output must not land on /map"
    )
    assert "pepin_bringup.camera_stream" in sf.strings(vslam)
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "vslam.launch.py" in laptop and "config:/ws/config" in laptop
    # the tuned camera+lidar set ("F+G2", 2026-09-25) is the table's own since 2026-09-28: the
    # visual half of Reg/Strategy 2 seeds the closure, and the proximity search is the parked A/B's
    table = _rtabmap("RTABMAP")
    assert table["RGBD/LoopClosureIdentityGuess"] == "false"
    assert {
        k: table[k]
        for k in (
            "RGBD/ProximityGlobalScanMap",
            "RGBD/ProximityMergedScanCovFactor",
            "RGBD/MaxLoopClosureDistance",
            "RGBD/ProximityOdomGuess",
        )
    } == {
        "RGBD/ProximityGlobalScanMap": "true",
        "RGBD/ProximityMergedScanCovFactor": "0.1",
        "RGBD/MaxLoopClosureDistance": "1.0",
        "RGBD/ProximityOdomGuess": "true",
    }


def test_the_laptop_containers_restart_on_their_own_and_never_ask_the_board() -> None:
    """The two node containers restart on their own, and nothing in laptop.sh talks to the
    board: the script sources lib.sh (a connect timeout, one ssh master) for the helpers only."""
    laptop = (REPO / "ros/laptop.sh").read_text()
    for container in ('"$NAV"', "pepin-vslam"):
        line = next(ln for ln in laptop.splitlines() if f"docker run -d --name {container} " in ln)
        assert "--restart unless-stopped" in line, container
    assert '. "$HERE/lib.sh"' in laptop and "SITE=" not in laptop
    acted = "\n".join(ln for ln in laptop.splitlines() if not ln.lstrip().startswith("#"))
    assert "ssh " not in acted and "/etc/default/pepin-ros" not in acted


def test_the_laptop_halves_create_the_names_flags_sh_looks_for() -> None:
    """ros/flags.sh finds a node's container by its name (pepin.deployment.node_host): the names
    the table lists are the names the launches create, and laptop.sh lets a container leave
    properly before replacing it."""

    assert _started_by_describe(sf.tree(VSLAM_LAUNCH))
    vslam = sf.tree(VSLAM_LAUNCH)
    assert ast.unparse(sf.keywords(_node_named(vslam, "rtabmap"))["namespace"]) == "'rtabmap'"
    assert _node_named(vslam, "foxglove_bridge") is not None
    for module in ("camera_stream", "depth_stream", "contact_scan", "goal_server"):
        node = sf.tree(f"{NODES}/{module}.py")
        assert f"super().__init__('{module}')" in sf.unparsed(node, ast.Call), module
    nav = sf.tree(NAV_LAUNCH)
    composed = {ast.unparse(sf.keywords(c)["name"]) for c in sf.calls_to(nav, "ComposableNode")}
    assert "'lifecycle_manager_navigation'" in composed
    container = sf.calls_to(nav, "respawned_container")[0]
    assert ast.unparse(container.args[0]) == "'nav2_container'"
    laptop = (REPO / "ros/laptop.sh").read_text()
    # The stop itself is ros/lib.sh's now (one way to stop a container, one window); what this
    # contract still owns is that nothing here removes a container without stopping it first.
    assert "pepin_remove_container pepin-vslam" in laptop
    acted = "\n".join(ln for ln in laptop.splitlines() if not ln.lstrip().startswith("#"))
    assert "docker rm -f" not in acted and "docker stop" not in acted
    # `docker stop` sends the container's stop signal: SIGINT is the one the launch answers by
    # shutting its nodes down (SIGTERM it answers by cancelling itself and the nodes are
    # SIGKILLed mid-write). A run wrapped over several lines is read as one command.
    runs = [
        c for c in sf.shell_commands(laptop) if "docker run -d --name " in c and "zrouter" not in c
    ]
    runs = [c for c in runs if "rmw_zenohd" not in c]
    assert len(runs) == 2, "the navigation container, the SLAM container"
    for command in runs:
        assert "--stop-signal SIGINT" in command, command


def test_the_camera_is_a_depth_sensor_scaled_by_the_lidar() -> None:
    """The laptop turns the camera's frames into depth images (a network on CPU, the lidar sets
    the scale), RTAB-Map builds its grid from the lidar AND that depth per node, the cloud is drawn
    by the operator's Foxglove connected to the laptop, and the image carries the weights."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.depth_stream" in sf.strings(vslam)
    table = _rtabmap("RTABMAP")
    assert table["subscribe_depth"] is False and table["subscribe_sensor_data"] is True
    assert "('depth/image', '/camera/depth')" in sf.unparsed(vslam, ast.Tuple)
    assert table["Grid/Sensor"] == "0", (
        "the grid starts on the scan and follows the snapshots at run time (pepin.graphmode):"
        " with both sensors the mono depth flattened the lidar tracker's match (2026-09-19)"
    )
    assert table["Grid/3D"] == "false", "the grid the board is handed is 2D"
    assert {"camera", "depth", "pack", "rtabmap", "frame", "foxglove"} <= _started_by_describe(
        vslam
    )
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert "/camera/depth" in sf.strings(node)
    # The correction is the library's pipeline, run whole; the node owns no copy of a stage.
    assert {"standard_pipeline", "AffineLaw", "FrameContext"} <= sf.imported(node)
    assert "self._pipeline.run" in sf.calls(node) and "standard_pipeline" in sf.calls(node)
    assert not {"beam_pairs", "apply_affine", "drop_edges", "floor_anchor"} & sf.imported(node)
    image = (REPO / "ros/Dockerfile.laptop").read_text()
    assert (
        "whl/cpu" in image and "HF_HUB_OFFLINE=1" in image and "ros-jazzy-foxglove-bridge" in image
    )
    assert "Depth-Anything-V2-Metric-Indoor-Small-hf" in image
    layout = json.loads((REPO / "ros/foxglove/pepin_nav.json").read_text())
    # RTAB-Map's own grid fights the static map for the floor: off by default in the nav view
    assert layout["configById"]["3D!nav"]["topics"]["/rtabmap/map"]["visible"] is False
    # the 3D view: the cloud, the path, the scan and the robot; no grid lies over the voxels
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    shown = {t for t, c in room["topics"].items() if c.get("visible")}
    # THE PLANNER'S OWN MAP IS SHOWN, and only one grid per plane. The owner watched the fused
    # surface over the LOCAL costmap and could not see why a path went through a chair: the path
    # is planned on the global costmap over /map, and both were hidden (2026-09-18). So those two
    # are on, and every other occupancy grid — the local costmap's window on the same cells,
    # RTAB-Map's own grid, and the volume's two other cross-sections — stays a click away,
    # because three grids in one horizontal plane fight each other for depth.
    # /map is RTAB-Map's grid, the one the global costmap's static layer reads; the board's
    # relocalizer and its /map_tracked went with the board's Nav2 (2026-10-01).
    assert {"/map", "/global_costmap/costmap", "/plan"} <= shown
    assert "/local_costmap/costmap" not in shown
    assert not {"/map_tracked", "/tracker_pose"} & set(room["topics"])
    assert not {"/map_lidar", "/map_camera", "/rtabmap/map"} & set(room["topics"]), (
        "the volume's own layers and RTAB-Map's namespaced grid went with World R"
    )
    laptop = (REPO / "ros/laptop.sh").read_text()
    vslam_run = next(
        c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-vslam" in c
    )
    assert "-p 8765:8765" in vslam_run


def test_the_camera_s_depth_reaches_the_costmap_and_its_frame_follows_the_graph() -> None:
    """The depth folded onto the plane goes to the board as /depth_scan and is wired into the
    local costmap's camera layer (which ships off, see the test below); the camera stamps frames
    with the board's capture time; and the graph owns no frame of its own on this robot — its
    optimised map frame IS `map`, RTAB-Map broadcasts map -> odom itself, and rtabmap_frame
    publishes no transform at all."""
    layer = _p("local_costmap")["camera_layer"]
    assert "depth_scan" in layer["observation_sources"].split()
    source = layer["depth_scan"]
    assert source["topic"] == "/depth_scan" and source["data_type"] == "LaserScan"
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert "/depth_scan" in sf.strings(node) and "depth_to_scan" in sf.calls(node)
    camera = sf.tree(f"{NODES}/camera_stream.py")
    assert "capture_time" in sf.calls(camera) and "cv2.VideoCapture" not in sf.calls(camera)
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.rtabmap_frame" in sf.strings(vslam)
    nav = sf.tree(NAV_LAUNCH)
    assert not any("odom_to_rtabmap" in s for s in sf.strings(nav) | sf.names(nav))
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert "TransformBroadcaster" not in sf.imported(frame), "it broadcasts nothing"
    assert "('map', 'rtabmap')" not in sf.unparsed(frame, ast.Tuple), "one frame"
    # RTAB-Map's odometry is the EKF's own, in every situation
    assert _rtabmap("RTABMAP")["odom_frame_id"] == "odom"
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    # the camera's marks reach the costmap the PLANNER reads, which is the one now on screen
    assert room["topics"]["/global_costmap/costmap"]["visible"]


SENSOR_LAYERS = ("lidar_layer", "camera_layer", "contact_layer")


def test_one_layer_per_sensor_so_a_demo_can_switch_one_off_live() -> None:
    """The three sensor layers stand in both costmaps in the order they write into the master
    grid, each an ObstacleLayer with its own ``enabled``: that is the switch behind
    ``ros2 param set /local_costmap/local_costmap camera_layer.enabled false``, a camera-only or
    lidar-only drive without a restart (CLAUDE.md rule 19). One shared obstacle_layer could not
    be switched per sensor, and its one grid let the lidar's rays clear the camera's marks."""
    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        plugins = params["plugins"]
        assert "obstacle_layer" not in plugins, f"{costmap}: the shared layer is split"
        sensors = [name for name in plugins if name in SENSOR_LAYERS]
        assert sensors == list(SENSOR_LAYERS), f"{costmap}: {plugins}"
        tof = [name for name in plugins if name.startswith("tof_")]
        if tof:  # the whiskers stand in the local costmap only
            assert plugins.index(sensors[-1]) < plugins.index(tof[0]), "sensors before the ToF"
        assert plugins[-1] == "inflation_layer", "inflation is always last"
        if "static_layer" in plugins:
            assert plugins[0] == "static_layer"
        for name in SENSOR_LAYERS:
            layer = params[name]
            assert layer["plugin"].endswith("ObstacleLayer"), name
            assert isinstance(layer["enabled"], bool), f"{costmap}.{name} must be switchable"
    topics = {
        name: _p("local_costmap")[name][_p("local_costmap")[name]["observation_sources"].split()[0]]
        for name in SENSOR_LAYERS
    }
    assert topics["lidar_layer"]["topic"] == "/scan"
    assert topics["camera_layer"]["topic"] == "/depth_scan"
    assert topics["contact_layer"]["topic"] == "/contact_scan"


def test_a_dead_sensor_cannot_stall_a_costmap_that_the_others_still_feed() -> None:
    """A source given an expected_update_rate declares its buffer stale when it goes quiet, the
    layer stops being current and every goal dies with "Costmap timed out waiting for update"
    (2026-09-07, the ToF's clock jump). With the sensors in separate layers that would be one
    silent laptop stalling the board's controller, so every source is explicitly 0.0 and the
    costmap keeps working with any subset of the sensors — down to none."""
    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        for name in SENSOR_LAYERS:
            layer = params[name]
            sources = layer["observation_sources"].split()
            assert sources, f"{costmap}.{name} has no source"
            for source in sources:
                assert layer[source]["expected_update_rate"] == 0.0, f"{costmap}.{name}.{source}"
        for sensor in ("front", "left", "right"):
            scan = params.get(f"tof_{sensor}_scan_layer")  # the local costmap's, since 2026-09-21
            if scan is not None:
                assert scan[f"tof_{sensor}_scan"]["expected_update_rate"] == 0.0, sensor


def test_only_the_measured_layer_is_on_by_default() -> None:
    """The split gives the camera a grid of its own that the lidar can no longer scrub. Both
    camera layers ship ON since 2026-09-14: the drives of 2026-09-13 measured the table top the
    lidar cannot see (the camera marked it, the lidar-only costmap drove into it), and the
    working state had lived in a live flip that every restart of 2026-09-14 reverted. The lidar
    and the contact layers stay on; each flips live (``camera_layer.enabled false``)."""
    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        assert params["lidar_layer"]["enabled"] is True
        assert params["camera_layer"]["enabled"] is True, (
            f"{costmap}: the camera layer ships on — 2026-09-13 measured the table top the lidar"
            " cannot see, and a live flip was lost at every restart of 2026-09-14"
        )
        for sensor in ("front", "left", "right"):
            scan = params.get(f"tof_{sensor}_scan_layer")
            if scan is not None:
                assert scan["enabled"] is True, sensor


def test_a_layer_that_never_clears_forgets_a_source_that_died() -> None:
    """The other half of expected_update_rate 0.0: a buffer keeps exactly one cloud without
    observation_persistence and re-applies it at every update forever, so a silent source freezes
    a phantom (2026-09-11, verified against Nav2 1.3.12's obstacle_layer.cpp). A layer whose
    sources all mark and never clear cannot erase that phantom by any means — ClearEntireCostmap
    empties the grid for one cycle and the frozen cloud marks it again — so every source in such
    a layer must forget. A layer that does clear (the lidar's, the camera's) scrubs its own."""
    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        for name in SENSOR_LAYERS:
            layer = params[name]
            sources = layer["observation_sources"].split()
            if any(layer[source].get("clearing") for source in sources):
                continue
            for source in sources:
                persistence = layer[source].get("observation_persistence", 0.0)
                assert persistence > 0.0, f"{costmap}.{name}.{source} would freeze forever"


def test_the_contact_scan_only_marks_and_stays_off_until_it_is_measured() -> None:
    """Where the floor ends is a range the camera's own geometry measures (pepin.contact), and
    the layer that reads it marks only: the scan never says the floor BEHIND a body is free, and
    a column whose floor drifted out of the band ends on nothing — the false mark the 2 m cap
    exists for. So no clearing, no inf (it would mark a lethal ring at range_max with clearing
    off), the layer's range is the module's own cap, and the layer ships disabled in both
    costmaps until a recorded drive says the marks are real (scratch/contact_validation.py)."""
    from pepin.contact import CONTACT_MAX_RANGE

    for costmap in ("local_costmap", "global_costmap"):
        layer = _p(costmap)["contact_layer"]
        assert layer["observation_sources"].split() == ["contact_scan"]
        source = layer["contact_scan"]
        assert source["topic"] == "/contact_scan" and source["data_type"] == "LaserScan"
        assert source["sensor_frame"] == "base_link"
        assert source["marking"] is True and source["clearing"] is False
        assert source["inf_is_valid"] is False, "inf with clearing off would mark a ring at 2 m"
        assert source["obstacle_max_range"] == CONTACT_MAX_RANGE
    # The node's own cap is the same number, and its scan's range_max with it: a mark the layer
    # would have to discard is a mark nobody sees.
    assert load_knobs("contact_scan")["max_range"] == CONTACT_MAX_RANGE
    # The camera's two scans keep the rule that separates clearing from marking: depth_scan
    # clears with inf, so the node's range_max must stay above the layer's obstacle range.
    camera = _p("local_costmap")["camera_layer"]["depth_scan"]
    depth = sf.tree(f"{NODES}/depth_stream.py")
    declared = sf.calls_to(depth, "self.declare_parameter")
    # The default is the module's DEPTH_REACH_M and not a literal of its own: since 2026-09-19 the
    # scan's cap and the PUBLISHED depth's reach are one number, because they are one claim about
    # how far this network's scale is still a measurement.
    default = next(
        ast.unparse(call.args[1])
        for call in declared
        if ast.unparse(call.args[0]) == "'scan_max_range'"
    )
    assert default == "DEPTH_REACH_M", "one constant for the scan's cap and the image's reach"
    scan_max_range = ast.literal_eval(sf.assignments(depth)["DEPTH_REACH_M"])
    assert camera["inf_is_valid"] is True and camera["obstacle_max_range"] < scan_max_range, (
        "an inf ray clears to range_max: below that, every one of them marks instead"
    )
    assert load_knobs("depth_stream")["depth_reach_m"] == scan_max_range


def test_the_camera_layer_clears_from_a_frame_and_marks_from_the_volume() -> None:
    """A single depth frame is an eyewitness of what is OPEN, not evidence that something is
    THERE: on the first stereo drive SGBM's blobs on the parquet (floor pixels lifted to
    0.15-0.24 m, about one false bearing a frame) left the costmap with 100-300 lethal cells the
    lidar never saw, 115 'collision ahead' a minute and 44 recoveries (tape ros/maps/rec/0415_*).
    So the camera layer's two sources are split by what each is evidence of: /depth_scan clears
    and marks nothing, /depth_marks — the fused volume's own surface sliced around the cart
    (pepin_bringup.depth_fusion, pepin.volume_scan) — marks and clears nothing. Both costmaps,
    because the global one does not roll and keeps a phantom longest."""
    from pepin.volume_scan import MARKS_RANGE_M

    node = sf.tree(f"{NODES}/depth_fusion.py")
    topic = ast.literal_eval(sf.assignments(node)["MARKS_TOPIC"])
    assert topic == "/depth_marks"
    assert "marks_ranges" in sf.calls(node)
    assert "/depth_scan" not in sf.strings(node), "the frame's fan clears; it never marks"
    assert ast.literal_eval(sf.assignments(node)["FREE_TOPIC"]) == "/depth_free"
    assert "free_ranges" in sf.calls(node), "the clearing half comes from the same walk"
    flags = load_table(REPO / NODES / "depth_fusion.py")
    # OFF as shipped, and the measurement says why: on the parked cart of 2026-09-23 the volume
    # held an occupied column on 717 of 720 bearings, and where the camera's own frame shared a
    # bearing with a mark it agreed within 0.20 m 96 % of the time (3 % saw past it) — a clearing
    # ray stops at the first column that is not open, so there was almost nothing for it to erase
    # (scratch/one_localiser/live_fan_vs_lidar.py).
    assert flags["marks_clear"] is False, "measured first, defaulted after"
    assert flags.flag("marks_clear").live, "an A/B without a restart, as every flag here"
    for costmap in ("local_costmap", "global_costmap"):
        layer = _p(costmap)["camera_layer"]
        assert layer["observation_sources"].split() == [
            "depth_scan",
            "depth_marks",
            "depth_free",
        ], costmap
        frame, volume = layer["depth_scan"], layer["depth_marks"]
        assert frame["marking"] is False and frame["clearing"] is True, costmap
        assert volume["topic"] == topic and volume["data_type"] == "LaserScan"
        assert volume["marking"] is True and volume["clearing"] is False, costmap
        # NaN is "the volume holds nothing here" and there is no inf on this topic at all; valid,
        # an inf would become a point at range_max and MARK a ring there, as the contact layer
        # taught us.
        assert volume["inf_is_valid"] is False, costmap
        assert volume["sensor_frame"] == "base_link"
        assert volume["expected_update_rate"] == 0.0, "no source may stall a costmap by dying"
        assert volume["observation_persistence"] > 0.0, "...and a dead source must be forgotten"
        # One window for the two words of one camera, and inside the fan's own reach: a mark the
        # layer would have to discard is a mark nobody sees.
        assert volume["obstacle_max_range"] == frame["obstacle_max_range"] < MARKS_RANGE_M
        # THE THIRD SOURCE, and the reason it is a source of its own: a LaserScan cannot say
        # "clear to here" without also marking there — Nav2's ObstacleLayer marks at the END of
        # every finite range and clears up to it — so a single source that cleared a ray at 1.2 m
        # would plant a lethal cell at the frontier of knowledge, which is the defect it exists to
        # cure. Clearing only, never marking, and silent until depth_fusion's `marks_clear` is on.
        open_to = layer["depth_free"]
        assert open_to["topic"] == "/depth_free" and open_to["data_type"] == "LaserScan"
        assert open_to["marking"] is False and open_to["clearing"] is True, costmap
        assert open_to["obstacle_max_range"] == 0.0, "nothing may ever mark from this one"
        assert open_to["raytrace_max_range"] == MARKS_RANGE_M, "the fan's own reach, no further"
        assert open_to["inf_is_valid"] is False, "unknown must stay unknown"
        assert open_to["sensor_frame"] == "base_link"
        assert open_to["expected_update_rate"] == 0.0, "a silent source may not stall a costmap"


def test_the_floor_s_edge_is_a_node_of_the_kit() -> None:
    """The contact scan runs where the depth network runs — on the laptop, off /camera/depth —
    and reaches the board's costmap the way /depth_scan does. The node is the kit's: a
    newest-wins worker, live switches printed in its report line, the floor's geometry rebuilt
    only when the lean or the optics move, and a launch entry that respawns it."""
    node = sf.tree(f"{NODES}/contact_scan.py")
    assert "super().__init__('contact_scan')" in sf.unparsed(node, ast.Call)
    assert {"/contact_scan", "/camera/depth", "/camera/camera_info"} <= sf.strings(node)
    # the IMU is not subscribed here by hand: one owner of the lean for every node (LeanFeed)
    assert "/imu/data_raw" in sf.strings(sf.tree(f"{NODES}/node_kit.py"))
    assert {"FloorPlane", "contact_scan", "ContactVerdict", "LeanFeed"} <= sf.imported(node)
    assert {"FloorPlane.of", "contact_scan", "scan_from_ranges"} <= sf.calls(node)
    # the fan is /depth_scan's own, not one of its own invention
    assert {"SCAN_HALF_FOV", "SCAN_STEP"} <= sf.names(node)
    # the kit, not a copy of it: one worker thread, the tally's stages, the switches' state
    assert {"Worker", "Switches", "Tally", "spin_main"} <= sf.imported(node)
    assert "self._switches.state" in sf.calls(node) and "self._worker.stop" in sf.calls(node)
    # the live flags (CLAUDE.md rule 19), the feature's own name first
    flags = _live_table("contact_scan")
    assert flags["imu_lean"] is True, "on since the gyro's sign was verified by hand (2026-09-13)"
    assert all(flag.live for flag in flags), "every one of them takes the next frame"
    # the plane is a cache with two keys: the lean and the optics
    assert "self._plane_up" in sf.unparsed(node, ast.Attribute)
    assert "self._plane_intr != intr" in sf.unparsed(node, ast.Compare)
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.contact_scan" in sf.strings(vslam)
    assert "contact" in _started_by_describe(vslam)


def test_the_drive_ends_on_position_and_the_goal_server_turns_to_the_heading() -> None:
    """RPP with reversing cannot rotate in place: the tree's FollowPath judges xy only, and the
    goal server pivots the residual heading with the behaviour server's Spin before "done"."""
    controller = _p("controller_server")
    assert "xy_only_goal_checker" in controller["goal_checker_plugins"]
    xy_only = controller["xy_only_goal_checker"]
    assert xy_only["xy_goal_tolerance"] == controller["general_goal_checker"]["xy_goal_tolerance"]
    assert xy_only["yaw_goal_tolerance"] >= 3.14
    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    # The checker is picked with the controller; the tree's own default is the RPP pair's.
    selector = next(tree.iter("GoalCheckerSelector"))
    assert selector.get("default_goal_checker") == "xy_only_goal_checker"
    assert selector.get("topic_name") == "goal_checker_selector"
    assert all(
        n.get("goal_checker_id") == selector.get("selected_goal_checker")
        for n in tree.iter("FollowPath")
    )
    server = sf.tree(f"{NODES}/goal_server.py")
    assert ast.literal_eval(sf.assignments(server)["RPP_GOAL_CHECKER"]) == "xy_only_goal_checker"
    assert "ActionClient(self, Spin, 'spin')" in sf.unparsed(server, ast.Call)
    assert "self._pivot_to" in sf.calls(server)
    # The pivot stops where the general checker would have called the heading met.
    pivot_deg = ast.literal_eval(sf.assignments(server)["PIVOT_TOLERANCE_DEG"])
    met_deg = math.degrees(controller["general_goal_checker"]["yaw_goal_tolerance"])
    assert pivot_deg == pytest.approx(met_deg, abs=0.1)


def test_the_lidar_mount_is_published_by_both_sides_from_one_file() -> None:
    """The board's launch reads base_link -> laser from config/lidar.json itself (no launch
    argument can override the calibration), the laptop's camera node publishes the same
    transform through the same module's reader (pepin.mounts), and ros/sync.sh puts config/
    where the board's container sees it (beside the library: /ws/pepin_src/config,
    pepin.deployment.config_file)."""
    robot = sf.tree(ROBOT_LAUNCH)
    assert sf.assignments(robot)["MOUNTS"] == "Mounts.load()"
    assert sf.assignments(robot)["LASER"] == "MOUNTS.lidar.transform()"
    declared = {ast.unparse(c.args[0]) for c in sf.calls_to(robot, "DeclareLaunchArgument")}
    assert not declared & {"'laser_roll'", "'laser_yaw'"} and "MOUNT" not in sf.imported(robot)
    assert "quaternion(laser_roll, laser_pitch, laser_yaw)" in sf.unparsed(robot, ast.Call)
    camera = sf.tree(f"{NODES}/camera_stream.py")
    assert (
        "transform_from_mount('base_link', LASER_FRAME, load_lidar_mount(config_dir), stamp)"
        in sf.unparsed(camera, ast.Call)
    )
    # The same parser as the launch (pepin.mounts), but only the files whose frames this node
    # publishes: Mounts.load also reads imu.json and tof.json, and a missing or broken file of
    # a sensor the camera never publishes would crash-loop it under the launch's RESPAWN.
    assert {"load_lidar_mount", "load_camera_mounts"} <= sf.calls(camera)
    assert "Mounts.load" not in sf.calls(camera), "not the whole config directory"
    assert {"load_lidar_mount", "LASER_FRAME"} <= sf.imported(camera)
    sync = (REPO / "ros/sync.sh").read_text()
    assert '"$HERE/../config/" "root@$BOARD:/root/pepin-ros/pepin_src/config/"' in sync
    assert "--exclude 'pepin_src'" in sync, "the ros/ rsync must not delete pepin_src/config"
    run = (REPO / "ros/run.sh").read_text()
    assert '"$HERE/pepin_src:/ws/pepin_src:ro"' in run


def test_both_publishers_of_base_link_to_laser_agree_and_the_driver_turns_ccw() -> None:
    """The board's launch turns config/lidar.json into a quaternion with its own helper (a
    launch file imports as little as it can), the laptop's camera node with
    pepin.camera.quaternion_from_rpy: on the real mount (roll pi, yaw -87.5 degrees) the two
    must agree to 1e-9, or the two sides publish two lasers. The roll mirrors the LD19's scan
    once; the driver's verse stays CCW beside it, so nothing mirrors it twice."""
    from collections.abc import Callable
    from typing import cast

    from pepin.camera import quaternion_from_rpy
    from pepin.mounts import Mounts

    robot = sf.tree(ROBOT_LAUNCH)
    helper = next(
        n for n in robot.body if isinstance(n, ast.FunctionDef) and n.name == "quaternion"
    )
    namespace: dict[str, object] = {"math": math}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), ROBOT_LAUNCH, "exec"), namespace)
    launch_quaternion = cast(
        Callable[[float, float, float], tuple[float, ...]], namespace["quaternion"]
    )
    roll, pitch, yaw = Mounts.load().lidar.transform()[3:]
    assert roll == math.pi and pitch == 0.0 and yaw != 0.0
    assert launch_quaternion(roll, pitch, yaw) == pytest.approx(
        quaternion_from_rpy(roll, pitch, yaw), abs=1e-9
    )
    entries = sf.dict_items(robot)
    assert entries["lidar.rot_verse"] == {"'CCW'"}
    assert "qx" in entries["rotation.x"]  # the laser's transform takes the helper's answer


def test_the_camera_edge_has_exactly_one_publisher_on_each_side_of_the_switch() -> None:
    """The camera rides a two-servo neck. The board's C++ base bridge turns the neck's ticks of
    every state line into /neck/state and base_link -> camera_link (neck.hpp, the twin of
    pepin.neck), its geometry handed over by robot.launch.py from config/neck.json and the
    camera's link frame from config/camera.json; the laptop's camera node must then stop
    broadcasting its static copy of that edge — two publishers of one edge fight, and a static
    transform cannot be withdrawn, so that one is a launch switch (``static_camera_tf``), read
    once at start.

    No switch on the board: a neck that does not answer publishes nothing, and a rig without
    one runs ``ros/laptop.sh vslam --fixed-head``, which passes ``static_camera_tf:=true`` into
    vslam.launch.py, which hands it to the camera node alone. The board is never asked what it
    is doing: this launch talks to no one, and a guess would be the two-publisher case.
    """
    from pepin.neck import NeckConfig, bridge_parameters

    camera = sf.tree(f"{NODES}/camera_stream.py")
    # The switch is one of the camera node's flags (node_kit.Switches over its FLAGS table,
    # CLAUDE.md rule 19) and is printed in its report line — but it is declared not live: the
    # transform went out at start, and a static transform cannot be withdrawn.
    camera_flags = load_table(REPO / NODES / "camera_stream.py")
    assert "static_camera_tf" in camera_flags and not camera_flags.flag("static_camera_tf").live
    assert camera_flags["static_camera_tf"] is False, "the board's base bridge owns the edge"
    assert "Switches" in sf.imported(camera)
    report = sf.calls_to(camera, "self._switches.state")
    assert report and all(ast.unparse(sf.keywords(c)["live_only"]) == "False" for c in report), (
        "the report line carries the non-live flag too: it says which side owns the edge"
    )
    guarded = [
        n
        for n in ast.walk(camera)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self._switches.on('static_camera_tf')"
        and "camera.link_frame, camera.link" in ast.unparse(n.body)
    ]
    assert guarded, "base_link -> camera_link is broadcast only under the switch"
    assert "camera.link_frame, camera.link" not in ast.unparse(guarded[0].orelse), (
        "and not in the other branch"
    )
    kept = next(
        ast.unparse(n)
        for n in ast.walk(camera)
        if isinstance(n, ast.List) and "camera.optical, stamp" in ast.unparse(n)
    )
    assert "LASER_FRAME" in kept and "camera.link, stamp" not in kept, (
        "that edge alone goes, not the others"
    )
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "'static_camera_tf'" in {
        ast.unparse(c.args[0]) for c in sf.calls_to(vslam, "DeclareLaunchArgument")
    }
    commands = [s for s in sf.unparsed(vslam, ast.List) if "'pepin_bringup." in s]
    passes = [s for s in commands if "static_camera_tf:=" in s]
    assert len(passes) == 1 and "pepin_bringup.camera_stream" in passes[0], "the camera node only"
    # The board's side: the bridge's parameters carry the geometry and the frame, nothing else
    # publishes the edge, and no switch is left to turn it off.
    robot = sf.tree(ROBOT_LAUNCH)
    neck = next(
        ast.unparse(f)
        for f in ast.walk(robot)
        if isinstance(f, ast.FunctionDef) and f.name == "neck_parameters"
    )
    for piece in ("bridge_parameters(neck)", "MOUNTS.camera.link_frame", "list(JOINT_NAMES)"):
        assert piece in neck, piece
    assert "**neck_parameters()" in ast.unparse(robot), "the base bridge's parameters"
    assert "pepin_bringup.neck_state" not in sf.strings(robot)
    assert "'neck'" not in {
        ast.unparse(c.args[0]) for c in sf.calls_to(robot, "DeclareLaunchArgument")
    }
    bridge = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    names = [*bridge_parameters(NeckConfig.from_json(REPO / "config/neck.json"))]
    names += ["neck_camera_frame", "neck_parent_frame", "neck_joint_names", "neck_publish_hz"]
    for name in names:
        assert f'"{name}"' in bridge, f"{name}: declared by the bridge"
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    assert bringup and "LaunchConfiguration('neck')" not in sf.unparsed(bringup, ast.Call)
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "PEPIN_NECK" not in unit and "neck:=" not in unit
    assert "PEPIN_NECK" not in (REPO / "ros/feature.sh").read_text()
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "STATIC_CAMERA_TF=false\n" in laptop, "the board owns the edge by default"
    assert "--fixed-head) STATIC_CAMERA_TF=true ;;" in laptop, (
        "a flag anywhere after the subcommand"
    )
    assert any(
        '"static_camera_tf:=$STATIC_CAMERA_TF"' in command
        for command in sf.shell_commands(laptop)
        if "docker run -d --name pepin-vslam" in command
    )


def test_the_frames_are_fused_into_one_surface_beside_rtabmap_s_cloud() -> None:
    """The SLAM launch runs the fusion node; the node loads the grid from
    config/fusion.json and offers its switches as parameters; the 3D layout shows the fused
    surface and hides RTAB-Map's concatenated cloud by default (both stay available)."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.depth_fusion" in sf.strings(vslam)
    assert "fusion" in _started_by_describe(vslam)
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert sf.assignments(node)["CONFIG"] == "'/ws/config/fusion.json'"
    # The four are flags of the node's table (node_kit.Switches over pepin.flags), so ros2
    # param set reaches them and their state is printed in the report line (CLAUDE.md rule 19).
    fusion_flags = _live_table("depth_fusion")
    assert {"enabled", "min_weight", "surface_hz"} <= set(fusion_flags.names)
    assert fusion_flags.flag("surface_hz").range is not None, "a rate is bounded"
    assert "Switches" in sf.imported(node) and "self._switches.state" in sf.calls(node)
    assert {"/fusion/reset", "/fusion/surface"} <= sf.strings(node)


def test_the_volume_is_open_loop_and_no_slice_of_it_is_published() -> None:
    """pepin.worldmap in the node: /scan is integrated into the same volume the camera writes,
    the volume is the odometry's rolling window — and NOTHING localises against it. A tracker
    matching the slice it is painting has a null space it cannot see out of (2026-09-18: a cart
    with its wheels blocked walked 7 degrees and 5-7 cm in 35 minutes at fit 0.97-0.99), so the
    room's own geometry is the graph's grid and this node publishes one thing: the surface
    cloud."""
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert {"WorldMap", "PlanarMount", "ViewGate"} <= sf.imported(node)
    assert sf.assignments(node)["LIDAR_CONFIG"] == "'/ws/config/lidar.json'", "the plane is read"
    assert "/scan" in sf.strings(node)
    calls = sf.calls(node)
    assert "self._world.integrate_scan" in calls and "self._world.integrate_depth" in calls
    assert "self._world.save" not in calls, "a window painted through the odometry is not resumed"
    topics = {s for s in sf.strings(node) if s == "/map" or s.startswith("/map_")}
    # /map is READ, for its lattice alone (grid_map_topic): nothing goes out on a map's name.
    assert topics == {"/map"}, f"the volume reaches no matcher: {topics}"
    assert sf.assignments(node)["MAP_TOPIC"] == "'/map'"
    flags = load_table(REPO / NODES / "depth_fusion.py")
    # The one grid that does leave (grid_out, 2026-09-24) is an obstacle picture the costmaps'
    # camera_grid_layer draws, never a map anything seats a pose on: its own names, off as shipped.
    assert sf.assignments(node)["GRID_TOPIC"] == "'/camera_grid'"
    assert sf.assignments(node)["GRID_MAP_TOPIC"] == "'/camera_grid_map'"
    assert {"map_source", "map_hz", "lidar_map", "camera_map", "map_identity"}.isdisjoint(
        set(flags.names)
    ), "the flags that published the volume went with the publication"
    # The volume is painted in odom and nothing in the paint path reads map -> odom: no tracker
    # word, no graph bend, no snapshot (the map-frame room is on alt/volume-map-2026-10-02).
    assert {"PaintTrust", "GraphBend", "CorrectionFollower", "world_path_for"}.isdisjoint(
        sf.imported(node)
    )
    assert "/rtabmap/mapGraph" not in sf.strings(node) and "/localization_fit" not in sf.strings(
        node
    )


def test_the_cart_s_lean_is_one_thing_every_consumer_takes_from() -> None:
    """One estimator (pepin.lean through the kit's LeanFeed, off /imu/data_raw — which reaches
    the laptop in both modes, so a laptop node reads it directly and no /lean topic is needed),
    one flag name and one meaning in every node that uses it — imu_lean switches the estimator
    in all three and the poser in the two that place a frame — off until it is measured on the
    robot, and the lean in each of their report lines. The tracker is deliberately not among
    them: it runs on the board and already refuses an IMU subscription for a number the EKF
    gives it."""
    from pepin.lean import LEAN_QUALITY_FLOOR

    kit = sf.tree(f"{NODES}/node_kit.py")
    assert "/imu/data_raw" in sf.strings(kit) and "LeanEstimator" in sf.imported(kit)
    # and it starts from what the level floor measured, not from a zero it learns again per run
    assert "LevelPose" in sf.imported(kit), "the kit reads config/imu.json's level block"
    level = json.loads((REPO / "config/imu.json").read_text())["level"]
    assert len(level["gyro_bias_deg_s"]) == 3 and "note" in level
    for name in ("depth_fusion", "depth_stream", "contact_scan"):
        node = sf.tree(f"{NODES}/{name}.py")
        flags = load_table(REPO / NODES / f"{name}.py")
        assert flags["imu_lean"] is True, f"{name}: on since the gyro's sign was verified by hand"
        assert "LeanFeed" in sf.imported(node), name
        assert "/imu/data_raw" not in sf.strings(node), f"{name}: the kit owns the subscription"
        assert "self._lean.report" in sf.calls(node), f"{name}: the lean in the report line"
        # and the flag switches the estimator itself in every one of them, or two report lines
        # would print two different leans of the same body with every flag in the same state
        feed = sf.calls_to(node, "LeanFeed")
        assert len(feed) == 1, f"{name}: one lean feed"
        gyro = sf.keywords(feed[0]).get("use_gyro")
        assert gyro is not None and ast.unparse(gyro) == "self._switches.on('imu_lean')", name
        assert "self._lean.use_gyro" in sf.unparsed(node, ast.Attribute), (
            f"{name}: imu_lean is the estimator's switch live, not only at start-up"
        )
    # the poser is where the lean meets the pose, and only the two nodes that place a frame
    for name in ("depth_fusion", "depth_stream"):
        node = sf.tree(f"{NODES}/{name}.py")
        assert "self._poser.apply_lean" in sf.unparsed(node, ast.Attribute), name
    assert "apply_lean" not in sf.unparsed(sf.tree(f"{NODES}/contact_scan.py"), ast.Attribute)
    # the tape carries what an offline replay needs to run that same estimator
    recorder = sf.tree(f"{NODES}/run_recorder.py")
    assert "imu_record" in sf.imported(recorder) and "imu_record" in sf.calls(recorder)
    # the lidar's path follows the same switch: its scan is placed by the poser's pose (leaning
    # when imu_lean says so), and a revolution taken too far from level is dropped and counted
    fusion = sf.tree(f"{NODES}/depth_fusion.py")
    assert "LeanGate" in sf.imported(fusion) and "self._gate.admits" in sf.calls(fusion)
    # The scan's pose is the poser's: the odometry's, the frame the volume is painted in.
    assert "self._poser.base_in_map" in sf.calls(fusion), "the scan's pose is the poser's"
    gate = _live_table("depth_fusion").flag("lean_gate_deg")
    assert gate.live
    # and a lean gravity never voted for is no lean: one floor, in both nodes that place a
    # measurement, read by the poser so the gate and the pose make the same decision
    for name in ("depth_fusion", "depth_stream"):
        floor = _live_table(name).flag("lean_min_quality")
        assert floor.live and floor.default == LEAN_QUALITY_FLOOR, name
        attributes = sf.unparsed(sf.tree(f"{NODES}/{name}.py"), ast.Attribute)
        assert "self._poser.min_lean_quality" in attributes, name


def test_every_node_s_flags_are_one_table_the_kit_declares_and_the_report_line_prints() -> None:
    """CLAUDE.md rule 19, in one place per node: a node that has live switches builds them from
    its module-level FLAGS table (pepin.flags), which loads without the node (the README and
    ros/flags.sh read it there), prints their state in its report line, and declares no live
    parameter by hand — a switch outside the table is invisible to the tools."""
    from pepin.flags import FlagSet

    tables = {}
    for path in sorted((REPO / NODES).glob("*.py")):
        node = sf.tree(f"{NODES}/{path.name}")
        if "Switches" not in sf.imported(node):
            assert "add_on_set_parameters_callback" not in sf.calls(node), path.name
            continue
        flags = load_table(path)
        assert isinstance(flags, FlagSet) and len(flags), path.name
        tables[path.stem] = flags
        assert "self._switches.state" in sf.calls(node), f"{path.name}: the report line"
        switches = sf.calls_to(node, "Switches")
        # FLAGS itself, or flags_for(...): the same table with a depth source's own defaults;
        # with the node's config knobs (config/knobs.json) after it when the node has any
        table = ast.unparse(switches[0].args[1])
        knobs = load_knobs(path.stem)
        own = "FLAGS" if path.stem != "depth_stream" else "flags_for(source_name)"
        expected = f"with_knobs({own}, load_knobs('{path.stem}'))" if len(knobs) else own
        assert len(switches) == 1 and table == expected, (path.name, table)
        assert "self.add_on_set_parameters_callback" not in sf.calls(node), path.name
        # A name built per sensor (tof_bridge's f"{name}_x") is not a literal and cannot be
        # compared with a flag's name here; every literal one is.
        declared = {
            ast.literal_eval(c.args[0])
            for c in sf.calls_to(node, "self.declare_parameter")
            if isinstance(c.args[0], ast.Constant)
        }
        assert not declared & set(flags.names), f"{path.name}: a flag declared twice"
        assert not declared & set(knobs.names), f"{path.name}: a knob declared by hand"
        for flag in flags:
            assert flag.description, f"{path.name}: {flag.name} needs a sentence"
    assert {"depth_stream", "depth_fusion", "goal_server"} <= tables.keys()


def test_a_sensor_is_muted_where_it_is_published_and_both_bridges_know_the_same_two_names() -> None:
    """Switching a sensor off used to mean `ros/feature.sh imu off`: a restart of the whole
    board stack, a minute long, every live flag on it lost. The mute is the same test without
    the restart — the node that publishes the sensor stops publishing, live, and a consumer sees
    what a dead sensor looks like. The node is C++ and its table lives in base_bridge.py (what
    ros/flags.sh reads), so the two parameter names have to exist in both or `ros/sensor.sh mute
    imu` validates a name the running bridge does not have."""
    flags = load_table(REPO / NODES / "base_bridge.py")
    mutes = [name for name in flags.names if flags.flag(name).kind == "bool"]
    assert mutes == ["imu_publish", "odom_publish"]
    for name in mutes:
        assert flags.flag(name).live and flags[name] is True, f"{name}: on, and live, or no test"
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    for name in mutes:
        assert f'declare_parameter<bool>("{name}", true)' in cpp, f"{name} missing from the C++"
        assert f'get_parameter("{name}").as_bool()' in cpp, f"{name} read per message, not once"
    assert "switch_state()" in cpp, "the report line names the switches (CLAUDE.md rule 19)"
    # The board's bridge is the one that reads the chip, so `mute imu` has to reach it there.
    from pepin.deployment import node_host

    assert node_host("base_bridge") == ("board", "pepin-ros")


def test_odom_is_dated_by_the_encoder_read_the_neck_carries_behind_a_live_switch() -> None:
    """/odom was stamped on its arrival at the bridge while /neck/state of the same state line
    carried the encoder read: arrival ran p50 6.4 ms (max 26.4) behind it (2026-10-02). Held
    here: one LineTime per line feeds both publishers, /odom and its transform take the read
    under `odom_stamp` "encoder" (the table's default, the C++'s, read per line so the switch is
    live) and the arrival under "arrival", and the report line names the switch."""
    flag = load_table(REPO / NODES / "base_bridge.py").flag("odom_stamp")
    assert flag.default == "encoder" and set(flag.choices) == {"encoder", "arrival"}
    assert flag.live
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    assert 'declare_parameter<std::string>("odom_stamp", "encoder")' in cpp
    assert 'get_parameter("odom_stamp").as_string() != "arrival"' in cpp, "read per line"
    line = cpp[cpp.index("void on_state_line(") :]
    line = line[: line.index("\n  }\n")]
    assert "const LineTime when = line_time(*state);" in line
    assert "publish_neck(*state, when);" in line and "publish_state(*state, when);" in line
    state = cpp[cpp.index("void publish_state(const BaseState & state, const LineTime & when)") :]
    state = state[: state.index("\n  }\n")]
    assert "const rclcpp::Time stamp = encoder ? when.read : when.arrival;" in state
    assert state.count(".header.stamp = stamp;") == 2, "/odom and odom -> base_link alike"
    assert "now()" not in state, "no second clock read for the wheels"
    assert '" odom_stamp="' in cpp, "the report line names the switch"


def test_the_board_bridge_publishes_the_rest_zupt_the_ekf_fuses_behind_a_live_switch() -> None:
    """Parked on 2026-09-24 the EKF's heading crept ~5 deg/hour: odom2 fused /zupt, but beside
    RTAB-Map nothing published it (its one publisher was the tracker's slip watch). The C++
    bridge now does, from its own rest witness -- so the three ends of the wire are held together
    here: the bridge advertises `zupt` as an Odometry built by zupt.hpp, the switch is declared
    on (CLAUDE.md rule 19) and named in the report line, and the EKF's odom2 reads that topic as
    vx, vy and vyaw, which is exactly what the covariance claims."""
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    assert 'create_publisher<nav_msgs::msg::Odometry>("zupt", 5)' in cpp
    assert '#include "pepin_base_cpp/zupt.hpp"' in cpp
    assert 'declare_parameter<bool>("zupt_publish", true)' in cpp, "on by default"
    assert '" zupt_publish="' in cpp, "the report line names the switch"
    assert "rest_zupt_twist_covariance(zupt_var_linear_.load(), zupt_var_yaw_.load())" in cpp
    assert "gate.judge(now_s, rest_evidence(now_s))" in cpp
    ekf = yaml.safe_load((REPO / "ros/params/ekf.yaml").read_text())
    params = ekf["ekf_filter_node"]["ros__parameters"]
    assert params["odom2"] == "zupt"
    fused = [i for i, on in enumerate(params["odom2_config"]) if on]
    assert fused == [6, 7, 11], "vx, vy, vyaw: the three indices the update claims"
    header = (REPO / "ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp").read_text()
    diagonal = re.search(r"std::array<double, 6> diagonal = \{([^}]*)\}", header)
    assert diagonal, "rest_zupt_twist_covariance builds its diagonal in one literal"
    terms = [term.strip() for term in diagonal.group(1).split(",")]
    claimed = [i for i, term in enumerate(terms) if term in ("var_linear", "var_yaw")]
    assert [6 + i for i in claimed] == [6, 7, 11], "the covariance claims what odom2 fuses"


def test_every_zupt_tunable_of_the_bridge_is_a_live_range_checked_parameter() -> None:
    """Heading drift has many causes, and Artem tunes them on the robot, never in C++: every
    number that decides the zero-velocity update is a parameter of /base_bridge, set live with
    `ros2 param set` and in force at the next tick. Held here: each is declared through the one
    helper that range-checks a launch value, with the default zupt.hpp names (the two windows
    borrowing imu_bias_s and cmd_timeout_s without moving them); a set is refused outside
    zupt.hpp's kZuptRanges by the on-set callback and applied by the post-set one, which re-times
    the timer for a new rate; nothing is looked up per tick; and the status line prints every
    value in force."""
    header = (REPO / "ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp").read_text()
    start = header.index("kZuptRanges = {{")
    ranged = re.findall(r'\{"(zupt_\w+)",', header[start : header.index("}};", start)])
    assert len(ranged) == 6, ranged
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    declared = dict(re.findall(r'declare_zupt_number\(\s*"(zupt_\w+)",\s*(.+?),\n', cpp))
    assert set(declared) == set(ranged), "every live number has a range, every range a number"
    assert "descriptor.dynamic_typing = true" in cpp, "`ros2 param set ... 50` is not refused"
    assert "add_on_set_parameters_callback(" in cpp and "return check_zupt_settings(" in cpp
    assert "add_post_set_parameters_callback(" in cpp and "apply_zupt_settings(parameters)" in cpp
    for name in ranged:
        assert f'if (name == "{name}") {{return &' in cpp, f"{name}: applied live"
    assert "if (slot == &zupt_hz_) {\n        start_zupt_timer();" in cpp, "a new rate re-times"
    assert 'get_parameter("zupt_' not in cpp, "no parameter lookup per tick"
    settings = cpp[cpp.index("std::string zupt_settings() const") :]
    settings = settings[: settings.index("return line;")]
    for field in ("zupt_hz_", "zupt_var_linear_", "zupt_var_yaw_", "zupt_settle_s_"):
        assert f"{field}.load()" in settings, f"the status line prints {field}"
    assert "zupt_cmd_hold_s_.load()" in settings and "gyro_quiet_rad_s_.load()" in settings


def test_every_flag_says_why_its_default_is_what_it_is_and_when_to_move_it() -> None:
    """A switch nobody can argue with is a switch nobody dares touch: every flag carries the
    measured reason for its default — with the numbers and the file they were measured in — or
    says plainly that it is `default by design, unmeasured`, plus when to turn it on and when to
    turn it off (CLAUDE.md rule 19; ros/flags.sh flag NODE FLAG prints all of it)."""
    from pepin.flags import UNMEASURED

    unmeasured = []
    for path in sorted((REPO / NODES).glob("*.py")):
        if "Switches" not in sf.imported(sf.tree(f"{NODES}/{path.name}")):
            continue
        for flag in load_table(path):
            where = f"{path.stem}/{flag.name}"
            assert flag.why, f"{where}: why this default? (or say {UNMEASURED!r})"
            assert flag.on_when, f"{where}: when is it turned on?"
            assert flag.off_when, f"{where}: when is it turned off?"
            assert len(flag.details()) == 4, where
            if not flag.measured:
                unmeasured.append(where)
                continue
            assert any(c.isdigit() for c in flag.why), f"{where}: a measured reason has numbers"
    assert len(unmeasured) <= 17, (
        "a ratchet, not a budget: measure one of these instead of raising the number"
        f" ({len(unmeasured)} defaults rest on nothing measured: {unmeasured})"
    )


def test_the_known_map_graph_rides_the_filter_s_own_odometry() -> None:
    """CLAUDE.md's one odometry: RTAB-Map is built on the EKF's odom -> base_link, not on a
    localiser's pose, which teleports when it relocalises and made every loop closure
    unacceptable (a neighbour edge of 0.888 m against a 0.244 m sigma, error ratio 3.64 over
    RGBD/OptimizeMaxError's 3.0, 2026-09-14). The table that used to put the graph on the
    tracker's pose is gone with the argument that selected it."""
    table = _rtabmap("RTABMAP")
    assert table["odom_frame_id"] == "odom"
    assert table["map_frame_id"] == "map", "World R's one frame"
    names = sf.assignments(sf.tree(VSLAM_LAUNCH))
    assert {"TRACKER_ODOM", "KNOWN_MAP", "SLAM", "SLAM_LIDAR", "SLAM_CAMERA_ONLY"}.isdisjoint(
        names
    ), "one table for every situation: the mode tables and the tracker-odometry one are gone"
    vslam = sf.tree(VSLAM_LAUNCH)
    arguments = {ast.unparse(c.args[0]) for c in sf.calls_to(vslam, "DeclareLaunchArgument")}
    assert "'graph_odom'" not in arguments, "the switch between them is gone too"


def _launch_dicts(launch: str) -> dict[str, dict[str, object]]:
    """Every module-level dict literal of ``launch``, by name: the tables a node is told from."""
    out: dict[str, dict[str, object]] = {}
    for name, source in sf.assignments(sf.tree(launch)).items():
        try:
            value = ast.literal_eval(source)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, dict):
            out[name] = value
    return out


def test_rtabmap_is_told_one_table_reading_one_snapshot_topic_and_owns_no_transform() -> None:
    """Stage one of World R, held as a contract. A "mode" was which sensors are alive, which is
    data: so there is exactly ONE RTAB-Map table, it reads ONE topic
    (``subscribe_sensor_data``, mutually exclusive with every other subscription), and the
    transform it publishes is RTAB-Map's own (publish_tf). The synchronised triple of before
    2026-09-19 is in git history."""
    vslam = sf.tree(VSLAM_LAUNCH)
    tables = _launch_dicts(VSLAM_LAUNCH)
    assert [name for name, t in tables.items() if any(k.startswith("Grid/") for k in t)] == [
        "RTABMAP"
    ], "one grid, one table"
    assert [name for name, t in tables.items() if "map_frame_id" in t] == ["RTABMAP"], "one frame"
    assert [name for name, t in tables.items() if "subscribe_sensor_data" in t] == ["RTABMAP"], (
        "one table, and nothing else"
    )
    table = tables["RTABMAP"]
    assert table["subscribe_sensor_data"] is True
    for name in ("subscribe_depth", "subscribe_rgb", "subscribe_scan", "subscribe_odom"):
        assert table[name] is False, f"{name}: rtabmap turns it off anyway; say it out loud"
    # WHO OWNS map -> odom: RTAB-Map (True, PUBLISH_MAP_TO_ODOM, merged by rtabmap_parameters in
    # every session); the only False is rgbd_odometry's (VISUAL_ODOMETRY). Never a broadcaster of
    # our own.
    assert sf.dict_items(vslam)["publish_tf"] == {"False", "True"}
    assert _rtabmap("VISUAL_ODOMETRY")["publish_tf"] is False
    assert _rtabmap("PUBLISH_MAP_TO_ODOM")["publish_tf"] is True
    assert not {"TransformBroadcaster", "StaticTransformBroadcaster"} & sf.imported(vslam)

    # One literal topic name, written on both sides of the contract (read from the sources: this
    # file never imports a ROS node, and rclpy is not installed here).
    node = sf.assignments(sf.tree(f"{NODES}/sensor_pack.py"))
    assert node["SENSOR_DATA_TOPIC"] == "'/rtabmap/sensor_data'"
    assert sf.assignments(vslam)["SENSOR_DATA_TOPIC"] == node["SENSOR_DATA_TOPIC"]
    assert "('sensor_data', SENSOR_DATA_TOPIC)" in sf.unparsed(vslam, ast.Tuple)
    assert "pepin_bringup.sensor_pack" in sf.strings(vslam)

    # The arguments: the five ros/laptop.sh passes are all still declared, and every argument
    # that used to SELECT A MODE is gone (World R: one database, one grid, one arrangement).
    # camera_only survives as one flag of one node, not as a table.
    arguments = {ast.unparse(c.args[0]) for c in sf.calls_to(vslam, "DeclareLaunchArgument")}
    laptop = (REPO / "ros/laptop.sh").read_text()
    start = laptop.index("ros2 launch pepin_bringup vslam.launch.py")
    command = laptop[start : laptop.index(">/dev/null", start)]
    passed = {f"'{name}'" for name in re.findall(r'"?(\w+):=\$', command)}
    assert passed <= arguments, (
        f"ros/laptop.sh passes what this launch no longer declares: {passed - arguments}"
    )
    for gone in (
        "'sensor_pack'",
        "'neighbor_refining'",
        "'database'",
        "'camera'",
        "'graph_odom'",
        "'slam'",
        "'resume'",
        "'world_map'",
        "'room'",
        "'map_source'",
        "'memory'",
    ):
        assert gone not in arguments, f"{gone} selected a mode that no longer exists"
    passes_sources = [s for s in sf.unparsed(vslam, ast.JoinedStr) if "sources:=" in s]
    assert len(passes_sources) == 1, "camera_only reaches exactly one node's sources flag"


def test_no_launch_argument_reaches_a_node_as_an_empty_parameter_override() -> None:
    """An override with nothing after the ``:=`` is not "the default": rcl refuses to parse the
    rule and the process dies inside ``rclpy.init`` — "Couldn't parse parameter override rule:
    '-p seed_map:='" — every launch, before a line of the node runs. An argument that may be
    empty is passed only when it has a value.

    Measured in the laptop container on 2026-09-14: ``rclpy.init(args=[..., "-p",
    "seed_map:="])`` raises RCLError, ``"-p", "seed_map:=/maps/flat3_straight.yaml"`` does not.
    """
    launch = (REPO / VSLAM_LAUNCH).read_text()
    overrides = [ln.strip() for ln in launch.splitlines() if ":={" in ln]
    assert overrides, "the launch still hands the nodes parameter overrides"
    # Every one of them interpolates a word this file chooses, never a launch argument that may
    # arrive empty: no argument defaults to empty, and the rig is the word ``camera_rig`` answers
    # with — a name from config/camera.json, which is never empty because that reader raises
    # instead of shrugging.
    empty_by_default = {
        ast.unparse(c.args[0]).strip("'")
        for c in sf.calls_to(sf.tree(VSLAM_LAUNCH), "DeclareLaunchArgument")
        if ast.unparse(sf.keywords(c).get("default_value", ast.Constant(None))) == "''"
    }
    assert empty_by_default == set()
    assert not [ln for ln in overrides if "database:=" in ln]
    assert "rig = camera_rig(" in launch, "the rig is resolved in the launch, once"
    assert 'f"camera:={rig}"' in launch, "and it is the resolved word that reaches the node"


def test_a_reset_empties_the_volume_and_nothing_falls_back_to_a_picture() -> None:
    """``/fusion/reset`` empties the volume, and with no pgm in the loop "empty" means empty.
    Nothing outside this node reads it, so emptying it costs no tracker anything — it costs the
    surface cloud until the sensors have painted one again."""
    src = (REPO / NODES / "depth_fusion.py").read_text()
    assert src.count("WorldMap(self._spec, self._mount") == 3, (
        "the volume is built in three places only: the placeholder __init__ holds until the"
        " starting state is known, the starting state's own, and the reset's"
    )
    assert src.count("self._world = self._fresh_world()") == 1, "the reset alone"
    assert "self._fresh_world" in sf.calls(sf.tree(f"{NODES}/depth_fusion.py"))


def test_the_depth_network_runs_where_the_backend_flag_says_and_the_cpu_model_waits() -> None:
    """The depth node asks ONE depth source for a frame's raw depth (``self._source(views)``,
    2026-09-20: the stereo head is the second one) and the network source calls the backend with
    the left picture and nothing else; that backend is the switch between the laptop's GPU
    service and the CPU model in the container (pepin.depth_service.Fallback), picked live by
    the ``depth_backend`` flag whose default comes from PEPIN_DEPTH_BACKEND, and the CPU model is
    built on its first local frame, never at start. laptop.sh sets the flag and the service's
    address only when it starts the service; without them the node is on the CPU as before."""
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert {"Fallback", "RemoteDepth", "LazyDepth"} <= sf.imported(node)
    calls = sf.unparsed(node, ast.Call)
    assert "self._source(views)" in calls, "the worker asks one source for the raw depth"
    assert "self._backend()(views.rgb)" in calls, "the network source calls the backend"
    switch = sf.calls_to(node, "Fallback")
    assert len(switch) == 1 and sf.dotted(switch[0].args[0]) == "RemoteDepth()"
    assert ast.unparse(sf.keywords(switch[0])["mode"]) == "self._switches['depth_backend']"
    models = sf.calls_to(node, "MonoDepth")
    lambdas = [n for n in ast.walk(node) if isinstance(n, ast.Lambda)]
    assert models and all(any(m in ast.walk(lam) for lam in lambdas) for m in models), (
        "the CPU model is built inside LazyDepth's lambda, not at start"
    )
    assert "self._net.mode" in {
        ast.unparse(t) for n in ast.walk(node) if isinstance(n, ast.Assign) for t in n.targets
    }
    flags = load_table(REPO / NODES / "depth_stream.py")
    backend = flags.flag("depth_backend")
    assert backend.env == "PEPIN_DEPTH_BACKEND" and backend.live
    assert "self._net.status" in {
        ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }, "the switch's status is in the report line"
    url = next(
        c
        for c in sf.calls_to(node, "self.declare_parameter")
        if ast.unparse(c.args[0]) == "'depth_url'"
    )
    assert "PEPIN_DEPTH_URL" in ast.unparse(url.args[1]) and "DEFAULT_URL" in ast.unparse(
        url.args[1]
    )
    laptop = (REPO / "ros/laptop.sh").read_text()
    vslam_run = next(
        c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-vslam" in c
    )
    assert '${DEPTH_ENV[@]+"${DEPTH_ENV[@]}"}' in vslam_run
    assert (
        'DEPTH_ENV=(-e PEPIN_DEPTH_BACKEND=auto -e "PEPIN_DEPTH_URL=http://host.docker.internal:'
        in laptop
    )


def test_a_cpu_model_that_cannot_be_built_ends_the_node_in_local_mode() -> None:
    """Built in the constructor, an uncached model with no hub ended the process visibly. Built
    on its first frame it fails on the worker thread, where a raise is one logged frame among
    the next: so the node catches pepin.depth_service.DepthModelError (LazyDepth remembers the
    failure — no rebuild per frame) and in local mode leaves through the kit's Fatal, exit
    code 1, the launch respawns it, no more frames offered on the way out; in auto mode the
    frame is lost and the service keeps being probed."""
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert {"DepthModelError", "Fatal"} <= sf.imported(node)
    handled = {
        sf.dotted(h.type)
        for h in ast.walk(node)
        if isinstance(h, ast.ExceptHandler) and h.type is not None
    }
    assert "DepthModelError" in handled, "a failed build is a typed condition, not a traceback"
    assert len(sf.calls_to(node, "Fatal")) == 1 and sf.calls_to(node, "self._fatal.leave")
    assert "self._fatal.leaving" in sf.unparsed(node, ast.Attribute), "no frames on the way out"


@pytest.mark.slow
def test_the_flags_script_reaches_a_node_where_it_runs_and_refuses_before_any_host() -> None:
    """ros/flags.sh runs the ros2 CLI inside the container a node lives in (the laptop's by
    docker exec, the board's over ssh), one parameter dump per node for a listing, and asks
    the table first: a node without flags, a flag the node has not got, a value the flag
    refuses — each ends here with the reason and exit 2, no host touched (like kick)."""
    import os

    src = (REPO / "ros/flags.sh").read_text()
    assert '. "$HERE/lib.sh"' in src and "ros/tools/flags_doc.py" in src
    assert "docker exec" in src and 'ssh "root@$BOARD"' in src and "/pepin_entrypoint.sh" in src
    for verb in ("ros2 param dump", "ros2 param get", "ros2 param set"):
        assert verb in src, verb
    assert src.count("ssh ") == 1, "one path to the board: ros2_in"
    env = {**os.environ, "PEPIN_HOST": "127.0.0.1"}

    def flags_sh(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(REPO / "ros/flags.sh"), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    for args, reason in (
        (["get", "no_such_node", "x"], "no node with a flags table"),
        (["get", "depth_fusion", "gpu"], "no flag gpu"),
        (["set", "depth_stream", "depth_backend", "gpu"], "'gpu' is not one of remote, local"),
        (["set", "depth_fusion", "min_weight"], "usage"),
        (["flag", "depth_stream"], "usage"),
        (["flag", "depth_stream", "nope"], "no flag nope"),
        (["frob"], "usage"),
    ):
        refused = flags_sh(*args)
        assert refused.returncode == 2, (args, refused.stdout, refused.stderr)
        assert reason in refused.stdout + refused.stderr, (args, refused.stdout, refused.stderr)
    # the reading verb: the whole entry from the table, no node asked, no container entered
    told = flags_sh("flag", "depth_stream", "edge_filter")
    assert told.returncode == 0, told.stderr
    for label in ("What:", "Default:", "On when:", "Off when:"):
        assert f"\n{label}" in told.stdout, label


def test_the_floor_anchors_the_depth_and_leans_with_the_imu() -> None:
    """The depth node snaps floor pixels to the floor plane (switchable), the plane leans with
    the cart, and the IMU mount the laptop would apply is config/imu.json's (roll +90 deg: the
    chip's Y up), the one the board's bridge rotates its readings by."""
    node = sf.tree(f"{NODES}/depth_stream.py")
    flags = load_table(REPO / NODES / "depth_stream.py")
    kit = sf.tree(f"{NODES}/node_kit.py")
    assert "/imu/data_raw" in sf.strings(kit), "the lean's one subscription lives in the kit"
    assert {"LeanEstimator", "Mounts"} <= sf.imported(kit)
    # The anchor is the pipeline's stage of that name, fed the IMU's up vector through the
    # frame's context; the flags are one bool per stage, in the chain's order, and the chain's
    # defaults are the measured ones (the lidar's affine law alone: scratch/pipeline_vs_truth).
    from pepin.depth_pipeline import standard_pipeline

    pipeline = standard_pipeline()
    assert [f.name for f in flags][: len(pipeline.names)] == pipeline.names
    assert {name: flags[name] for name in pipeline.names} == pipeline.switches
    assert {"FrameContext", "LeanFeed"} <= sf.imported(node) and "self._lean.up" in sf.unparsed(
        node, ast.Attribute
    )
    # The mount is not read here by hand: one loader for every sensor's place on the cart,
    # and the kit's LeanFeed is the only caller of it for the IMU.
    assert "Mounts.load" in sf.calls(kit)
    mount = json.loads((REPO / "config/imu.json").read_text())["mount"]
    assert mount["roll_deg"] == 90.0
    # The launch publishes the lidar's file, not a copy of its numbers: every rotation and
    # translation value of its static transforms is a name, never a literal.
    robot = sf.tree(ROBOT_LAUNCH)
    entries = sf.dict_items(robot)
    for key, values in entries.items():
        if key.startswith(("rotation.", "translation.")):
            assert all(value.isidentifier() for value in values), (key, values)


def test_the_laptop_mounts_the_library_live_not_a_copy() -> None:
    """A copy of src/pepin went stale whenever anything but laptop.sh restarted a container: the
    laptop containers mount the library itself, like the ROS package."""
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert '"$HERE/../src/pepin:/ws/pepin_src/pepin:ro"' in laptop
    assert "rsync" not in laptop


def test_the_board_image_carries_no_lttng_tracer() -> None:
    """lttng-ust, pulled in by the binary tracetools, costs 128 MB of resident memory per ROS
    process on load: with nine processes the 1.5 GB board lived in swap and froze under load
    (2026-09-10). The image rebuilds tracetools without it and puts that build where every
    consumer looks, so no future base image brings the tracer back unnoticed."""
    dockerfile = (REPO / "ros/Dockerfile").read_text()
    stage = dockerfile[dockerfile.index("ros2_tracing") :]
    assert "-DTRACETOOLS_TRACEPOINTS_EXCLUDED=ON" in stage and "TRACETOOLS_DISABLED=ON" not in stage
    assert "cp -f install/tracetools/lib/libtracetools.so /opt/ros/jazzy/lib/" in stage
    assert dockerfile.index("ros2_tracing") < dockerfile.index("COPY pepin_bringup")


def test_a_crashed_navigation_container_comes_back_by_itself() -> None:
    """Respawned WITH its nodes, described anew for every start by a factory
    (pepin_bringup.launch_kit, behaviour in test_launch_kit): a reload of the first start's
    descriptions loses the controller's cmd_vel -> cmd_vel_nav remap."""
    nav = sf.tree(NAV_LAUNCH)
    assert not sf.calls_to(nav, "ComposableNodeContainer"), "only through respawned_container"
    container = sf.calls_to(nav, "respawned_container")[0]
    assert ast.unparse(container.args[1]) == "nav_parts"
    assert ast.unparse(sf.keywords(container)["parameters"]) == "[params]"
    parts = next(n for n in nav.body if isinstance(n, ast.FunctionDef) and n.name == "nav_parts")
    assert sf.calls_to(parts, "ComposableNode"), "the descriptions are built inside the factory"


def _launch_processes(name: str) -> dict[str, dict[str, object]]:
    """Every ExecuteProcess/Node call of a launch file, keyed by what it runs (the pepin_bringup
    module, else the executable), mapped to its keyword arguments: literals as values,
    anything else as ``ast.unparse`` renders it, ``**FLAGS`` unpacked from the module's own
    dict constants."""
    tree = sf.tree(f"ros/pepin_bringup/launch/{name}")
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    found: dict[str, dict[str, object]] = {}
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        if call.func.id not in ("ExecuteProcess", "Node"):
            continue
        keywords: dict[str, object] = {}
        for keyword in call.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.Name):
                keywords.update(constants[keyword.value.id])
            elif keyword.arg is not None and isinstance(keyword.value, ast.Constant):
                keywords[keyword.arg] = keyword.value.value
            elif keyword.arg is not None:
                keywords[keyword.arg] = ast.unparse(keyword.value)
        strings = [
            n.value
            for n in ast.walk(call)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        modules = [s for s in strings if s.startswith(("pepin_bringup.", "pepin."))]
        module = next(iter(modules), None)
        key = module.split(".", 1)[1] if module else str(keywords["executable"])
        found[key] = keywords
    return found


def _respawning(launch: dict[str, dict[str, object]]) -> set[str]:
    return {name for name, keywords in launch.items() if keywords.get("respawn") is True}


def test_a_node_comes_back_by_itself_but_the_watches_exit_on_purpose() -> None:
    """A code change is one kicked process, not a container restart: the launches respawn our
    nodes (and Foxglove's bridge) two seconds after they exit. The link watch must not: it
    cancels and keeps running. RTAB-Map stays out too: the graph is its state and its crash must
    stay visible. Its database is never wiped by the launch (no ``-d``): a launch that wiped it
    lost the map each time; an empty start is ``ros/laptop.sh vslam --fresh``."""
    vslam_launch = sf.tree(VSLAM_LAUNCH)
    for callee in ("Node", "ExecuteProcess"):
        assert not any("arguments" in sf.keywords(c) for c in sf.calls_to(vslam_launch, callee))
    laptop = (REPO / "ros/laptop.sh").read_text()
    fresh = laptop[laptop.index("    vslam)") : laptop.rindex("    *)")]
    assert "--fresh) FRESH=true ;;" in fresh and 'rm -f "$HERE"/maps/rtabmap.db' in fresh
    assert fresh.index("rm -f") < fresh.index("docker run")
    vslam = _launch_processes("vslam.launch.py")
    nav = _launch_processes("nav.launch.py")
    assert _respawning(vslam) == {
        "camera_stream",
        "depth_stream",
        "contact_scan",
        "depth_fusion",
        # The one input RTAB-Map reads (2026-09-19). It keeps nothing across a restart but a
        # second of each source's stamps, which is what it takes to measure a period again.
        "sensor_pack",
        "rtabmap_frame",
        # The room's vocabulary: its book is on disk beside the database, so a restart loses
        # nothing but the moment before the first graph arrives.
        "places",
        # Who painted the costmap's lethal cells (2026-09-22): it holds nothing across a restart
        # — the next grid is one second away — and a drive nobody was auditing is a drive to
        # repeat, so it comes back by itself like everything else here.
        "marks_audit",
        "foxglove_bridge",
        # The camera as a third odometry: rtabmap's node and ours. Neither carries state the
        # way RTAB-Map's graph does — rgbd_odometry's is the last frame, and a restart of it is
        # a jump pepin.visual_odometry.VoGate drops. stereo_odometry is the same role under
        # vo_input:=stereo; the launch starts one of the two.
        "rgbd_odometry",
        "stereo_odometry",
        "visual_odometry",
    }
    # Two recorders, one of which the launch starts (nav.launch.py's ``recorder`` argument): the
    # JSONL tape, or `ros2 bag record` under a node that subscribes to nothing. Both respawn, and
    # a kick reaches whichever is running. The gaze arbiter keeps its requests in memory only:
    # a respawn drops them, and every waiting caller already has its own timeout.
    assert _respawning(nav) == {"run_recorder", "bag_recorder", "goal_server", "gaze"}
    # The board's sensor launch runs our processes too (the neck node among them until it went
    # into the base bridge, 2026-10-02). The drivers around them are ROS packages the container
    # restarts with the launch.
    robot = _launch_processes("robot.launch.py")
    # tof_bridge since 2026-09-21: it died once on the robot and stayed dead, and a near-field
    # sensor that silently never comes back is worse than one that was never on.
    # ...and, since 2026-09-22, the lidar's own odometry: a third-party binary, respawned like
    # ours because the EKF reads its topic.
    # ...and the board's own recording's supervisor (pepin.board_bag, board_bag:=true): a dead
    # one takes its recorder with it, and the respawn starts both on a new directory.
    assert _respawning(robot) == {
        "tof_bridge",
        "rf2o_laser_odometry_node",
        "board_bag",
    }
    for launch in (vslam, nav, robot):
        for name, keywords in launch.items():
            if keywords.get("respawn"):
                assert 0.0 < float(str(keywords["respawn_delay"])) <= 5.0, name
    assert "link_watch" not in nav and "respawn" not in vslam["rtabmap"]


def test_the_laptop_image_provides_what_the_laptop_nodes_import() -> None:
    """The laptop launch's modules import what only ros/Dockerfile.laptop installs (the depth
    network, OpenCV, RTAB-Map's messages): a node that imported a library the image lacks died
    on the first frame. Every third-party import of those modules maps to an apt or pip name
    on the image's install lines (the base image, ros/Dockerfile, provides the ROS core)."""
    import ast
    import sys

    laptop_image = (REPO / "ros/Dockerfile.laptop").read_text()
    base_image = (REPO / "ros/Dockerfile").read_text()
    installs = " ".join(
        line.strip()
        for text in (laptop_image, base_image)
        for line in text.splitlines()
        if "install" in line or line.strip().startswith(("ros-jazzy-", "python3-"))
    )
    # import name -> what provides it, as it reads on the install lines
    provided = {
        "torch": ("torch",),
        "torchvision": ("torchvision",),
        "transformers": ("transformers",),
        "PIL": ("pillow",),
        "cv2": ("python3-opencv",),
        "rtabmap_msgs": ("ros-jazzy-rtabmap-ros", "ros-jazzy-rtabmap-msgs"),
        "cv_bridge": ("ros-jazzy-cv-bridge",),
        # navigation2 depends on message_filters (nav2_costmap_2d): the base image carries it
        "message_filters": ("ros-jazzy-message-filters", "ros-jazzy-navigation2"),
        # ...and on map_msgs (StaticLayer's OccupancyGridUpdate): depth_fusion's camera grids
        "map_msgs": ("ros-jazzy-map-msgs", "ros-jazzy-navigation2"),
    }
    ros_core = {
        "rclpy", "tf2_ros", "std_msgs", "sensor_msgs", "geometry_msgs", "nav_msgs", "std_srvs",
        "rcl_interfaces", "tf2_msgs", "action_msgs", "builtin_interfaces", "visualization_msgs",
        "numpy", "yaml", "pepin", "pepin_bringup",
    }  # fmt: skip
    for module in (
        "camera_stream",
        "depth_stream",
        "contact_scan",
        "depth_fusion",
        "rtabmap_frame",
    ):
        tree = ast.parse((REPO / "ros/pepin_bringup/pepin_bringup" / f"{module}.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        third_party = imported - set(sys.stdlib_module_names) - ros_core
        for name in sorted(third_party):
            assert name in provided, f"{module} imports {name}: say what provides it"
            assert any(token in installs for token in provided[name]), (
                f"{module} imports {name}; none of {provided[name]} is on the install lines"
            )


def test_the_hook_checks_the_shell_scripts_and_the_books_stay_on_the_mac() -> None:
    """The pose the tracker writes on the laptop's mount is not a tracked file; the places books
    live on the Mac with the maps (2026-10-01: the board is a sensor box), so they are neither
    pushed to the board nor fetched from it, and turn_full refuses to turn under a goal."""
    ignored = (REPO / ".gitignore").read_text().splitlines()
    assert "ros/maps/last_pose.json" in ignored
    fetch = (REPO / "ros/fetch.sh").read_text()
    assert "root@$BOARD:/root/pepin-ros/maps/rec/" in fetch, "the board's recordings come home"
    assert "places.yaml" not in fetch, "the board's copy of a book is no longer the truth"
    sync = (REPO / "ros/sync.sh").read_text()
    # No map goes to the sensor box, and --delete never reaches what the board wrote under maps/.
    assert "--exclude 'maps/*'" in sync
    tracked = subprocess.run(
        ["git", "ls-files", "ros/maps"], capture_output=True, text=True, cwd=REPO, check=True
    ).stdout
    assert ".places.yaml" in tracked, "the books stay tracked, here"
    turn = sf.tree("ros/tools/turn_full.py")
    limits = sf.assignments(turn)
    # a full circle at 0.35 rad/s takes 18.5 s; a turn still running past a minute is stuck
    assert 18.5 < ast.literal_eval(limits["MAX_S"]) <= 60.0
    assert 0.0 < ast.literal_eval(limits["ODOM_WAIT_S"]) <= 5.0
    assert "f'/{action}/_action/status'" in sf.unparsed(turn, ast.JoinedStr)
    assert {"/odometry/filtered", "/odom"} <= sf.strings(turn)
    tape_starts = min(
        n.lineno
        for n in ast.walk(turn)
        if isinstance(n, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "cmd" for k in n.keys)
    )
    refusals = {
        ast.literal_eval(c.args[0]) for c in sf.calls_to(turn, "sys.exit") if c.lineno < tape_starts
    }
    assert {2, 3} <= refusals, "refuse before the tape starts"


def test_one_node_can_be_kicked_without_a_container_restart() -> None:
    """laptop.sh kick / board.sh kick: SIGINT to one process (never -9, never the container),
    then the log is read for the line the node prints once up — a line its source really
    contains, from the new pid (ros/kick_ready.awk, tests/unit/test_kick.py). The names a kick
    knows are exactly the nodes the launches respawn, and an unknown name is refused before any
    host is touched."""
    import os

    known: dict[str, set[str]] = {}
    for script in ("laptop.sh", "board.sh"):
        src = (REPO / "ros" / script).read_text()
        kick = src[src.index("    kick)") :].split("\n    *)")[0]
        assert 'pgrep -f "pepin_bringup[./]$1"' in kick and "kill -INT" in kick, script
        assert "kill -9" not in kick, script
        assert "docker restart" not in kick and "systemctl restart" not in kick, script
        table = re.search(r"^kick_(?:target|line)\(\) \{.*?^\}", src, re.M | re.S)
        assert table is not None, script
        rows = re.findall(r'^\s+(\w+)\) echo "([^"]+)" ;;', table.group(0), re.M)
        assert rows, script
        for name, target in rows:
            line = target.rpartition("|")[2]
            node = (REPO / "ros/pepin_bringup/pepin_bringup" / f"{name}.py").read_text()
            assert line in node, (script, name, line)
        listed = re.search(r'^KICKABLE="([^"]+)"', src, re.M)
        assert listed is not None and set(listed.group(1).split()) == {n for n, _ in rows}
        refused = subprocess.run(
            ["bash", str(REPO / "ros" / script), "kick", "no_such_node"],
            env={**os.environ, "PEPIN_MAP": "/maps/x.yaml", "PEPIN_HOST": "127.0.0.1"},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert refused.returncode == 2, (script, refused.stdout, refused.stderr)
        assert all(name in refused.stdout for name, _ in rows), refused.stdout
        known[script] = {name for name, _ in rows}
    vslam, nav = _launch_processes("vslam.launch.py"), _launch_processes("nav.launch.py")
    robot = _launch_processes("robot.launch.py")
    # Foxglove's bridge and rtabmap's two odometries are not ours to kick: the kick sends SIGINT
    # to a "pepin_bringup.<module>" command line, and none of them is one.
    not_ours = {"foxglove_bridge", "rgbd_odometry", "stereo_odometry"}
    kickable = (_respawning(vslam) - not_ours) | _respawning(nav)
    assert known["laptop.sh"] == kickable
    # Everything of OURS the board respawns is kickable: the sensor launch's own nodes (a code
    # change on the board is one kicked process, never a restart). rf2o's binary is not ours and
    # not a "pepin_bringup.<module>" command line — it changes only when the image is rebuilt, so
    # there is nothing to kick it for. Nor the recording's supervisor: a kick would only close one
    # recording and open the next, which the next stack restart does anyway.
    assert known["board.sh"] == _respawning(robot) - {"rf2o_laser_odometry_node", "board_bag"}


def test_the_graphs_grid_is_the_one_map_and_rtabmap_owns_map_to_odom() -> None:
    """World R with one localiser. RTAB-Map on the laptop IS the map: the EKF's odometry under
    it, its grid relayed onto /map (transient local, read by both costmaps' static layers), ONE
    database that is never wiped here, and the one publisher of map -> odom."""
    vslam = sf.tree(VSLAM_LAUNCH)
    table = _rtabmap("RTABMAP")
    assert table["odom_frame_id"] == "odom" and table["map_frame_id"] == "map"
    # One owner: publish_tf True, from PUBLISH_MAP_TO_ODOM (the False is rgbd_odometry's).
    assert sf.dict_items(vslam)["publish_tf"] == {"False", "True"}
    # The grid IS the map. It leaves RTAB-Map on its own topic and pepin_bringup.rtabmap_frame
    # relays it onto /map.
    assert "remappings.append(('map', '/rtabmap/grid'))" in sf.unparsed(vslam, ast.Call), (
        "unconditionally: there is no second map left for it to make way for"
    )
    assert "('map', '/map')" not in sf.unparsed(vslam, ast.Tuple), "/map has one publisher"
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert {"/rtabmap/grid", "/map"} <= set(sf.strings(frame))
    assert not [
        n for n in ast.walk(vslam) if isinstance(n, ast.If) and "slam" in ast.unparse(n.test)
    ], "no mode decides anything in this launch any more"
    # One database, never wiped here: ros/laptop.sh vslam --fresh deletes the file, and the launch
    # reads whether it exists to pick the memory mode.
    assert sf.dict_items(vslam)["delete_db_on_start"] == {"False"}
    names = sf.assignments(vslam)
    assert "SLAM_DATABASE" not in names and names["DATABASE"] == "'/maps/rtabmap.db'"
    assert "Path(database).is_file()" in sf.unparsed(vslam, ast.Call), "an empty room is a fact"
    # The grid is the same one in every situation: from BOTH sensors per node, 2D for Nav2's
    # static layer, ray-traced because Grid/Sensor 2 gives up the cheap path that carves a scan's
    # own free space, and reaching as far as the LIDAR does — the camera's own reach travels in
    # the camera's data (pepin_bringup.depth_stream's depth_reach), not in this parameter.
    assert table["Grid/Sensor"] == "0" and table["Grid/3D"] == "false"
    assert table["Grid/RayTracing"] == "true", (
        "kept for the depth-built grid of a lidar-less wake-up"
    )
    assert float(str(table["Grid/RangeMax"])) == 8.0
    assert table["RGBD/NeighborLinkRefining"] == "false", (
        "refined links made OptimizeMaxError reject 87 closures on the first real drive; the gyro's"
        " bias tracked at rest removed the reason they had been switched on (2026-09-19)"
    )
    assert table["Reg/Strategy"] == "1", "ICP; 2 would drop every node that has no picture"
    assert table["Mem/BadSignaturesIgnored"] == "false", "a node with no picture is KEPT"
    assert table["RGBD/ProximityPathMaxNeighbors"] == "10", (
        "the wrapper only inserts this while a scan is SUBSCRIBED (CoreWrapper.cpp:489-504), and"
        " it no longer is: unsaid it falls back to 0, which disables one-to-many proximity"
    )
    assert table["RGBD/OptimizeFromGraphEnd"] == "false", "the jump belongs in map -> odom"
    # Nav2: no tracker, no retired frame owner, no served pgm.
    nav = sf.tree(NAV_LAUNCH)
    assert not {"pepin_bringup.slam_frame", "relocalizer"} & set(sf.strings(nav))
    # The grid the board plans on: the static layer takes /map, latched, and every
    # planner may route through what nobody has looked at yet — a map that is still growing.
    assert _p("global_costmap")["static_layer"]["map_subscribe_transient_local"] is True
    planners = _p("planner_server")
    for name in planners["planner_plugins"]:
        assert planners[name]["allow_unknown"] is True, name


def test_one_gesture_per_side_brings_the_stack_up_and_one_saves_the_map() -> None:
    """Two gestures, and neither of them names a mode any more (World R): ros/laptop.sh nav and
    ros/laptop.sh vslam, neither of which asks the board anything. ros/map.sh save freezes the
    grid into the pair map_server would read."""
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert not re.search(r"\b(nav|slam|slam_toolbox|map|side|recorder):=", unit)
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    args = {ast.unparse(c.args[0]) for c in sf.calls_to(bringup, "DeclareLaunchArgument")}
    assert not {"'nav'", "'slam'", "'slam_toolbox'", "'map'", "'side'", "'recorder'"} & args
    laptop = (REPO / "ros/laptop.sh").read_text()
    blocks = _case_blocks(laptop)
    assert "nav" in blocks and "start" not in blocks and "laptop-slam" not in laptop
    assert "nav.launch.py" in blocks["nav"] and "side:=" not in laptop and ".mode" not in laptop
    vslam_run = next(
        c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-vslam" in c
    )
    assert '"camera_only:=$CAMERA_ONLY"' in vslam_run
    for gone in ("slam:=", "resume:=", "world_map:=", "room:=", "resume_volume:="):
        assert gone not in vslam_run, f"{gone} selected a mode that no longer exists"
    save = (REPO / "ros/map.sh").read_text()
    assert "map_saver_cli" in save and "-t /map" in save and "pepin-vslam" in save
    assert "save_map_timeout" in save


def test_the_camera_s_optics_have_exactly_one_reader() -> None:
    """A stack that measures with a guessed field of view looks exactly like one measuring with
    a calibration, so there is one function that decides — pepin.camera.optics — and no node
    builds a pinhole from hfov_deg for itself. Held on the syntax trees: a node that goes back
    to reading cfg.hfov_deg would be a second place to remember when a calibration is written.
    """
    nodes = sorted((REPO / "ros/pepin_bringup/pepin_bringup").glob("*.py"))
    readers = []
    for path in nodes:
        module = sf.tree(str(path.relative_to(REPO)))
        called = sf.calls(module)
        if "optics" in called:
            readers.append(path.stem)
        assert "intrinsics" not in called, f"{path.stem}: build the optics through pepin.camera"
        assert "camera_info_arrays" not in called, f"{path.stem}: optics().camera_info_arrays()"
    assert readers == ["camera_stream", "depth_stream"], readers
    # The one reader answers both provenances and says which in words, for the report lines.
    camera = sf.tree("src/pepin/camera.py")
    assert "Optics" in sf.names(camera) and "Calibration" in sf.names(camera)
    assert "write_calibration" in {f.name for f in camera.body if isinstance(f, ast.FunctionDef)}


def test_the_calibration_tool_is_reachable_as_one_command() -> None:
    """ros/calibrate.sh is the whole procedure — print the board, collect, fit, write — so no
    step lives only in someone's memory (the brief for this camera: one command)."""
    script = (REPO / "ros/calibrate.sh").read_text()
    assert "scripts/calibrate_camera.py" in script and "--host" in script
    runner = sf.tree("scripts/calibrate_camera.py")
    called = sf.calls(runner)
    assert {"calibrate", "find_corners", "board_pdf", "write_calibration"} <= called
    # the mount's tilt was fitted together with the field of view the checkerboard replaces, so
    # the run that writes the intrinsics must say the tilt is now stale.
    assert any("pitch" in text and "depth_fit_models" in text for text in sf.strings(runner))


def test_no_node_shadows_an_rclpy_node_attribute() -> None:
    """rclpy.Node keeps its clock, logger, parameters and handles in private attributes; a node
    that assigns its own `self._clock` dies at its first timer (2026-09-13, the fusion node's
    snapshot clock; 2026-09-13 again, the bridge watch's `self._subscriptions: dict = {}`,
    which the regex without the annotation missed — rclpy appends every new subscription to
    that list and the node died at its first attach). The stubs cannot catch it, so the
    contract does."""
    import re
    from pathlib import Path

    owned = (
        "_clock",
        "_logger",
        "_parameters",
        "_handle",
        "_context",
        "_executor",
        "_publishers",
        "_subscriptions",
        "_timers",
        "_clients",
        "_services",
        "_guards",
        "_waitables",
        "_default_callback_group",
    )
    nodes = Path(__file__).resolve().parents[2] / "ros" / "pepin_bringup" / "pepin_bringup"
    offenders = [
        f"{p.name}: self.{name}"
        for p in sorted(nodes.glob("*.py"))
        for name in owned
        if re.search(rf"self\.{name}\s*(:[^=\n]*)?=", p.read_text())  # annotated too
    ]
    assert offenders == [], offenders


def test_no_node_publishes_on_a_topic_it_listens_to() -> None:
    """A node that publishes on a topic it subscribes to receives its own messages back (ROS 2
    delivers local publications too). The relocalizer told AMCL its seed on /initialpose and,
    once it listened there for the operator, fed itself: every seed came back as a new seed,
    20 times a second, map -> odom flipping 16 cm (2026-09-13). Literal topic names only;
    a topic named through a constant is not seen here."""
    import re
    from pathlib import Path

    nodes = Path(__file__).resolve().parents[2] / "ros" / "pepin_bringup" / "pepin_bringup"
    literal = r'\(\s*[^,()]+,\s*"([^"]+)"'
    offenders = []
    for p in sorted(nodes.glob("*.py")):
        source = p.read_text()
        published = set(re.findall(r"create_publisher" + literal, source))
        listened = set(re.findall(r"create_subscription" + literal, source))
        offenders += [f"{p.name}: {topic}" for topic in sorted(published & listened)]
    assert not offenders, f"a node would hear itself on: {offenders}"


def test_the_filter_survives_a_dead_gyro_because_the_launch_never_gated_it_on_one() -> None:
    """No single point of failure in the odometry: the EKF is the stack's one publisher of
    odom -> base_link and of /odometry/filtered — the topic the relocalizer and the run recorder
    read — so it must run on whichever sources are alive. It was conditioned on ``imu:=true``
    until 2026-09-15, and ``ros/feature.sh imu off`` therefore took the filter down with the
    gyro: zero messages on /odometry/filtered, the relocalizer carrying scans on an odometry
    that never arrived, goto refusing with "not localized". The bridge hands over the transform
    whenever the filter runs; the way back to no filter is its own switch.
    """
    robot = sf.tree(ROBOT_LAUNCH)
    condition = ast.unparse(sf.keywords(_node_named(robot, "ekf_filter_node"))["condition"])
    assert condition == "IfCondition(LaunchConfiguration('ekf'))"
    assert "'imu'" not in condition, "the gyro is a source of this filter, never its switch"
    # The bridge hands over odom -> base_link to whoever publishes it: keyed on the filter, so
    # `imu off` can never leave the edge unpublished nor let both publish it.
    assert "ekf_on = LaunchConfiguration('ekf').perform(context).lower() == 'true'" in sf.unparsed(
        robot, ast.Assign
    )
    # False is the laser odometry's, and for the same reason from the other side: with the filter
    # up exactly one node publishes odom -> base_link, and it is the filter.
    assert sf.dict_items(robot)["publish_tf"] == {"not ekf_on", "False"}, (
        "the bridge follows the filter; the laser odometry never publishes the edge"
    )
    assert sf.dict_items(robot)["imu_enable"] == {"imu_on"}, "the gyro keeps its own switch"
    # Reachable end to end: one operator gesture, one env var, one argument.
    assert "'ekf'" in {ast.unparse(c.args[0]) for c in sf.calls_to(robot, "DeclareLaunchArgument")}
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    assert "LaunchConfiguration('ekf')" in sf.unparsed(bringup, ast.Call)
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "ekf:=${PEPIN_EKF}" in unit
    assert "ekf) VAR=PEPIN_EKF ;;" in (REPO / "ros/feature.sh").read_text()
    # And without the gyro the heading has to come from somewhere: the wheels' own yaw rate.
    ekf_yaml = yaml.safe_load((REPO / "ros/params/ekf.yaml").read_text())["ekf_filter_node"][
        "ros__parameters"
    ]
    assert ekf_yaml["odom0_config"][11] is True, "the wheels carry the heading when the gyro dies"


def test_the_laser_odometry_is_a_twist_the_filter_can_weigh_and_never_a_second_transform() -> None:
    """The board keeps odometry and no map localiser (2026-09-22), so the lidar's own scan-to-scan
    motion reaches the EKF as odom3. Three things make that safe, and each was a bug somewhere:

    - TWIST ONLY. rf2o integrates a pose from its own first scan; fused as a pose it would drag
      odom -> base_link onto a second trajectory beside the wheels' (the lesson odom0 learned on
      2026-09-08). vx and vyaw are velocities and have no origin to disagree about.
    - NO TRANSFORM. ``publish_tf`` false: the filter owns odom -> base_link, and two publishers
      of one edge is the oldest failure in this stack.
    - A COVARIANCE THE FILTER CAN WEIGH. Upstream publishes an empty matrix, and
      robot_localization raises a zero variance to 1e-9 (ekf.cpp) and then believes the source
      over the wheels and the gyro both. The number is stamped where the message is born, as the
      camera's is — three parameters the patched node takes from the launch.

    And it is a source of the filter, never its switch: one operator gesture, one env var, one
    launch argument, and the way back is the stack of the day before.
    """
    from pepin.deployment import LASER_ODOM_HZ, LASER_ODOM_TOPIC, LASER_ODOM_TWIST_VARIANCE

    ekf = yaml.safe_load((REPO / "ros/params/ekf.yaml").read_text())["ekf_filter_node"][
        "ros__parameters"
    ]
    assert ekf["odom3"] == LASER_ODOM_TOPIC
    assert ekf["odom3_config"][6] is True, "vx: the lidar's witness of distance"
    assert ekf["odom3_config"][11] is False, "vyaw: +4.3 deg/min at rest, the parked heading creep"
    assert not any(ekf["odom3_config"][:6]), "no pose: it would fight the wheels' integration"
    assert ekf["odom3_config"][7] is False, "the wheels' vy = 0 is the kinematic truth, not this"
    assert ekf["odom3_differential"] is False, "a velocity is already differential"
    assert ekf["odom3_twist_rejection_threshold"] > 0, "a corridor makes it wrong and confident"
    # The one file the wheels and the gyro live in is untouched by this source.
    assert ekf["odom0"] == "odom" and ekf["imu0"] == "imu/data_raw"
    robot = sf.tree(ROBOT_LAUNCH)
    node = next(
        c
        for c in sf.calls_to(robot, "Node")
        if ast.unparse(sf.keywords(c).get("package", ast.Constant(None))) == "'rf2o_laser_odometry'"
    )
    # Read as source, not as values: the point of half of these is that the launch spells no
    # number of its own — every one of them comes from the one constant in pepin.deployment.
    params = {
        ast.literal_eval(key): ast.unparse(value)
        for table in ast.walk(node)
        if isinstance(table, ast.Dict)
        for key, value in zip(table.keys, table.values, strict=True)
    }
    assert params["publish_tf"] == "False", "the EKF owns odom -> base_link"
    assert params["laser_scan_topic"] == "'/scan'"
    assert "LASER_ODOM_TOPIC" in params["odom_topic"], "one name, in pepin.deployment"
    assert (params["base_frame_id"], params["odom_frame_id"]) == ("'base_link'", "'odom'")
    assert params["freq"] == "LASER_ODOM_HZ"
    assert 0 < LASER_ODOM_HZ <= 10.0, (
        "the loop consumes the newest scan each turn: above the LD19's own 10 Hz it only finds"
        " nothing to do, and below it it throws scans away"
    )
    assert params["init_pose_from_topic"] == "''", (
        "upstream's default is /base_pose_ground_truth, a simulator topic nothing here publishes,"
        " and the node processes no scan at all until one arrives on it"
    )
    # The patch's two switches, both at the values that make the message honest.
    assert params["base_frame_twist"] == "True", (
        "upstream's twist.linear.x is the LASER's own x step per second, and this lidar hangs"
        " upside down and yawed -87.5 deg (config/lidar.json)"
    )
    for axis in ("vx", "vy", "vyaw"):
        assert params[f"twist_covariance_{axis}"] == f"LASER_ODOM_TWIST_VARIANCE[{axis!r}]", axis
        assert LASER_ODOM_TWIST_VARIANCE[axis] > 0, f"{axis}: 0 is 1e-9 to the filter, not unknown"
    patch = (REPO / "ros/patches/rf2o-base-twist.patch").read_text()
    dockerfile = (REPO / "ros/Dockerfile").read_text()
    assert "rf2o-base-twist.patch" in dockerfile and "rf2o_laser_odometry" in dockerfile
    assert "ARG RF2O_COMMIT=" in dockerfile and "checkout --detach" in dockerfile, (
        "pinned by commit: a moved branch rebuilds this board for an hour with code nobody read"
    )
    for parameter in ("base_frame_twist", "twist_covariance_vx", "twist_covariance_vyaw"):
        assert f'declare_parameter<bool>("{parameter}"' in patch or (
            f'declare_parameter<double>("{parameter}"' in patch
        ), f"the patch is what teaches rf2o {parameter}"
    # Reachable end to end: one gesture, one env var, one argument.
    assert "'laser_odom'" in {
        ast.unparse(c.args[0]) for c in sf.calls_to(robot, "DeclareLaunchArgument")
    }
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    assert "LaunchConfiguration('laser_odom')" in sf.unparsed(bringup, ast.Call)
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "laser_odom:=${PEPIN_LASER_ODOM}" in unit
    assert "laser_odom) VAR=PEPIN_LASER_ODOM ;;" in (REPO / "ros/feature.sh").read_text()
    # Declared on the board with a budget before it is allowed to run there (CLAUDE.md rule 20).
    entry = next(
        p
        for p in json.loads((REPO / "config/board_manifest.json").read_text())["processes"]
        if p["name"] == "laser_odometry"
    )
    assert set(entry["on_board_because"]) == {"real-time", "wifi-loss"}
    assert entry["budget"]["cpu_percent"] > 0 and entry["budget"]["rss_mb"] > 0
    assert "/odom_laser" in (REPO / "ros/restart.sh").read_text(), "a restart asks whether it flows"


def test_the_visual_odometry_reaches_the_ekf_without_being_able_to_move_the_odom_frame() -> None:
    """The camera's own odometry is a third input to the EKF, and the shape of that input is
    the whole safety argument: rtabmap_odom's rgbd_odometry on the laptop, gated by
    pepin_bringup.visual_odometry, fused DIFFERENTIALLY (two poses differenced into a velocity,
    so this source's origin — and its restarts — can never move odom -> base_link) in x, y and —
    since 2026-09-15 — yaw, which is the same differential path and therefore the same safety
    argument. The gyro still owns heading: at vo_yaw_sigma_deg's 5 degrees the camera carries a
    quarter of the gyro's weight per sample and ~5 % of its information per second, which is a
    second opinion that keeps a heading alive if the MPU6050 dies, not a rival (measured
    2026-09-13: with the gyro the EKF's turn error is ~5 %, the wheels alone over-report a turn
    by 40-70 %)."""
    ekf = yaml.safe_load((REPO / "ros/params/ekf.yaml").read_text())["ekf_filter_node"][
        "ros__parameters"
    ]
    assert ekf["odom0"] == "odom" and ekf["imu0"] == "imu/data_raw", "the wheels and the gyro stay"
    assert ekf["odom0_config"][6] is True and ekf["odom0_config"][:6] == [False] * 6
    assert ekf["imu0_config"][11] is True, "the gyro's yaw rate, as before"
    assert ekf["odom1"] == "vo"
    pose_x, pose_y, *_rest = ekf["odom1_config"]
    assert pose_x is True and pose_y is True
    assert ekf["odom1_config"][5] is True, "the camera's yaw, differenced into a yaw rate"
    assert ekf["odom1_config"][11] is False, (
        "index 11 would read rtabmap's own twist, which the gate never rewrites; the fused"
        " heading must come through the differential path that the gate's covariance sizes"
    )
    assert not any(ekf["odom1_config"][2:5]), "z, roll and pitch are not the camera's either"
    assert ekf["odom1_differential"] is True, "a VO restart must never jump the odom frame"
    assert ekf["odom1_pose_rejection_threshold"] > 0, "an outlier may not yank odom"
    from pepin.deployment import bridged_qos

    assert bridged_qos("/vo") == ("reliable", ekf["odom1_queue_size"]), (
        "robot_localization subscribes RELIABLE at odom1_queue_size: the laptop's publisher"
        " must agree, or the reader and the writer do not match"
    )
    assert bridged_qos("/odom") == ("reliable", 10), (
        "the rest watch reads the board's wheels from the laptop, and base_bridge.cpp writes"
        " /odom RELIABLE ten deep"
    )
    source = (REPO / "ros/pepin_bringup/pepin_bringup/visual_odometry.py").read_text()
    for topic in ("VO_TOPIC", "WHEELS_TOPIC"):
        assert f"bridged_qos_profile({topic})" in source, (
            f"{topic} crosses the link: its endpoint takes the pinned QoS, not one of its own"
        )


def test_the_visual_odometry_runs_on_the_laptop_behind_one_launch_switch() -> None:
    """CLAUDE.md rule 20: what consumes the camera lives on the laptop, and the board gets a
    finished measurement. Nothing new runs there — the EKF reads one more topic. On this side
    both processes are one argument (``vo``), started like every other node of this launch,
    rgbd_odometry publishes no transform (the EKF owns odom -> base_link)
    and takes no guess from TF (a visual odometry seeded with the filter's own answer is not a
    third opinion)."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "'vo'" in {ast.unparse(c.args[0]) for c in sf.calls_to(vslam, "DeclareLaunchArgument")}
    table = dict(ast.literal_eval(sf.assignments(vslam)["VISUAL_ODOMETRY"]))
    assert table["publish_tf"] is False and table["odom_frame_id"] == "odom"
    assert table["guess_frame_id"] == ""
    assert table["approx_sync"] is False, (
        "the depth carries its own picture's stamp and the CameraInfo the same one"
        " (1024/1024 and 1397/1397 bit-equal on the wire, 2026-09-14): there is one correct"
        " pair per depth frame, and ApproximateTime took the neighbouring picture instead"
    )
    assert table["publish_null_when_lost"] is True, "a lost frame must be countable"
    node = _node_named(vslam, "rgbd_odometry")
    keywords = sf.keywords(node)
    assert ast.unparse(keywords["package"]) == "'rtabmap_odom'"
    assert "VISUAL_ODOMETRY" in ast.unparse(keywords["parameters"])
    assert "'base_link'" in ast.unparse(keywords["parameters"]), "the pose is the cart's"
    remapped = ast.unparse(keywords["remappings"])
    for pair in (
        "('rgb/image', '/camera/image')",
        "('rgb/camera_info', '/camera/camera_info')",
        "('depth/image', '/camera/depth')",
        "('odom', VO_RAW_TOPIC)",
    ):
        assert pair in remapped, remapped
    assert ast.literal_eval(sf.assignments(vslam)["VO_RAW_TOPIC"]) == "/vo/raw", (
        "rgbd_odometry's raw output is kept off /vo until the gate has seen it"
    )
    assert "LaunchConfiguration('vo')" in ast.unparse(keywords["condition"])
    # vo_input:=stereo: the same role, the same table and output, on the two rectified eyes —
    # all four topics carry one stamp, so it runs at the camera's rate without the depth.
    declared = {
        ast.unparse(c.args[0]): sf.keywords(c) for c in sf.calls_to(vslam, "DeclareLaunchArgument")
    }
    assert ast.unparse(declared["'vo_input'"]["default_value"]) == "'stereo'", "today's input"
    assert ast.literal_eval(sf.assignments(vslam)["VO_INPUTS"]) == ("depth", "stereo")
    stereo = sf.keywords(_node_named(vslam, "stereo_odometry"))
    assert ast.unparse(stereo["executable"]) == "'stereo_odometry'"
    assert ast.unparse(stereo["parameters"]) == ast.unparse(keywords["parameters"])
    assert ast.unparse(stereo["condition"]) == ast.unparse(keywords["condition"])
    remapped = ast.unparse(stereo["remappings"])
    for pair in (
        "('left/image_rect', '/camera/image')",
        "('left/camera_info', '/camera/camera_info')",
        "('right/image_rect', '/camera/right/image')",
        "('right/camera_info', '/camera/right/camera_info')",
        "('odom', VO_RAW_TOPIC)",
    ):
        assert pair in remapped, remapped
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "--vo-depth) VO_INPUT=depth ;;" in laptop and '"vo_input:=$VO_INPUT"' in laptop
    gate = next(
        c
        for c in sf.calls_to(vslam, "ExecuteProcess")
        if "pepin_bringup.visual_odometry" in ast.unparse(c)
    )
    assert "LaunchConfiguration('vo')" in ast.unparse(sf.keywords(gate)["condition"])
    started = _started_by_describe(vslam)
    assert {"odometry", "vo"} <= started, "both start with the launch"
    from pepin.deployment import LAPTOP_SLAM_NODES

    assert {"/rgbd_odometry", "/stereo_odometry", "/visual_odometry"} <= set(LAPTOP_SLAM_NODES), (
        "ros/flags.sh finds both in the SLAM container"
    )


def test_the_visual_memory_survives_a_kill_and_the_launches_wait_for_it_to_close() -> None:
    """RTAB-Map's database is the map beside a known map, and 0.22.1's own defaults lose it to a
    SIGKILL: the rollback journal lives in RAM (JournalMode 3 = MEMORY) and nothing is ever
    flushed (Synchronous 0 = OFF), so a killed process leaves a torn file — which is what
    ros/maps/rtabmap.db became on 2026-09-13. Two halves of the fix, and a contract on each:
    an on-disk journal so a kill is survivable, and a shutdown long enough that a stop is not a
    kill in the first place (launch escalates to SIGTERM after 5 s by default, and a 20-28 GB
    database does not close in five seconds)."""
    from pepin.deployment import CONTAINER_STOP_TIMEOUT_S

    table = _rtabmap("RTABMAP")
    assert table["DbSqlite3/JournalMode"] in {"0", "1"}, (
        "MEMORY (3, the default), PERSIST and OFF all lose the journal a rollback needs"
    )
    assert table["DbSqlite3/Synchronous"] in {"1", "2"}, "0 = OFF never reaches the disk"

    for path in (VSLAM_LAUNCH, NAV_LAUNCH, "ros/pepin_bringup/launch/bringup.launch.py"):
        launch = sf.tree(path)
        budget = {
            ast.literal_eval(call.args[0]): ast.unparse(call.args[1])
            for call in sf.calls_to(launch, "SetLaunchConfiguration")
            if len(call.args) == 2 and isinstance(call.args[0], ast.Constant)
        }
        assert budget.keys() >= {"sigterm_timeout", "sigkill_timeout"}, path
        # The constant itself, not a copy of today's value: one number for the whole stop path.
        assert budget["sigterm_timeout"] == "str(CONTAINER_STOP_TIMEOUT_S)", path
        assert 0 < float(ast.literal_eval(budget["sigkill_timeout"])) <= CONTAINER_STOP_TIMEOUT_S, (
            path
        )
        first = ast.unparse(sf.calls_to(launch, "LaunchDescription")[0].args[0].elts[0])
        assert first == "*SHUTDOWN", f"{path}: the budget must be set before anything it covers"


def test_goto_s_cancel_cancels_a_goal_it_never_sent() -> None:
    """`ros/goto.sh cancel` crashed in rclpy's teardown ("the given context is not valid") and
    cancelled nothing even when it did not: nav2_simple_commander's cancelTask cancels the goal
    THIS process sent, and a process started to type "cancel" has sent none. The action's own
    cancel service takes a zero goal id, which means every goal, and needs no handle."""
    goto = sf.tree("ros/tools/goto_ros.py")
    assert "CancelGoal" in sf.imported(goto)
    cancel = next(
        f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "cancel_all"
    )
    body = ast.unparse(cancel)
    assert "_action/cancel_goal" in body and "CancelGoal.Request()" in body
    assert "spin_until_future_complete" in body and "CANCEL_CONFIRM_S" in body
    assert float(sf.assignments(goto)["CANCEL_CONFIRM_S"]) <= 30.0, "confirmed while he watches"
    main = next(f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "main")
    lines = ast.unparse(main).splitlines()
    cancelled = next(i for i, line in enumerate(lines) if "cancel_all(node)" in line)
    built = next(i for i, line in enumerate(lines) if "BasicNavigator()" in line)
    assert cancelled < built, "no commander is built for a cancel: building one is what crashed"
    assert "rclpy.ok()" in ast.unparse(main), "a context already shut down refuses every call"
    assert "stop.sh" in " ".join(sf.strings(goto)), "the hard stop is named where cancel can fail"


def test_zenoh_router_starts_before_the_stack_and_outlives_its_restarts() -> None:
    """The board's router is its own unit: ordered before the stack, never dragged by it."""
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    assert "Before=pepin-ros.service" in unit, "the router is up before the nodes that dial it"
    assert "PartOf=" not in unit and "BindsTo=" not in unit, (
        "a router restarted with every stack restart would drop the laptop's half each time"
    )
    assert "Restart=always" in unit
    assert "--network host" in unit, "the board's nodes reach it over the host loopback"
    assert "rmw_zenohd" in unit
    assert "WantedBy=multi-user.target" in unit, "it comes back on its own after a reboot"


def test_the_camera_rig_decides_who_measures_the_depth() -> None:
    """One choice, the rig's name: a stereo head gets the matched eyes as its depth source, the
    webcam gets the network, and the depth node is told at launch (it decides its subscriptions)."""
    launch = (REPO / VSLAM_LAUNCH).read_text()
    assert 'f"depth_source:={depth_source(rig)}"' in launch
    assert 'return "stereo" if CameraConfig.load(config_file("camera.json"), rig).stereo' in launch
    config = json.loads((REPO / "config/camera.json").read_text())
    assert "rig" in config["stereo"] and "rig" not in config["overview"]


def test_the_lidar_and_the_base_die_apart_and_each_comes_back() -> None:
    """A dead lidar is the failure a camera-only cart must ride out, so it must not take the
    wheels with it: on 2026-09-24 the LD19 driver aborted on a deactivate and, sharing one
    process with the base bridge, left the cart with no wheels, no IMU and no gyro-bias tracker
    for hours. Two processes, each respawned; and the deactivate that aborts the driver
    (sensor.sh --hard) is refused with the reason."""
    src = (REPO / "ros/pepin_bringup/launch/robot.launch.py").read_text()
    assert "ComposableNodeContainer(" not in src, "no shared process to fall back to"
    lidar_block = src[src.index("def lidar_parts(") : src.index("def base_parts(")]
    assert 'name="ldlidar_node"' in lidar_block and 'name="scan_filter"' in lidar_block
    base_block = src[src.index("def base_parts(") : src.index("def sensors_container(")]
    assert 'package="pepin_base_cpp"' in base_block
    assert 'package="pepin_base_cpp"' not in lidar_block
    # A respawn restores the nodes, wired: each start loads descriptions its factory built anew
    # (pepin_bringup.launch_kit; the generator trap is held in test_launch_kit).
    split = src[src.index("def sensors_container(") : src.index("def generate_launch_description(")]
    assert 'respawned_container("lidar_container", lidar_parts,' in split
    assert 'respawned_container("base_container", base_parts,' in split
    assert "def respawned_container(" not in src, "one helper, pepin_bringup.launch_kit"
    names = {
        p["name"]
        for p in json.loads((REPO / "config/board_manifest.json").read_text())["processes"]
    }
    assert {"lidar_container", "base_container"} <= names and "sensors_container" not in names
    assert "refused: --hard" in (REPO / "ros/sensor.sh").read_text()
