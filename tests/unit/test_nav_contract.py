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

from pepin.deployment import rmw_is_zenoh
from pepin.flags import load_table
from pepin.watch import PAINT_SIGMA_M

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


def _started_after_ghost_wait(launch: ast.Module) -> set[str]:
    """What the launch's ``OnProcessExit(target_action=ghost_wait, on_exit=...)`` starts: the
    names of a list literal, or the one name that holds the list (nav: ``actions``)."""
    handlers = sf.calls_to(launch, "OnProcessExit")
    assert len(handlers) == 1
    handler = sf.keywords(handlers[0])
    assert ast.unparse(handler["target_action"]) == "ghost_wait"
    started = handler["on_exit"]
    if isinstance(started, ast.List):
        return {ast.unparse(e) for e in started.elts}
    return {ast.unparse(started)}


def _rtabmap(table: str) -> dict[str, object]:
    """One of vslam.launch.py's RTAB-Map tables as a dict. Since 2026-09-19 there is ONE — the
    ``RTABMAP`` table that serves every situation — beside ``LOCALIZE``, ``TF_ODOMETRY_VARIANCE``
    and ``TRIPLE_SUBSCRIPTIONS`` (the way back to the three subscriptions), so a contract names
    the one it means instead of the union of every dict literal in the file."""
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))[table]))


def _case_blocks(script: str) -> dict[str, str]:
    """A shell script's top-level ``case`` branches by label: ``on)`` up to the next label."""
    labels = list(re.finditer(r"^    (\w+|\*)\)", script, re.M))
    ends = [m.start() for m in labels[1:]] + [len(script)]
    return {m.group(1): script[m.end() : end] for m, end in zip(labels, ends, strict=True)}


def _p(node: str) -> dict:  # type: ignore[type-arg]
    block = PARAMS[node]
    if (
        "ros__parameters" not in block
    ):  # the costmaps nest one level deeper: node/node/ros__parameters
        block = block[node]
    return block["ros__parameters"]  # type: ignore[no-any-return]


def test_progress_and_manoeuvres_are_judged_against_the_map_not_the_wheels() -> None:
    # 2026-09-06: a slipping wheel fed odom a metre of motion; Nav2 saw progress for 58 s.
    assert _p("local_costmap")["global_frame"] == "map"
    assert _p("behavior_server")["local_frame"] == "map"


def test_stuck_is_declared_within_seconds_and_short_goals_stay_reachable() -> None:
    checker = _p("controller_server")["progress_checker"]
    assert checker["required_movement_radius"] <= 0.15
    assert checker["movement_time_allowance"] <= 6.0


def test_turning_in_place_counts_as_progress() -> None:
    """2026-09-08: a 70 deg pivot at the base was called 'no progress' and answered by reversing."""
    checker = _p("controller_server")["progress_checker"]
    assert checker["plugin"].endswith("PoseProgressChecker")
    assert 0.0 < checker["required_movement_angle"] <= 0.6


def test_an_auxiliary_sensor_cannot_stall_the_costmap() -> None:
    # 2026-09-07: the range layer went stale once (clock jump) and every goal died for 13 minutes.
    costmap = _p("local_costmap")
    for sensor in ("front", "left", "right"):
        assert costmap[f"tof_{sensor}_layer"]["no_readings_timeout"] == 0.0


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

    Two shapes of the same promise. The kept RangeSensorLayer blocks say it with
    ``inflate_cone: 1.0`` over the real 27 degree ``phi``; the fan that feeds the ObstacleLayers
    says it by putting the one measured distance on EVERY beam of the cone, and by having enough
    beams that no costmap cell inside the arc is skipped at the sensor's own ceiling.
    """
    from pepin.tof_horizon import cone_beams

    for costmap in ("local_costmap", "global_costmap"):
        for sensor in ("front", "left", "right"):
            layer = _p(costmap)[f"tof_{sensor}_layer"]
            assert layer["inflate_cone"] == 1.0
            assert layer["phi"] <= 0.5, "phi must model the real 27 degree cone"
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
    # ...and the blocks of the plugin they replaced are kept, unlisted, for the way back.
    for sensor in ("front", "left", "right"):
        assert _p("local_costmap")[f"tof_{sensor}_layer"]["topics"] == [f"/tof/{sensor}"]


def test_the_whiskers_do_not_run_in_the_plugin_with_the_unbounded_loop() -> None:
    """2026-09-21, the second RangeSensorLayer defect and the reason the ToF left it.

    ``range_sensor_layer.cpp:362-369`` clamps ``bx0``/``by0`` at zero and ``bx1``/``by1`` at the
    grid's size, then walks ``for (unsigned int x = bx0; x <= (unsigned int)bx1; x++)``: a cone
    entirely off the grid's left or bottom edge leaves the upper bounds NEGATIVE, the cast makes
    them about 4e9, and one thread spins for ever holding the costmap's mutex. It needs one jump
    of the pose in ``map`` between a reading's stamp and the update — a tracker restart, a
    relocalisation, the cart lifted — so no publisher can gate it the way ``tf_gate`` gates the
    first defect. Reproduced with ``ros/thin.sh kick relocalizer`` (tid 191 of the Nav2
    container: 415 s of CPU in 700 s). So: no costmap may LIST a RangeSensorLayer, the whiskers
    are ObstacleLayers fed by a fan, and the flag that goes back is live.
    """
    for costmap in ("local_costmap", "global_costmap"):
        parameters = _p(costmap)
        for name in parameters["plugins"]:
            assert not parameters[name]["plugin"].endswith("RangeSensorLayer"), (
                f"{costmap}.{name}: the unbounded loop is one pose jump away"
            )
    flags = load_table(REPO / NODES / "tof_bridge.py")
    assert flags["range_as"] == "scan" and flags.flag("range_as").live
    assert flags.flag("range_as").choices == ("scan", "range")
    source = (REPO / "ros/params/nav2_params.yaml").read_text()
    assert "range_sensor_layer.cpp:362-369" in source, "the defect is named where it is answered"


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


def test_the_tof_whiskers_serve_the_local_costmap_only() -> None:
    """2026-09-21: the ToF are short whiskers for the controller's map. In the global costmap
    they bought a room-scale plan nothing and were the worst amplifier of the RangeSensorLayer
    wedge (transform_tolerance 1.0 s x 15 Hz = 15x per update cycle), which hung planner_server's
    activation on 4 of 7 board starts. Their blocks stay in the file: returning them is one
    line of the plugin list."""
    plugins = _p("global_costmap")["plugins"]
    assert not [name for name in plugins if name.startswith("tof_")], plugins
    for sensor in ("front", "left", "right"):
        assert _p("global_costmap")[f"tof_{sensor}_layer"]["topics"] == [f"/tof/{sensor}"]


def test_a_pivot_the_cart_cannot_make_is_preferred_less_than_an_arc() -> None:
    """The rear corners sweep 0.407 m; RPP checks 7.03 deg per step, so the speed sets the sweep."""
    follow = _p("controller_server")["FollowPath"]
    assert follow["rotate_to_heading_angular_vel"] <= 0.5
    assert follow["rotate_to_heading_min_angle"] >= 1.0
    # A Spin judged over 2 s of yaw needs the whole swing circle and is refused before it starts.
    assert _p("behavior_server")["simulate_ahead_time"] <= 1.0


def test_the_behavior_tree_backs_up_first_and_keeps_trying() -> None:
    bt_path = _p("bt_navigator")["default_nav_to_pose_bt_xml"]
    tree = ET.parse(REPO / "ros/params" / Path(bt_path).name)
    recovery = tree.find(".//RecoveryNode[@name='NavigateRecovery']")
    assert recovery is not None and int(recovery.get("number_of_retries", "0")) >= 8
    round_robin = tree.find(".//RoundRobin[@name='RecoveryActions']")
    assert round_robin is not None
    children = list(round_robin)
    assert len(children) >= 6
    assert children[1].tag == "BackUp" or children[0].tag == "BackUp"
    known = {"BackUp", "Spin", "Wait", "DriveOnHeading", "Sequence", "ClearEntireCostmap"}
    assert {child.tag for child in round_robin.iter()} - {"RoundRobin"} <= known


def test_the_operator_scripts_parse_and_keep_their_safety_lines() -> None:
    for script in (
        "stop.sh",
        "goto.sh",
        "go.sh",
        "tour.sh",
        "lib.sh",
        "mode.sh",
        "map.sh",
        "feature.sh",
        "laptop.sh",
        "thin.sh",
        "flags.sh",
        "restart.sh",
        "tools/coldstart_soak.sh",
    ):
        subprocess.run(["bash", "-n", str(REPO / "ros" / script)], check=True)
    stop = (REPO / "ros/stop.sh").read_text()
    assert "timeout 3" in stop and "pkill -9 -f" in stop and "systemctl restart pepin-ros" in stop
    goto = (REPO / "ros/goto.sh").read_text()
    assert "tail -n +1 -F" in goto and re.search(r"trap .*EXIT", goto)
    assert re.search(r'/stop\.sh"', goto), "Ctrl-C must run stop.sh"
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
    The global costmap keeps the tuned inflation pair and follows a moving person at 2 Hz."""
    assert _p("planner_server")["Smac2D"]["cost_travel_multiplier"] >= 5.0
    assert "cost_factor" not in _p("planner_server")["GridBased"], "nav2's NavFn has no such knob"
    inflation = _p("global_costmap")["inflation_layer"]
    assert (inflation["inflation_radius"], inflation["cost_scaling_factor"]) == (0.55, 2.0)
    assert _p("global_costmap")["update_frequency"] >= 2.0
    follow = _p("controller_server")["FollowPath"]
    assert follow["max_allowed_time_to_collision_up_to_carrot"] >= 0.7
    local = _p("local_costmap")["inflation_layer"]
    assert follow["inflation_cost_scaling_factor"] == local["cost_scaling_factor"]
    assert follow["cost_scaling_dist"] <= local["inflation_radius"]


def test_the_tof_layers_never_stall_either_costmap() -> None:
    """A whisker that goes quiet must never be able to stop the robot, and since 2026-09-21 its
    silence is a DESIGNED state: tof_bridge's tf_gate withheld 480 readings over 8 s while the
    tracker restarted. So the scan sources carry no expected_update_rate (a buffer given a rate
    calls itself stale and Nav2 answers every goal with "Costmap timed out waiting for update",
    2026-09-07), and the kept range layers keep their no_readings_timeout."""
    for sensor in ("front", "left", "right"):
        layer = _p("local_costmap")[f"tof_{sensor}_scan_layer"]
        assert layer[f"tof_{sensor}_scan"]["expected_update_rate"] == 0.0
    for costmap in ("local_costmap", "global_costmap"):
        for sensor in ("front", "left", "right"):
            assert _p(costmap)[f"tof_{sensor}_layer"]["no_readings_timeout"] == 0.0


def test_the_range_layers_are_fed_only_what_they_can_be_asked_to_transform() -> None:
    """The Nav2 wedge of 2026-09-21. tf2's canTransform blocks the WHOLE timeout on any failure
    and RangeSensorLayer calls it once per message with the message's own stamp, so three layers
    at 15 Hz against a 0.3 s (local) and 1.0 s (global) tolerance amplify 4.5x and 15x: the
    backlog outgrows the drain, every message ages past the 10 s TF cache, the first costmap
    update never ends and planner_server hangs in Activating (scratch/nav2_hang/wedge_gain.py).
    The tolerances stay where they are — stability would need one under 67 ms, below this
    robot's own TF latency — and the PUBLISHER holds the precondition instead: no Range leaves
    tof_bridge while the chain that must place it is broken, and the mounts go out with every
    reading so no late joiner can be missing the frame. Both are live switches with the old
    behaviour one `ros/flags.sh set` away (rule 19)."""
    assert _p("local_costmap")["transform_tolerance"] == 0.3
    assert _p("global_costmap")["transform_tolerance"] == 1.0
    flags = load_table(REPO / NODES / "tof_bridge.py")
    assert flags.names == ("dynamic_mounts", "tf_gate", "tf_gate_max_lag_s", "range_as")
    assert all(flag.live for flag in flags), "a wedge is turned off in the field, not reverted"
    assert flags["dynamic_mounts"] is True and flags["tf_gate"] is True
    facts = sf.assignments(sf.tree(f"{NODES}/tof_bridge.py"))
    assert facts["_GLOBAL_FRAME"] == "'map'", "the frame the costmaps place a cone in"
    assert float(facts["_STAMP_LAG_S"]) < flags["tf_gate_max_lag_s"], (
        "a reading must not be born already too stale for the gate that judges it"
    )


def test_a_cold_start_is_judged_and_can_be_soaked() -> None:
    """The gap that let the wedge ship: the transport's acceptance had no "N cold starts, Nav2
    fully active" test, so a hang that appears on 4 of 7 starts looked like bad luck. The
    restart's checks now name it, and the soak repeats it without ever moving the robot."""
    restart = (REPO / "ros/restart.sh").read_text()
    assert "planner_server connected with bond" in restart, "active, not merely running"
    assert "Range sensor layer can't transform" in restart, "the wedge's own line"
    soak = REPO / "ros/tools/coldstart_soak.sh"
    assert soak.stat().st_mode & 0o111, "it is run as a command"
    text = soak.read_text()
    assert "nav_goal_running.py" in text, "it refuses to restart the stack under a live goal"
    assert "THE ROBOT DOES NOT MOVE" in text
    assert "systemctl restart pepin-ros" in text, "restart.sh's own mechanism"
    # The other half of a cold start: `docker run` returns before rmw_zenohd accepts, and the
    # stack started behind a router that was still binding is what delayed /tf_static by 157 s.
    router = (REPO / "board/pepin-zrouter.service").read_text()
    assert "ExecStartPost=" in router and "/dev/tcp/127.0.0.1/7447" in router


def test_two_controllers_and_only_the_footprint_planner_may_plan_a_reverse() -> None:
    """FollowPath never reverses and may pivot; FollowPathRS follows the cusps of a Hybrid-A*
    plan and, as RPP demands, gives up rotate-to-heading for it. The lattice stays forward-only."""
    cs = _p("controller_server")
    assert cs["controller_plugins"] == ["FollowPath", "FollowPathRS"]
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
    assert catalogue["hybrid"] == ("Hybrid", "FollowPathRS")
    assert "hybrid" in (REPO / "ros/go.sh").read_text()


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


def test_the_cart_drives_at_the_base_s_full_speed_and_nothing_clips_it() -> None:
    """One speed cap, the base's own: the controller asks for it, the smoother lets it through,
    and the velocity-scaled lookahead has room for it. Every tape until 2026-09-09 sat at 0.20 m/s
    because two Nav2 numbers said so while the wheels had never been asked for more."""
    from pepin.deployment import BASE_MAX_ANGULAR_RAD_S, BASE_MAX_LINEAR_M_S

    cfg = json.loads((REPO / "config/base.json").read_text())
    base = cfg["max_speed_m_s"]
    assert (base, cfg["max_yaw_rate_rad_s"]) == (BASE_MAX_LINEAR_M_S, BASE_MAX_ANGULAR_RAD_S)
    robot = sf.dict_items(sf.tree(ROBOT_LAUNCH))
    assert robot["max_linear_m_s"] == {"BASE_MAX_LINEAR_M_S"}, "the C++ bridge keeps its 0.25 cap"
    follow = _p("controller_server")["FollowPath"]
    smoother = _p("velocity_smoother")
    assert follow["desired_linear_vel"] == smoother["max_velocity"][0], (
        "two caps: one wins silently"
    )
    assert follow["desired_linear_vel"] <= base, "the base clamps anything above its own cap"
    assert follow["desired_linear_vel"] >= 0.30, "the base allows 0.30 m/s: ask for it"
    assert follow["max_lookahead_dist"] >= follow["lookahead_time"] * follow["desired_linear_vel"]
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
    assert _p("controller_server")["failure_tolerance"] >= 3.0, "a person needs time to step aside"


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


def test_the_board_half_does_not_autostart_and_the_laptop_knows_its_side() -> None:
    nav = sf.dict_items(sf.tree(NAV_LAUNCH))
    assert "autostart_for(side)" in nav["autostart"]
    assert "side" in nav["side"], "the goal server must know it is the laptop half"
    server = sf.tree(f"{NODES}/goal_server.py")
    assert "next_transition" in sf.calls(server) and "ChangeState" in sf.imported(server)


def test_a_goal_without_a_tracker_is_judged_on_the_transform_the_slam_half_publishes() -> None:
    """In SLAM mode nothing publishes /localization_fit, and the goal server used to refuse
    every goal with "the tracker is not up" (2026-09-13 14:05). The rule lives in
    pepin.watch.GoalGate, the node only supplies the two readings — the tracker's fit, or the
    age of map -> base_link — and the blind-drive watch, which reads a fit, is armed only where
    a tracker publishes one."""
    server = sf.tree(f"{NODES}/goal_server.py")
    assert {"GoalGate", "Readiness", "TF_FRESH_S"} <= sf.imported(server), "the rule is pepin's"
    assert "TfLookup" in sf.imported(server) and "self._tf.transform" in sf.calls(server)
    assert sf.assignments(server)["MAP_FRAME"] == "'map'"
    assert sf.assignments(server)["BASE_FRAME"] == "'base_link'"
    assert "self._gate.verdict" in sf.calls(server)
    armed = [
        s for s in sf.unparsed(server, ast.IfExp) if s.startswith("BlindDriveWatch() if ready")
    ]
    assert armed and armed[0].endswith("else None"), "no fit to watch without a tracker"
    # The fallback is a flag, so the old behaviour is one `ros/flags.sh set` away (rule 19).
    # start_on_a_known_pose joined the table on 2026-09-19: with a sigma published, a goal is
    # refused for the pose's sake only where nothing has ever corrected it.
    flags = load_table(REPO / NODES / "goal_server.py")
    assert flags.names == (
        "tf_pose",
        "correction_watch",
        "sigma_gate",
        "places_from_the_file",
        "start_on_a_known_pose",
    )
    assert all(flags.flag(name).live for name in flags.names)
    assert "self._switches.state" in sf.calls(server), "and it is printed in the node's own line"


def test_the_slam_drive_is_judged_on_the_correction_and_not_on_the_edge_it_feeds() -> None:
    """A fresh map -> base_link proves nothing about the laptop: pepin_bringup.slam_frame
    re-broadcasts the LAST correction at 10 Hz with a fresh stamp, so the edge stays
    milliseconds old with the laptop shut down — the gate passed and Nav2, whose costmaps read
    that same edge against a 0.3 s tolerance, never aborted either. So the goal server listens
    to the correction itself, on the topic the board's frame node reads, and cuts the drive when
    the pulse stops: the SLAM analogue of the blind-drive watch, which has no fit to read."""
    server = sf.tree(f"{NODES}/goal_server.py")
    assert sf.assignments(server)["CORRECTION_TOPIC"] == "'/map_odom'"
    assert "self.create_subscription" in sf.calls(server)
    assert "Correction" in sf.imported(server), "the rule is pepin.watch's, not the node's"
    assert "self._correction" in sf.calls(server)
    cut = [s for s in sf.unparsed(server, ast.Call) if ".stale(" in s]
    assert cut, "the drive loop asks the same question the gate does"
    assert "handle.cancel_goal_async" in sf.calls(server)
    # The board's own node never stops broadcasting for a silence: Nav2 there would lose its
    # global frame to a WiFi hiccup. It says so in the log, and the decision lives in the gate.
    frame = sf.tree(f"{NODES}/slam_frame.py")
    assert "self._report_silence" in sf.calls(frame)
    assert "self._tf.sendTransform" in sf.calls(frame)


def test_the_recorder_is_its_own_node_on_the_board_side() -> None:
    """The split's first tapes were written on the laptop and had no scans: the recorder lives
    where the sensors are, and the goal server only sends it a command."""
    nav = sf.tree(NAV_LAUNCH)
    assert "runs_here(side, 'run_recorder')" in sf.unparsed(nav, ast.Call)
    assert "pepin_bringup.run_recorder" in sf.strings(nav)
    setup = sf.dict_items(sf.tree("ros/pepin_bringup/setup.py"))
    scripts = ast.literal_eval(next(iter(setup["console_scripts"])))
    assert "run_recorder = pepin_bringup.run_recorder:main" in scripts
    server = sf.tree(f"{NODES}/goal_server.py")
    assert "RunRecorder" not in sf.calls(server) and "RunRecorder" not in sf.imported(server)
    assert not any("curl" in s for s in sf.strings(server))
    assert {"RUN_COMMAND_TOPIC", "RUN_STATUS_TOPIC"} <= sf.names(server)


def test_both_doors_to_a_goal_open_the_numbered_tape() -> None:
    """Two clients send this robot to a place: the goal server (ros/go.sh) and goto_ros.py
    (ros/goto.sh). Only the first ever asked the recorder for a tape, so every drive started at
    the second went unrecorded by it while its own session log kept running — the numbered tapes
    stop at 0248 on 2026-09-13 18:00 and the goto ones continue to midnight. Both speak
    pepin.runlink now, and goto names the tape it got in its own log."""
    goto = sf.tree("ros/tools/goto_ros.py")
    assert {"RUN_COMMAND_TOPIC", "RUN_STATUS_TOPIC"} <= sf.names(goto)
    assert {"start_command", "stop_command"} <= sf.calls(goto)
    # ...and the drive must actually use it: the protocol being imported proved nothing about
    # the goal path calling it, which is exactly how this went unnoticed for four hours.
    assert {"Tape", "tape.open"} <= sf.calls(goto)
    # ``tape.close`` may be CALLED or handed to the shutdown's guard (2026-09-18: every step of an
    # interrupt runs through `guarded` so one failure cannot swallow the next), so the contract is
    # that the closing happens at all, not the spelling of the call.
    assert "tape.close" in sf.calls(goto) or "tape.close" in {
        ast.unparse(n) for n in ast.walk(goto) if isinstance(n, ast.Attribute)
    }, "the drive must close the tape it opened"
    assert any("taped" in text for text in sf.strings(goto)), "the log must name the tape"
    assert "--no-tape" in sf.strings(goto), "the drive without a numbered tape stays reachable"
    script = (REPO / "ros/goto.sh").read_text()
    assert "--no-tape" in script and "PEPIN_GOTO_TAPE" in script


def test_the_static_layers_read_the_map_the_tracker_is_on_and_no_pgm_is_served() -> None:
    """One map on the board (2026-09-18): the relocalizer republishes the grid it tracks on and both
    costmaps' static layers read THAT, so the planner's static map cannot disagree with the
    tracker's. The topic is written as a literal in both places and held equal here; the pgm leaves
    the loop, reachable again with `map_server:=true`."""
    params = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
    layers = [
        params[costmap][costmap]["ros__parameters"]["static_layer"]
        for costmap in ("local_costmap", "global_costmap")
    ]
    assert [layer["map_topic"] for layer in layers] == ["/map_tracked", "/map_tracked"]
    assert all(layer["map_subscribe_transient_local"] for layer in layers), "latched, or it waits"
    node = sf.tree("ros/pepin_bringup/pepin_bringup/relocalizer.py")
    assert "/map_tracked" in {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }, "the relocalizer must carry the same literal"
    launch = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    assert 'DeclareLaunchArgument("map_server", default_value="false"' in launch
    assert 'if map_server and runs_here(side, "map_server"):' in launch


def test_ctrl_c_cancels_the_goal_before_it_can_do_anything_else() -> None:
    """The contract the two rocking-chair legs broke (2026-09-17): rclpy's own SIGINT handler had
    torn the context down, the interrupt handler's first statement created a publisher for a note,
    that raised, and the cancel on the next line never ran — the goal stayed alive on the board.
    So: rclpy must not take SIGINT, and the interrupt path's first act is the cancel."""
    goto = sf.tree("ros/tools/goto_ros.py")
    init = sf.calls_to(goto, "rclpy.init")
    assert init and all(
        any(kw.arg == "signal_handler_options" for kw in call.keywords) for call in init
    ), "rclpy must be told not to shut the context down under the handler"
    handler = next(
        node
        for node in ast.walk(goto)
        if isinstance(node, ast.FunctionDef) and node.name == "interrupted"
    )
    first = next(
        sf.dotted(n.func)
        for n in ast.walk(handler)
        if isinstance(n, ast.Call) and sf.dotted(n.func) in {"guarded", "note", "nav.cancelTask"}
    )
    assert first == "guarded", "the cancel goes through the guard, first"
    guards = [
        n for n in ast.walk(handler) if isinstance(n, ast.Call) and sf.dotted(n.func) == "guarded"
    ]
    assert ast.unparse(guards[0].args[1]) == "nav.cancelTask", ast.unparse(guards[0])


def test_goto_judges_online_slam_on_the_map_frame_not_on_a_fit() -> None:
    """goto's localisation gate asked two services the board does not run in online SLAM
    (/where_am_i, /relocalize — pepin.deployment.runs_here keeps the relocalizer off where there
    is no saved map to match against), so it refused both goals of the 2026-09-14 20:00 session
    after ten seconds of waiting for absent services, whatever the map was doing. In that mode
    the evidence is the map frame itself: pepin_bringup.slam_frame broadcasts map -> odom at
    10 Hz on the board, and a frame younger than a second is what a goal is judged on."""
    goto = sf.tree("ros/tools/goto_ros.py")
    defined = {f.name for f in ast.walk(goto) if isinstance(f, ast.FunctionDef)}
    assert {"tracker_here", "map_frame_age_s"} <= defined, (
        "the two halves of the gate: is there a tracker, and is the map frame fresh"
    )
    gate = next(
        f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "ensure_localized"
    )
    body = ast.unparse(gate)
    assert "tracker_here(nav)" in body, "ask whether a tracker exists BEFORE asking it anything"
    assert "map_frame_age_s(nav)" in body and "MAP_FRAME_FRESH_S" in body
    assert "map -> base_link" in " ".join(sf.strings(goto)), "the log must name what was judged"


def test_one_recorder_writes_a_drive_not_two() -> None:
    """ros/goto.sh started ros/tools/session_logger.py for every drive while the board's run
    recorder was already subscribed to the same topics: two rclpy processes turning the same
    10 Hz LaserScan into Python objects on four A53 cores (15 % of a core and ~140 MB for the
    second one). The numbered tape carries the fusion's two String topics now — the only records
    the session logger had to itself — so goto starts it only when there is no numbered tape."""
    recorder = sf.tree(f"{NODES}/run_recorder.py")
    assert {"/localization/measurement", "/localization/sources"} <= set(sf.strings(recorder)) or {
        "measurement",
        "sources",
    } <= set(sf.strings(recorder)), "the tape carries what the session logger alone carried"
    assert {"meas", "srcs"} <= set(sf.strings(recorder)), "under the names camera_error.py reads"
    flags = load_table(REPO / NODES / "run_recorder.py")
    assert flags.flag("fusion_records").live and flags["fusion_records"] is True
    script = (REPO / "ros/goto.sh").read_text()
    starter = next(ln for ln in script.splitlines() if "session_logger.py $REC" in ln)
    assert starter.startswith("    "), "the session logger is started inside a condition now"
    assert "PEPIN_SESSION_LOGGER" in script, "and by name when a drive wants it anyway"
    assert "NO TAPE" in script, "a drive with no tape at all must say so loudly"


def test_goto_waits_on_its_log_watcher_instead_of_blocking_on_it() -> None:
    """bash defers a trap until the running foreground command returns. goto.sh's last command
    was a foreground `ssh | sed` that ends only when the board writes GOTO_EXIT, and a TERM to
    the script never reached that ssh: three goto.sh survived their SIGTERM on 2026-09-13, 1-2 h
    old, each holding an ssh. The watcher is a background job now, `wait`ed on — which a trapped
    signal does interrupt — killed by a `finish` that runs exactly once."""
    script = (REPO / "ros/goto.sh").read_text()
    assert "trap finish EXIT\n" in script and "trap 'finish; exit 143' HUP TERM" in script
    assert 'wait "$TAILPID"' in script and "TAILPID=$!" in script
    assert 'if [ "$FINISHED" = 1 ]; then return 0; fi' in script, "finish must not run twice"
    watcher = next(ln for ln in script.splitlines() if "GOTO_EXIT=/q" in ln)
    assert watcher.rstrip().endswith("&"), "the watcher must never be the foreground command"


def test_the_numbered_tape_says_which_clock_named_it() -> None:
    """The recorder runs in a container on UTC while the board's shell, the laptop and every
    ros/goto.sh file are on local time: a bare 220039 was read as a drive four hours later than
    it was (2026-09-13). The stamp is UTC and carries the letter that says so."""
    recorder = sf.tree(f"{NODES}/run_recorder.py")
    assert "time.gmtime" in sf.calls(recorder), "the stamp is UTC on purpose, not by accident"
    assert any(text == "Z" for text in sf.strings(recorder)), "and the name says which clock"
    assert (REPO / "ros/maps/README.md").exists(), "the two clocks are written down beside rec/"


def test_the_camera_slam_lives_beside_the_tracker_never_over_it() -> None:
    """World R's one frame, owned by nobody here: RTAB-Map's map frame IS ``map`` and it publishes
    no transform into it, because the board's tracker owns ``map -> odom`` and this graph's
    correction travels as a measurement and as a grid. The camera is nominal until calibrated and
    says so."""
    vslam = sf.tree(VSLAM_LAUNCH)
    params = sf.dict_items(vslam)
    assert params["publish_tf"] == {"False", "True"}, (
        "RTAB-Map is in the tf tree in exactly one situation: PEPIN_LOCALIZER=rtabmap"
    )
    assert _rtabmap("RTABMAP")["map_frame_id"] == "map", "one frame, since 2026-09-19"
    rtabmap = sf.keywords(_node_named(vslam, "rtabmap"))
    assert ast.unparse(rtabmap["namespace"]) == "'rtabmap'", (
        "its relative 'map' output must not land on /map"
    )
    assert _rtabmap("RTABMAP")["Reg/Strategy"] == "1"
    assert "pepin_bringup.camera_stream" in sf.strings(vslam)
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "vslam.launch.py" in laptop and "config:/ws/config" in laptop
    # a mono camera cannot seed a loop-closure transform: ICP from identity does
    assert _rtabmap("RTABMAP")["RGBD/LoopClosureIdentityGuess"] == "true"


def test_the_bridge_routes_only_what_the_split_needs_and_only_one_way() -> None:
    """Routing every node's parameter services wedged the link; routing a topic both ways looped
    it (a bridge finds its own writer and calls it a publisher). Each side's config lists its own
    publishers and the other side's subscribers, generated from pepin.deployment."""
    from pepin.deployment import bridge_allow, bridge_config

    for side in ("board", "laptop"):
        written = json.loads((REPO / f"ros/zenoh-bridge-{side}.json").read_text())
        assert written == bridge_config(side), f"regenerate ros/zenoh-bridge-{side}.json"
    board, laptop = bridge_allow("board"), bridge_allow("laptop")
    pub_b, sub_l = re.compile(board["publishers"][0]), re.compile(laptop["subscribers"][0])
    for name in (
        "/scan",
        "/tf",
        "/tf_static",
        "/map_tracked",
        "/odometry/filtered",
        "/tof/front",
        "/tracker_pose",
        "/localization/sources",
        "/pepin/run_status",
    ):
        assert pub_b.search(name) and sub_l.search(name), name
        assert not re.compile(laptop["publishers"][0]).search(name), f"{name} would loop"
    for name in (
        "/plan",
        "/pepin/run",
        "/laptop/heartbeat",
        "/planner_selector",
        "/map",
        "/rtabmap/mapGraph",
        "/depth_scan",
        "/depth_marks",
        "/contact_scan",
    ):
        assert re.compile(laptop["publishers"][0]).search(name) and re.compile(
            board["subscribers"][0]
        ).search(name)
        assert not pub_b.search(name), f"{name} would loop"
    assert re.compile(board["action_servers"][0]).search("/navigate_to_pose")
    assert re.compile(laptop["action_servers"][0]).search("/compute_path_to_pose")
    assert re.compile(board["service_servers"][0]).search("/bt_navigator/change_state")
    assert re.compile(laptop["service_servers"][0]).search(
        "/global_costmap/clear_entirely_global_costmap"
    )
    for noise in ("/launch_ros_1/get_parameters", "/rosout", "/camera/image", "/ldlidar_node/scan"):
        for block in (*board.values(), *laptop.values()):
            assert not re.compile(block[0]).search(noise), noise
    # Vision mode (the board drives, the laptop maps): the plan comes FROM the board and the
    # laptop publishes none of it; each side's file is generated too, the board's unit reads
    # the one ros/thin.sh names with the mode, laptop.sh picks its own by the board's side.
    for side in ("board", "laptop"):
        written = json.loads((REPO / f"ros/zenoh-bridge-{side}-vision.json").read_text())
        assert written == bridge_config(side, "vision"), f"regenerate zenoh-bridge-{side}-vision"
    vision_b, vision_l = bridge_allow("board", "vision"), bridge_allow("laptop", "vision")
    for name in ("/plan", "/global_costmap/costmap", "/local_costmap/published_footprint"):
        assert re.compile(vision_b["publishers"][0]).search(name), name
        assert re.compile(vision_l["subscribers"][0]).search(name), name
        assert not re.compile(vision_l["publishers"][0]).search(name), f"{name} would loop"
    # The laptop's whole-map watchdog: its candidates cross to the board, and everything it
    # needs to compute one (the scan, the map the board ACCEPTED, what the tracker believes)
    # crosses to it. The pose it publishes beside them for Foxglove stays on the laptop, where
    # Foxglove is. /map goes the other way: it is the laptop's own grid (World R).
    for name in ("/scan", "/map_tracked", "/tracker_pose", "/localization_fit", "/tf_static"):
        assert re.compile(vision_b["publishers"][0]).search(name), name
        assert re.compile(vision_l["subscribers"][0]).search(name), name
    assert re.compile(vision_l["publishers"][0]).search("/map")
    assert re.compile(vision_b["subscribers"][0]).search("/map")
    assert not re.compile(vision_b["publishers"][0]).search("/map"), "one map, one publisher"
    assert re.compile(vision_l["publishers"][0]).search("/localization/candidate")
    assert re.compile(vision_b["subscribers"][0]).search("/localization/candidate")
    assert not re.compile(vision_b["publishers"][0]).search("/localization/candidate")
    for block in (*vision_b.values(), *vision_l.values()):
        assert not re.compile(block[0]).search("/localization/candidate_pose")
    unit = (REPO / "board/pepin-bridge.service").read_text()
    assert "Environment=PEPIN_BRIDGE_CONFIG=zenoh-bridge-board.json" in unit
    assert "/root/pepin-ros/$PEPIN_BRIDGE_CONFIG:/config.json:ro" in unit
    thin = (REPO / "ros/thin.sh").read_text()
    vision = thin[thin.index("    vision)") : thin.index("    off)")]
    assert "PEPIN_BRIDGE_CONFIG=zenoh-bridge-board-vision.json" in vision
    assert "    slam)" not in thin, "the SLAM mode went with World R (2026-09-19)"
    for other in ("    on)", "    off)"):
        block = thin[thin.index(other) :].split("\n    ")[1]
        assert "/^PEPIN_BRIDGE_CONFIG=/d" in thin and "board-vision" not in block, other
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "CONFIG=zenoh-bridge-laptop.json" in laptop
    assert "CONFIG=zenoh-bridge-laptop-vision.json" in laptop
    assert laptop.index('[ "$SIDE" = board ]') < laptop.index("CONFIG=zenoh-bridge-laptop.json")


def test_the_laptop_half_restarts_with_the_board_s_bridge() -> None:
    """A subscription does not follow a bridge through its restart: both laptop launches carry
    the bridge watch (exit -> launch shutdown -> container restart), the containers restart on
    their own, and the board's bridge is settled BEFORE the containers start, never after."""
    for launch in (NAV_LAUNCH, VSLAM_LAUNCH):
        src = sf.tree(launch)
        assert "pepin_bringup.bridge_watch" in sf.strings(src), launch
        assert "Shutdown" in sf.calls(src), launch
    laptop = (REPO / "ros/laptop.sh").read_text()
    # The two NODE containers restart on their own. Named, not counted: under
    # PEPIN_RMW=zenoh this side also runs a router container with the same policy, and a
    # bare count would have read that as one of the halves.
    for container in ("pepin-vslam", "pepin-laptop"):
        line = next(ln for ln in laptop.splitlines() if f"docker run -d --name {container} " in ln)
        assert "--restart unless-stopped" in line, container
    lines = laptop.splitlines()
    settles = [i for i, line in enumerate(lines) if line.split("#")[0].strip() == "settle_bridge"]
    starts = [i for i, line in enumerate(lines) if "docker run -d --name pepin-laptop" in line]
    assert len(settles) == 1 and len(starts) == 1, "one settle, one navigation container"
    assert settles[0] < starts[0], "the bridge settles BEFORE the container, never after"
    # A settle that fails is fatal, and a board that does not answer is fatal and loud: the
    # script sources lib.sh (a connect timeout, one ssh master) and never silently ends on a
    # failed command substitution (2026-09-10 20:02).
    assert '. "$HERE/lib.sh"' in laptop and "SITE=" not in laptop
    settle = laptop[laptop.index("settle_bridge() {") :].split("\n}")[0]
    assert "return 1" in settle
    assert "cannot read the board's side over ssh" in laptop
    # The board is asked only on the start path: stop, logs, vslam and kick never ssh.
    subcommands = laptop[laptop.index('case "${1:-start}"') : laptop.index("esac")]
    assert "ssh " not in subcommands and "MAP=" not in subcommands
    assert laptop.index("MAP=") > laptop.index("esac")
    # The bridge watch: a silent admin restarts the half (a minute: shorter and a WiFi hiccup
    # restarts it, longer and a wedged bridge is tolerated), the settle threshold is measured
    # against the previous bridge's count, never a fixed number.
    watch = sf.tree(f"{NODES}/bridge_watch.py")
    assert 30.0 <= ast.literal_eval(sf.assignments(watch)["SILENCE_S"]) <= 120.0
    assert "BridgeIdentity(silence_s=SILENCE_S)" in sf.unparsed(watch, ast.Call)
    settled = sf.calls_to(watch, "routes_settled")
    assert settled and not any(isinstance(c.args[1], ast.Constant) for c in settled)
    assert "self._identity.observe(zid, now)" in sf.unparsed(watch, ast.Call)
    # ...and the watch verifies FLOW, not route counts: it counts the messages of every topic
    # both bridges say should arrive here, and repairs the bridge container before this half
    # (a route can exist, carry nothing, and look perfect on both admins — 2026-09-12/13).
    calls = sf.unparsed(watch, ast.Call)
    assert "FlowWatch()" in calls and "self._flow.starved(now)" in calls
    assert "self._repair.restart()" in calls and "self._repair.available()" in calls
    assert "topic_flows(routes, self._local_zid, self._allowed, watcher=WATCH_NODE)" in calls
    mend = sf.unparsed(watch, ast.FunctionDef)
    assert "self.restart_half" in next(m for m in mend if m.startswith("def mend")), (
        "the whole half stays as the escalation"
    )
    # The board's unit fails itself when the REST admin stays silent, so Restart= fires.
    unit = (REPO / "board/pepin-bridge.service").read_text()
    post = next(line for line in unit.splitlines() if line.startswith("ExecStartPost="))
    assert "127.0.0.1:8000/@/local/router" in post and "exit 1" in post and "seq 1 30" in post
    assert "Restart=on-failure" in unit


def test_the_laptop_halves_start_their_nodes_only_after_their_ghosts_are_gone() -> None:
    """The bridge keeps routes by node name: a half restarted within its predecessor's DDS lease
    loses its routes when the ghost expires. Both laptop launches start their ROS nodes on the
    ghost wait's exit, the names they wait for are the names they create, and laptop.sh lets a
    container leave properly before replacing it."""
    from pepin.deployment import laptop_launch_nodes

    for launch, half in ((NAV_LAUNCH, "nav"), (VSLAM_LAUNCH, "slam")):
        src = sf.tree(launch)
        assert "pepin_bringup.ghost_wait" in sf.strings(src), launch
        assert _started_after_ghost_wait(src), launch
        assert f"laptop_launch_nodes('{half}')" in sf.unparsed(src, ast.Call), launch
    vslam = sf.tree(VSLAM_LAUNCH)
    assert ast.unparse(sf.keywords(_node_named(vslam, "rtabmap"))["namespace"]) == "'rtabmap'"
    assert _node_named(vslam, "foxglove_bridge") is not None
    for module in ("camera_stream", "depth_stream", "contact_scan", "goal_server"):
        node = sf.tree(f"{NODES}/{module}.py")
        assert f"super().__init__('{module}')" in sf.unparsed(node, ast.Call), module
    assert laptop_launch_nodes("slam") == (
        "/camera_stream",
        "/depth_stream",
        "/contact_scan",
        "/depth_fusion",
        "/laptop_localizer",
        "/sensor_pack",
        "/rtabmap/rtabmap",
        "/rtabmap_frame",
        "/places",
        "/foxglove_bridge",
        "/rgbd_odometry",
        "/visual_odometry",
    )
    nav = sf.tree(NAV_LAUNCH)
    composed = {ast.unparse(sf.keywords(c)["name"]) for c in sf.calls_to(nav, "ComposableNode")}
    assert "f'lifecycle_manager_navigation_{side}'" in composed
    container = sf.keywords(sf.calls_to(nav, "ComposableNodeContainer")[0])
    assert ast.unparse(container["name"]).startswith("f'nav2_container_{side}' if side != 'all'")
    laptop = (REPO / "ros/laptop.sh").read_text()
    # The stop itself is ros/lib.sh's now (one way to stop a container, one window); what this
    # contract still owns is that nothing here removes a container without stopping it first.
    assert "pepin_remove_container pepin-vslam" in laptop
    acted = "\n".join(ln for ln in laptop.splitlines() if not ln.lstrip().startswith("#"))
    assert "docker rm -f" not in acted and "docker stop" not in acted
    # `docker stop` sends the container's stop signal: SIGINT is the one the launch answers by
    # shutting its nodes down (SIGTERM it answers by cancelling itself and the nodes are
    # SIGKILLed without a dispose — the ghost on every container stop). A run wrapped over
    # several lines is read as one command.
    runs = [c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-" in c]
    assert len(runs) == 3, "the bridge, the navigation half, the SLAM half"
    for command in runs:
        assert ("--stop-signal SIGINT" in command) == ("pepin-zenoh" not in command), command


def test_every_respawned_node_waits_for_its_own_ghost_first() -> None:
    """A crash disposes nothing: the name outlives the process by the DDS lease, the launch
    respawns it in two seconds, and the bridge drops the name's routes when the ghost expires
    (2026-09-11 03:07, the Nav2 container). Every respawned command in both launches starts
    through a ghost wait of its own names (prefix; ghost_wait execs the command after), the
    board's launch asks the bridge on its own host, and an unreachable admin is not waited for."""
    from pepin.deployment import nav_container_nodes

    for name in ("nav.launch.py", "vslam.launch.py", "robot.launch.py"):
        src = sf.tree(f"ros/pepin_bringup/launch/{name}")
        assert any(isinstance(n, ast.FunctionDef) and n.name == "_after_ghost" for n in src.body)
        # the laptop halves also wait for their whole set once; the board's launch only prefixes
        assert any("pepin_bringup.ghost_wait" in s for s in sf.strings(src)), name
        for node, keywords in _launch_processes(name).items():
            if keywords.get("respawn") is True:
                assert "_after_ghost(" in str(keywords.get("prefix")), (name, node)
    nav = sf.tree(NAV_LAUNCH)
    container = sf.keywords(sf.calls_to(nav, "ComposableNodeContainer")[0])
    assert (
        ast.unparse(container["prefix"])
        == "f'nice -n 5 {_after_ghost(admin, *nav_container_nodes(side))}'"
    )
    by_side = [
        n
        for n in ast.walk(nav)
        if isinstance(n, ast.BoolOp)
        and isinstance(n.op, ast.Or)
        and "bridge_admin_for(side)" in {ast.unparse(v) for v in n.values}
    ]
    assert by_side, "an empty bridge_admin argument is resolved by side"
    admin = next(
        c
        for c in sf.calls_to(nav, "DeclareLaunchArgument")
        if ast.unparse(c.args[0]) == "'bridge_admin'"
    )
    assert ast.unparse(sf.keywords(admin)["default_value"]) == "''"
    assert "/nav2_container_board" in nav_container_nodes("board")
    for node in ("relocalizer", "run_recorder", "goal_server"):
        assert f"_after_ghost(admin, '/{node}')" in sf.unparsed(nav, ast.Call), node
    vslam = sf.tree(VSLAM_LAUNCH)
    for node in (
        "camera_stream",
        "depth_stream",
        "contact_scan",
        "depth_fusion",
        "rtabmap_frame",
        "foxglove_bridge",
    ):
        assert f"_after_ghost('/{node}')" in sf.unparsed(vslam, ast.Call), node
    robot = sf.tree(ROBOT_LAUNCH)
    assert "_after_ghost('/neck_state')" in sf.unparsed(robot, ast.Call)
    assert "bridge_admin_for('board')" in sf.unparsed(robot, ast.Call), "its own host's bridge"
    wait = sf.tree(f"{NODES}/ghost_wait.py")
    assert {"os.execvp(command[0], command)", "rest.index('--')"} <= sf.unparsed(wait, ast.Call)
    assert any("not waiting" in s for s in sf.strings(wait))
    # The board's container shares the host's network: 127.0.0.1:8000 is its bridge's admin.
    run = (REPO / "ros/run.sh").read_text()
    assert "--network host" in run


def test_the_camera_is_a_depth_sensor_scaled_by_the_lidar() -> None:
    """The laptop turns the camera's frames into depth images (a network on CPU, the lidar sets
    the scale), RTAB-Map builds its grid from the lidar AND that depth per node, the cloud is drawn
    by the operator's Foxglove connected to the laptop, and the image carries the weights."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.depth_stream" in sf.strings(vslam)
    table = _rtabmap("RTABMAP")
    assert table["subscribe_depth"] is False and table["subscribe_sensor_data"] is True
    assert _rtabmap("TRIPLE_SUBSCRIPTIONS")["subscribe_depth"] is True, "the way back, and only it"
    assert "('depth/image', '/camera/depth')" in sf.unparsed(vslam, ast.Tuple)
    assert table["Grid/Sensor"] == "0", (
        "the grid starts on the scan and follows the snapshots at run time (pepin.graphmode):"
        " with both sensors the mono depth flattened the lidar tracker's match (2026-09-19)"
    )
    assert table["Grid/3D"] == "false", "the grid the board is handed is 2D"
    assert "Grid/MaxObstacleHeight" in table
    assert {"camera", "depth", "pack", "rtabmap", "frame", "foxglove"} <= _started_after_ghost_wait(
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
    assert layout["configById"]["3D!nav"]["topics"]["/rtabmap/cloud_map"]["colorMode"] == "rgb"
    # RTAB-Map's own grid fights the static map for the floor: off by default in the nav view
    assert layout["configById"]["3D!nav"]["topics"]["/rtabmap/map"]["visible"] is False
    # the 3D view: the cloud, the path, the scan and the robot; no grid lies over the voxels
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    shown = {t for t, c in room["topics"].items() if c.get("visible")}
    assert "/fusion/surface" in shown  # the fused surface is what the operator sees
    assert "/rtabmap/cloud_map" in room["topics"]  # RTAB-Map's cloud stays a click away
    # THE PLANNER'S OWN MAP IS SHOWN, and only one grid per plane. The owner watched the fused
    # surface over the LOCAL costmap and could not see why a path went through a chair: the path
    # is planned on the global costmap over /map, and both were hidden (2026-09-18). So those two
    # are on, and every other occupancy grid — the local costmap's window on the same cells,
    # RTAB-Map's own grid, and the volume's two other cross-sections — stays a click away,
    # because three grids in one horizontal plane fight each other for depth.
    # /map_tracked, not /map: the board's relocalizer republishes the map its TRACKER is on and
    # drives on, latched, while /map is the laptop's live grid a second or two ahead of it — so
    # the tracked one is what the planner and the operator must be looking at, and the live one
    # is a click away for the moments the two part (a loop closure).
    assert {"/map_tracked", "/global_costmap/costmap", "/plan"} <= shown
    assert not shown & {"/map", "/local_costmap/costmap"}
    assert "/map" in room["topics"], "the laptop's live grid is in the layout, one click away"
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
    optimised map frame IS `map`, so the only edge rtabmap_frame ever publishes is SLAM's
    map -> odom, as a message home."""
    from pepin.deployment import LAPTOP_PUBLISHES

    assert "depth_scan" in LAPTOP_PUBLISHES
    layer = _p("local_costmap")["camera_layer"]
    assert "depth_scan" in layer["observation_sources"].split()
    source = layer["depth_scan"]
    assert source["topic"] == "/depth_scan" and source["data_type"] == "LaserScan"
    assert source["inf_is_valid"] is True
    assert "depth_marks" in LAPTOP_PUBLISHES, "and what that layer MARKS with, see below"
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert "/depth_scan" in sf.strings(node) and "depth_to_scan" in sf.calls(node)
    camera = sf.tree(f"{NODES}/camera_stream.py")
    assert "capture_time" in sf.calls(camera) and "cv2.VideoCapture" not in sf.calls(camera)
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.rtabmap_frame" in sf.strings(vslam)
    nav = sf.tree(NAV_LAUNCH)
    assert not any("odom_to_rtabmap" in s for s in sf.strings(nav) | sf.names(nav))
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert "/rtabmap/mapGraph" in sf.strings(frame), "read in SLAM, where it IS map -> odom"
    tuples = sf.unparsed(frame, ast.Tuple)
    assert "('map', 'odom')" in tuples and "('map', 'rtabmap')" not in tuples, "one frame"
    # RTAB-Map's odometry is the EKF's own, in every situation
    assert _rtabmap("RTABMAP")["odom_frame_id"] == "odom"
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    assert room["topics"]["/depth_scan"]["visible"]
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
            assert params[f"tof_{sensor}_layer"]["no_readings_timeout"] == 0.0
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
            assert params[f"tof_{sensor}_layer"]["enabled"] is True
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
        assert layer["enabled"] is False, f"{costmap}: not validated on the robot yet"
        assert layer["observation_sources"].split() == ["contact_scan"]
        source = layer["contact_scan"]
        assert source["topic"] == "/contact_scan" and source["data_type"] == "LaserScan"
        assert source["sensor_frame"] == "base_link"
        assert source["marking"] is True and source["clearing"] is False
        assert source["inf_is_valid"] is False, "inf with clearing off would mark a ring at 2 m"
        assert source["obstacle_max_range"] == CONTACT_MAX_RANGE
    # The node's own cap is the same number, and its scan's range_max with it: a mark the layer
    # would have to discard is a mark nobody sees.
    assert load_table(REPO / NODES / "contact_scan.py")["max_range"] == CONTACT_MAX_RANGE
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
    assert load_table(REPO / NODES / "depth_stream.py")["depth_reach_m"] == scan_max_range


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
    assert "marks_ranges" in sf.calls(node) and "/depth_scan" in sf.strings(node)
    flags = load_table(REPO / NODES / "depth_fusion.py")
    assert flags["marks_source"] == "volume", "the volume marks by default"
    assert flags.flag("marks_source").choices == ("volume", "frame"), "the old way, live"
    for costmap in ("local_costmap", "global_costmap"):
        layer = _p(costmap)["camera_layer"]
        assert layer["observation_sources"].split() == ["depth_scan", "depth_marks"], costmap
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


def test_the_floor_s_edge_is_a_node_of_the_kit_and_crosses_the_bridge() -> None:
    """The contact scan runs where the depth network runs — on the laptop, off /camera/depth —
    and reaches the board's costmap the way /depth_scan does: through the bridge's allow-list,
    one way, in both bridge modes. The node is the kit's: a newest-wins worker, live switches
    printed in its report line, the floor's geometry rebuilt only when the lean or the optics
    move, and a launch entry that respawns it behind a wait for its own ghost."""
    from pepin.deployment import LAPTOP_PUBLISHES, VISION_LAPTOP_PUBLISHES, bridge_allow

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
    flags = load_table(REPO / NODES / "contact_scan.py")
    assert flags.names == ("contact_scan", "shadow", "imu_lean", "max_range")
    assert flags["imu_lean"] is True, "on since the gyro's sign was verified by hand (2026-09-13)"
    assert flags["contact_scan"] is True and flags["shadow"] is True
    assert all(flag.live for flag in flags), "every one of them takes the next frame"
    # the plane is a cache with two keys: the lean and the optics
    assert "self._plane_up" in sf.unparsed(node, ast.Attribute)
    assert "self._plane_intr != intr" in sf.unparsed(node, ast.Compare)
    assert "contact_scan" in LAPTOP_PUBLISHES and "contact_scan" in VISION_LAPTOP_PUBLISHES
    for mode in ("split", "vision"):
        laptop, board = bridge_allow("laptop", mode), bridge_allow("board", mode)
        assert re.compile(laptop["publishers"][0]).search("/contact_scan"), mode
        assert re.compile(board["subscribers"][0]).search("/contact_scan"), mode
        assert not re.compile(board["publishers"][0]).search("/contact_scan"), "it would loop"
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.contact_scan" in sf.strings(vslam)
    assert "contact" in _started_after_ghost_wait(vslam)


def test_the_drive_ends_on_position_and_the_goal_server_turns_to_the_heading() -> None:
    """RPP with reversing cannot rotate in place: the tree's FollowPath judges xy only, and the
    goal server pivots the residual heading with the behaviour server's Spin before "done"."""
    controller = _p("controller_server")
    assert "xy_only_goal_checker" in controller["goal_checker_plugins"]
    xy_only = controller["xy_only_goal_checker"]
    assert xy_only["xy_goal_tolerance"] == controller["general_goal_checker"]["xy_goal_tolerance"]
    assert xy_only["yaw_goal_tolerance"] >= 3.14
    tree = ET.parse(REPO / "ros/params/pepin_nav_to_pose.xml")
    assert any(n.get("goal_checker_id") == "xy_only_goal_checker" for n in tree.iter("FollowPath"))
    server = sf.tree(f"{NODES}/goal_server.py")
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
    """The camera rides a two-servo neck. The board's node (pepin_bringup.neck_state) asks the
    base server for the encoders and publishes /neck/state and, behind its live ``neck_tf``,
    base_link -> camera_link from them; the laptop's camera node must then stop broadcasting its
    static copy of that edge — two publishers of one edge fight, and a static transform cannot
    be withdrawn, so that one is a launch switch (``static_camera_tf``), read once at start.

    One operator gesture per side, each carried by one launch argument: ``ros/feature.sh neck
    on`` sets PEPIN_NECK, the unit passes ``neck:=`` down through bringup to robot.launch.py;
    ``ros/laptop.sh vslam --neck`` passes ``static_camera_tf:=false`` into vslam.launch.py, which
    hands it to the camera node alone. The board is never asked what it is doing: this launch
    talks to no one, and a guess would be the two-publisher case.
    """
    node = sf.tree(f"{NODES}/neck_state.py")
    declared = {
        ast.unparse(c.args[0]): c.args[1] for c in sf.calls_to(node, "self.declare_parameter")
    }
    assert {"'host'", "'port'", "'poll_hz'", "'tf_hz'", "'config'"} <= declared.keys()
    neck_flags = load_table(REPO / NODES / "neck_state.py")
    assert "neck_tf" in neck_flags and neck_flags.flag("neck_tf").live
    assert sf.assignments(node)["_NECK_REQUEST"] == 'b\'{"cmd":"neck"}\\n\''
    assert {"parse_neck", "joint_angles", "camera_pose", "NeckConfig"} <= sf.imported(node)
    assert "JsonLineLink" in sf.imported(node), "the reconnecting link, not a socket of its own"
    assert "super().__init__('neck_state')" in sf.unparsed(node, ast.Call)
    # The bus is polled at 2 Hz and the edge published at 10: a read costs 13.5 ms of a core and
    # the head is still while the cart drives, so the last edge is republished with a fresh stamp
    # (the live ``tf_republish``). A 2 Hz TF stream would fail lookups at recent stamps.
    assert ast.unparse(declared["'poll_hz'"]) == "2.0"
    assert ast.unparse(declared["'tf_hz'"]) == "_TF_HZ"
    assert sf.assignments(node)["_TF_HZ"] == "10.0"
    assert "tf_republish" in neck_flags and neck_flags.flag("tf_republish").live
    # The switch defaults off while the model is unchecked against the hardware: the reference
    # ticks unread (every pose is then the static mount) or the servo signs unverified.
    from pepin.neck import NeckConfig

    reference = NeckConfig.from_json(REPO / "config/neck.json").reference
    if not (reference.known and reference.signs_verified):
        assert neck_flags["neck_tf"] is False, "unverified: no transform by default"
    camera = sf.tree(f"{NODES}/camera_stream.py")
    # The switch is one of the camera node's flags (node_kit.Switches over its FLAGS table,
    # CLAUDE.md rule 19) and is printed in its report line — but it is declared not live: the
    # transform went out at start, and a static transform cannot be withdrawn.
    camera_flags = load_table(REPO / NODES / "camera_stream.py")
    assert "static_camera_tf" in camera_flags and not camera_flags.flag("static_camera_tf").live
    assert camera_flags["static_camera_tf"] is True, "this side owns the edge unless told not to"
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
    robot = sf.tree(ROBOT_LAUNCH)
    assert "pepin_bringup.neck_state" in sf.strings(robot)
    neck_arg = next(
        c for c in sf.calls_to(robot, "DeclareLaunchArgument") if ast.unparse(c.args[0]) == "'neck'"
    )
    assert ast.unparse(sf.keywords(neck_arg)["default_value"]) == "'false'"
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    assert bringup and "LaunchConfiguration('neck')" in sf.unparsed(bringup, ast.Call)
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "Environment=PEPIN_NECK=false" in unit and "neck:=${PEPIN_NECK}" in unit
    feature = (REPO / "ros/feature.sh").read_text()
    assert "neck) VAR=PEPIN_NECK ;;" in feature
    assert (
        "PEPIN_(CPP_BRIDGE|IMU|EKF|TOF|NECK|SIDE|BRIDGE|BRIDGE_CONFIG)"
        in (REPO / "ros/mode.sh").read_text()
    ), "a mode change must not wipe the bridge the board was told to run"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "--neck) STATIC_CAMERA_TF=false ;;" in laptop, "a flag anywhere after the subcommand"
    assert any(
        '"static_camera_tf:=$STATIC_CAMERA_TF"' in command
        for command in sf.shell_commands(laptop)
        if "docker run -d --name pepin-vslam" in command
    )


def test_the_frames_are_fused_into_one_surface_beside_rtabmap_s_cloud() -> None:
    """The SLAM launch runs the fusion node after the ghost wait; the node loads the grid from
    config/fusion.json and offers its switches as parameters; the 3D layout shows the fused
    surface and hides RTAB-Map's concatenated cloud by default (both stay available)."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.depth_fusion" in sf.strings(vslam)
    assert "fusion" in _started_after_ghost_wait(vslam)
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert sf.assignments(node)["CONFIG"] == "'/ws/config/fusion.json'"
    # The four are flags of the node's table (node_kit.Switches over pepin.flags), so ros2
    # param set reaches them and their state is printed in the report line (CLAUDE.md rule 19).
    fusion_flags = load_table(REPO / NODES / "depth_fusion.py")
    assert {"enabled", "align", "min_weight", "surface_hz"} <= set(fusion_flags.names)
    assert fusion_flags.flag("surface_hz").range is not None, "a rate is bounded"
    assert not fusion_flags.flag("resume_volume").live, "a start-up choice, not a live switch"
    assert "Switches" in sf.imported(node) and "self._switches.state" in sf.calls(node)
    assert {"/fusion/reset", "/fusion/surface"} <= sf.strings(node)
    for name in ("pepin_3d.json", "pepin_nav.json"):
        layout = json.loads((REPO / "ros/foxglove" / name).read_text())
        panel = next(v for k, v in layout["configById"].items() if k.startswith("3D"))
        assert panel["topics"]["/fusion/surface"]["visible"] is True
        assert panel["topics"]["/fusion/surface"]["colorMode"] == "rgb"
        assert panel["topics"]["/rtabmap/cloud_map"]["visible"] is False


def test_the_volume_is_open_loop_and_no_slice_of_it_is_published() -> None:
    """pepin.worldmap in the node: /scan is integrated into the same volume the camera writes, the
    snapshot is what a next run resumes from — and NOTHING localises against it. A tracker matching
    the slice it is painting has a null space it cannot see out of (2026-09-18: a cart with its
    wheels blocked walked 7 degrees and 5-7 cm in 35 minutes at fit 0.97-0.99), so the room's own
    geometry is the graph's grid and this node publishes one thing: the surface cloud."""
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert {"WorldMap", "PlanarMount", "SnapshotClock", "ViewGate"} <= sf.imported(node)
    assert sf.assignments(node)["LIDAR_CONFIG"] == "'/ws/config/lidar.json'", "the plane is read"
    assert "/scan" in sf.strings(node)
    calls = sf.calls(node)
    assert "self._world.integrate_scan" in calls and "self._world.integrate_depth" in calls
    assert "self._world.save" in calls
    topics = {s for s in sf.strings(node) if s == "/map" or s.startswith("/map_")}
    assert topics == set(), f"the volume reaches no matcher and no planner: {topics}"
    assert "occupancy_grid" not in sf.imported(node), "no grid leaves this node at all"
    flags = load_table(REPO / NODES / "depth_fusion.py")
    assert {"map_source", "map_hz", "lidar_map", "camera_map", "map_identity"}.isdisjoint(
        set(flags.names)
    ), "the flags that published the volume went with the publication"
    assert not flags.flag("no_return_free").default, "the old behaviour is the default"
    # The snapshot belongs to the graph DATABASE whose frame every voxel was painted in, so a
    # fresh database means a fresh volume rather than a room drawn in coordinates nothing shares.
    assert sf.assignments(node)["DATABASE"] == "'/maps/rtabmap.db'"
    assert "world_path_for" in sf.imported(node)


def test_nothing_is_painted_at_a_pose_nobody_trusts() -> None:
    """Both paint paths ask one predicate (pepin.watch.PaintTrust) before they write, and the
    launch turns both gates ON in every session: a tracker always runs and always publishes a fit
    (World R). They used to come off in SLAM, where none did, or the session fused nothing
    (2026-09-13 14:05, the camera's own case)."""
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert "PaintTrust" in sf.imported(node), "the predicate is the watch's, not a local number"
    assert "self._paint_refusal" in sf.calls(node), "asked before anything is written"
    asked = (REPO / NODES / "depth_fusion.py").read_text().count("self._paint_refusal(")
    assert asked == 2, "both paint paths ask it: the camera's frame and the lidar's revolution"
    assert "/localization/sigma" in sf.strings(node), "the tracker's own sigma, where it speaks"
    flags = load_table(REPO / NODES / "depth_fusion.py")
    assert flags.flag("lidar_fit_gate").default is True, "a revolution is gated like a frame"
    assert flags.flag("paint_sigma_m").default == PAINT_SIGMA_M
    assert flags.flag("paint_sigma_m").range == (0.01, 2.0)
    passed = sf.strings(sf.tree(VSLAM_LAUNCH))
    assert "lidar_fit_gate:=true" in passed, "on in every session, like fit_gate"


def test_the_graphs_bend_moves_the_volume_and_the_carts_recovery_never_does() -> None:
    """The map follows the loop closure: the node carries the volume by the graph's own bend
    before it paints into it, on both paint paths (a camera frame and a revolution).

    THE BEND AND NOT ``map -> odom``, in every mode, and that is World R's own correction to
    itself. The volume is painted at the BOARD TRACKER's pose now, and that edge moves for two
    opposite reasons — the room bent (follow it) and the CART was found after drifting (do not,
    or the painted room is dragged off the real one by the whole size of the recovery). Only the
    optimised node poses separate the two, because a re-localisation leaves every node where it
    was, so the signal is their change between two ``/rtabmap/mapGraph`` messages
    (:class:`pepin.graphbend.GraphBend`). There is no loop: nothing localises against the volume.
    """
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert "CorrectionFollower" in sf.imported(node)
    assert "GraphBend" in sf.imported(node), "the room's own movement, not the cart's"
    assert "/rtabmap/mapGraph" in sf.strings(node), "the optimised node poses come from there"
    body = {f.name: f for f in ast.walk(node) if isinstance(f, ast.FunctionDef)}
    follow = body["_follow"]
    assert "self._world.shift" in sf.calls(follow), "the move lives in pepin.worldmap"
    assert "self._world.shift" not in sf.calls(node) - sf.calls(follow), "and nowhere else"
    assert "self._bend.drift" in sf.unparsed(follow, ast.Attribute), "the graph's accumulated bend"
    assert "self._tf.pose" not in sf.calls(follow), (
        "map -> odom moves when the CART is found, so it is never what the volume follows"
    )
    assert "self._bend.observe" in sf.calls(body["_on_graph"]), "one graph in, one increment out"
    assert "map_to_odom" not in sf.unparsed(body["_on_graph"], ast.Attribute), (
        "the message carries it and this node deliberately does not read it"
    )
    for path in ("_fuse", "_on_scan_work"):
        assert "self._follow" in sf.calls(body[path]), f"{path} follows before it paints"
        # ...and refuses to paint while a move is owed: an observation placed under the new
        # correction and fused into a volume still standing in the old one is carried past the
        # truth by the whole of that move when it lands (30 cm closure -> a wall 20 cm out,
        # scratch/follow_refute.py, 2026-09-14)
        guards = [
            n
            for n in ast.walk(body[path])
            if isinstance(n, ast.If) and "self._follow(" in ast.unparse(n.test)
        ]
        assert guards, f"{path} paints only when the follower lets it"
        assert all(any(isinstance(b, ast.Return) for b in guard.body) for guard in guards), (
            f"{path} returns when the volume owes the graph a move"
        )
    flags = load_table(REPO / NODES / "depth_fusion.py")
    assert flags.flag("follow_correction").default is True
    for name in ("follow_correction_min_m", "follow_correction_min_deg", "follow_correction_min_s"):
        assert flags.flag(name).range is not None, f"{name}: a threshold is bounded"
        assert flags.flag(name).live, f"{name}: tunable while a map is being built"
    # ...and the resample law is a live switch whose default REVERSED on 2026-09-18, when
    # LidarLaw.beam_footprint took away blend's reason for being it: with the free space in front
    # of a wall weakly weighted, the weighted average is pulled into the wall and widens it — 430
    # occupied cells to 516, worst cell 5.85 cm out, against nearest's 431 and exactly half a
    # voxel (tests/unit/test_follow_correction.py, scratch/volume_shift_cost.py for the ms).
    law = flags.flag("follow_correction_law")
    assert law.choices == ("blend", "nearest") and law.live and law.default == "nearest"
    reads = sf.unparsed(follow, ast.Subscript)
    assert "self._switches['follow_correction_law']" in reads, "the move reads the law it uses"


def test_the_paint_gates_are_on_in_every_session_now_that_a_tracker_always_runs() -> None:
    """They were set by hand in the first world-map SLAM session (2026-09-13 14:05): the volume
    fused 0 frames until fit_gate came off, because no tracker ran in SLAM and /localization_fit
    never arrived. Under World R a tracker always runs and always publishes a fit, so both gates
    are on in every session — and they stay live flags, so a session can still compare A against B
    without a restart."""
    vslam = sf.tree(VSLAM_LAUNCH)
    passed = {s for s in sf.strings(vslam) if s.startswith(("fit_gate:=", "lidar_fit_gate:="))}
    assert passed == {"fit_gate:=true", "lidar_fit_gate:=true"}, passed
    assert not any("fit_gate" in s for s in sf.unparsed(vslam, ast.JoinedStr)), "no mode decides"
    fusion = load_table(REPO / NODES / "depth_fusion.py")
    assert fusion.flag("fit_gate").live and fusion.flag("fit_gate").default is True
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert "self._switches.on" in sf.calls(node), "the node reads the flag, not the mode"


def test_the_cart_s_lean_is_one_thing_every_consumer_takes_from() -> None:
    """One estimator (pepin.lean through the kit's LeanFeed, off /imu/data_raw — which crosses
    the bridge in both modes, so a laptop node reads it directly and no /lean topic is needed),
    one flag name and one meaning in every node that uses it — imu_lean switches the estimator
    in all three and the poser in the two that place a frame — off until it is measured on the
    robot, and the lean in each of their report lines. The tracker is deliberately not among
    them: it runs on the board and already refuses an IMU subscription for a number the EKF
    gives it."""
    from pepin.deployment import BOARD_PUBLISHES, bridge_allow
    from pepin.lean import LEAN_QUALITY_FLOOR

    assert "imu/data_raw" in BOARD_PUBLISHES
    for mode in ("split", "vision"):
        allowed = bridge_allow("laptop", mode)["subscribers"][0]
        assert re.compile(allowed).search("/imu/data_raw"), mode
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
    assert "self._poser.base_in_map" in sf.calls(fusion), "the scan's pose is the poser's"
    gate = load_table(REPO / NODES / "depth_fusion.py").flag("lean_gate_deg")
    assert gate.live and gate.default == 3.0 and gate.range == (0.0, 90.0)
    assert "leaned_out" in sf.strings(fusion), "the report line counts what the gate dropped"
    # and a lean gravity never voted for is no lean: one floor, in both nodes that place a
    # measurement, read by the poser so the gate and the pose make the same decision
    for name in ("depth_fusion", "depth_stream"):
        floor = load_table(REPO / NODES / f"{name}.py").flag("lean_min_quality")
        assert floor.live and floor.default == LEAN_QUALITY_FLOOR, name
        assert floor.range == (0.0, 1.0), name
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
        # FLAGS itself, or flags_for(...): the same table with a depth source's own defaults
        table = ast.unparse(switches[0].args[1])
        assert len(switches) == 1 and (table == "FLAGS" or table.startswith("flags_for(")), (
            path.name
        )
        assert "self.add_on_set_parameters_callback" not in sf.calls(node), path.name
        # A name built per sensor (tof_bridge's f"{name}_x") is not a literal and cannot be
        # compared with a flag's name here; every literal one is.
        declared = {
            ast.literal_eval(c.args[0])
            for c in sf.calls_to(node, "self.declare_parameter")
            if isinstance(c.args[0], ast.Constant)
        }
        assert not declared & set(flags.names), f"{path.name}: a flag declared twice"
        for flag in flags:
            assert flag.description, f"{path.name}: {flag.name} needs a sentence"
    assert {"depth_stream", "depth_fusion", "relocalizer", "neck_state"} <= tables.keys()


def test_a_sensor_is_muted_where_it_is_published_and_both_bridges_know_the_same_two_names() -> None:
    """Switching a sensor off used to mean `ros/feature.sh imu off`: a restart of the whole
    board stack, a minute long, every live flag on it lost. The mute is the same test without
    the restart — the node that publishes the sensor stops publishing, live, and a consumer sees
    what a dead sensor looks like. One node name, `base_bridge`, two implementations
    (robot.launch.py picks one), so the two parameter names have to exist in both or
    `ros/sensor.sh mute imu` reaches whichever bridge is running and does nothing."""
    flags = load_table(REPO / NODES / "base_bridge.py")
    assert flags.names == ("imu_publish", "odom_publish")
    for name in flags.names:
        assert flags.flag(name).live and flags[name] is True, f"{name}: on, and live, or no test"
    cpp = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    for name in flags.names:
        assert f'declare_parameter<bool>("{name}", true)' in cpp, f"{name} missing from the C++"
        assert f'get_parameter("{name}").as_bool()' in cpp, f"{name} read per message, not once"
    assert "switch_state()" in cpp, "the report line names the switches (CLAUDE.md rule 19)"
    # The board's bridge is the one that reads the chip, so `mute imu` has to reach it there.
    from pepin.deployment import node_host

    assert node_host("base_bridge") == ("board", "pepin-ros")


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


def test_the_tracker_matches_the_lidar_here_and_takes_the_camera_as_a_measurement() -> None:
    """The architecture of 2026-09-13, held as a contract. The lidar's revolutions reach the
    tracker through one path (pepin.sources.SourceFeed: the anchor's scan released once the
    odometry covers it, the others carried to its moment, Localizer.update_from matching them
    all around one prediction). The camera's scans do NOT: they are matched on the laptop that
    produces them and arrive here as pose measurements (pepin.measurements), carried to the
    update that takes them by the same history and fused by information — because matching them
    on this board took the tracker to 147 ms a revolution, 4.7 Hz and 50 cm p90 of live error
    (scratch/drive_bisect.py, runs 0238-0241). The flags name the sources and the fusion; every
    source's word goes out on /localization/sources and the feed's status is in the report
    line."""
    from pepin.deployment import LAPTOP_PUBLISHES, VISION_LAPTOP_PUBLISHES

    node = sf.tree(f"{NODES}/relocalizer.py")
    assert {"/localization/measurement", "/localization/sources"} <= sf.strings(node)
    assert "/depth_scan" not in sf.strings(node), "the camera's scans are not matched here"
    assert "/contact_scan" not in sf.strings(node)
    assert {"SourceFeed", "SourceRegistry", "ScanObservation"} <= sf.imported(node)
    assert {"MeasurementGate", "RemoteMeasurement"} <= sf.imported(node)
    assert "ScanGate" not in sf.imported(node), "the feed is the gate: one trigger path"
    calls = sf.calls(node)
    assert {
        "self._feed.offer",
        "self._feed.take",
        "self._feed.gather",
        "self._feed.picture",
        "self._feed.full_picture",
        "self._feed.status",
        "self._measurements.offer",
        "self._measurements.take",
        # With no scan source driving, the remote gates drive between them — the camera's word,
        # the pose graph's — one update per call and never one per gate (pepin.measurements).
        "remote_update",
        "loc.update_from",
        "loc.sources_report",
        "target.switch",  # every flag is written to whichever object names it (``switches``)
    } <= calls
    assert "loc.update" not in calls and "self._gate.take" not in calls, "one path, not two"
    assert len(sf.calls_to(node, "loc.update_from")) == 2, "the scan's update and the camera's"
    flags = load_table(REPO / NODES / "relocalizer.py")
    # since 2026-09-16 the graph is a source by default: without the lidar the pose rides the
    # graph's words, not dead reckoning (Artem: "по умолчанию должно быть всё сразу")
    assert flags["sources"] == ("lidar", "graph") and flags["fusion"] is True
    assert flags.flag("sources").choices == ("lidar", "depth", "contact", "camera", "graph")
    assert flags["measurement_max_age_s"] == 0.5
    # The camera's scans still cross for the costmap; the pose it measures crosses beside them.
    assert {"depth_scan", "contact_scan"} <= set(LAPTOP_PUBLISHES)
    assert "localization/measurement" in VISION_LAPTOP_PUBLISHES


def test_the_pose_graph_reaches_the_tracker_as_a_measurement_of_its_own() -> None:
    """RTAB-Map's graph is the second localisation on this robot and it never owns a frame on
    the board: its answer travels as one more measurement, on a topic of its own, into a gate of
    its own named `graph` (the camera's gate fuses everything it holds into one word called
    `camera`, so a graph word dropped in there would move the pose under the camera's name).
    The route is pinned on both ends — a bridged topic whose two ends ask for different QoS gets
    a route decided by a race — and the word is refused unless the sources flag names it."""
    from pepin.deployment import VISION_LAPTOP_PUBLISHES, bridged_qos
    from pepin.sources import DEFAULT_SOURCES, GRAPH

    topic = "/localization/graph_measurement"
    assert "localization/graph_measurement" in VISION_LAPTOP_PUBLISHES, "vision mode only"
    assert bridged_qos(topic) == ("reliable", 5), "one QoS, both ends, no race"
    board = sf.tree(f"{NODES}/relocalizer.py")
    assert topic in sf.strings(board)
    assert f"bridged_qos_profile({'GRAPH_MEASUREMENT_TOPIC'})" in sf.unparsed(board, ast.Call)
    assert {"self._graph.offer", "self._graph.take", "self._graph.forget"} <= sf.calls(board)
    assert "MeasurementGate" in sf.imported(board) and "GRAPH" in sf.imported(board)
    graph = next(s for s in DEFAULT_SOURCES if s.name == GRAPH)
    assert graph.remote, "no scan of it here: it is never matched and never anchors"
    laptop = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert topic in sf.strings(laptop)
    assert {"graph_measurement", "compose"} <= sf.imported(laptop)
    # ONE FRAME: the graph's optimised map frame IS `map`, so the word is the localisation itself
    # and there is nothing between the two to learn, fit or table.
    assert "graph_anchor" not in sf.imported(laptop) and "Tie" not in sf.imported(laptop)
    assert "/rtabmap/localization_pose" in sf.strings(laptop), "a localisation, not an integration"
    flags = load_table(REPO / NODES / "rtabmap_frame.py")
    assert flags["graph_measurement"] is True, (
        "on since 2026-09-14: harmless beside the lidar (test A), the pose of a lidar-less cart"
    )
    gone = {"graph_odom", "graph_tie_from_pairs", "graph_word_from_nodes", "fresh_frame"}
    assert gone.isdisjoint(set(flags.names)), (
        "the tie and the node table are gone, and so are the flags that chose between them"
    )


def test_the_known_map_graph_rides_the_filter_s_own_odometry() -> None:
    """CLAUDE.md's one odometry: RTAB-Map is built on the EKF's odom -> base_link, not on the
    tracker's pose, which teleports when it relocalises and made every loop closure unacceptable
    (a neighbour edge of 0.888 m against a 0.244 m sigma, error ratio 3.64 over
    RGBD/OptimizeMaxError's 3.0, 2026-09-14). It owns no transform, in any situation, and the
    table that used to put the graph back on the tracker's pose is gone with the argument that
    selected it."""
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
    assert "publish_tf" in sf.strings(vslam), "and RTAB-Map owns no transform in any of them"
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert {"odom", "base_link"} <= sf.strings(frame), "the odometry a word is stamped by"
    assert "self._lookup.transform" in sf.calls(frame)


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
    (``subscribe_sensor_data``, mutually exclusive with every other subscription), and it
    publishes no transform — the board's tracker owns map -> odom. The way back to the
    synchronised triple is one launch argument, and it is the only other place a subscription is
    named."""
    vslam = sf.tree(VSLAM_LAUNCH)
    tables = _launch_dicts(VSLAM_LAUNCH)
    assert [name for name, t in tables.items() if any(k.startswith("Grid/") for k in t)] == [
        "RTABMAP"
    ], "one grid, one table"
    assert [name for name, t in tables.items() if "map_frame_id" in t] == ["RTABMAP"], "one frame"
    assert [name for name, t in tables.items() if "subscribe_sensor_data" in t] == [
        "RTABMAP",
        "TRIPLE_SUBSCRIPTIONS",
    ], "the table and the way back, and nothing else"
    table = tables["RTABMAP"]
    assert table["subscribe_sensor_data"] is True
    for name in ("subscribe_depth", "subscribe_rgb", "subscribe_scan", "subscribe_odom"):
        assert table[name] is False, f"{name}: rtabmap turns it off anyway; say it out loud"
    # WHO OWNS map -> odom, and it is one switch: the two values of publish_tf in this file are
    # the tracker's (False, the node's own parameter) and the localiser's (True, in
    # PUBLISH_MAP_TO_ODOM, merged by rtabmap_parameters only under PEPIN_LOCALIZER=rtabmap). Never
    # a broadcaster of our own: the transform is RTAB-Map's own or it is nobody's here.
    assert sf.dict_items(vslam)["publish_tf"] == {"False", "True"}
    assert _rtabmap("PUBLISH_MAP_TO_ODOM")["publish_tf"] is True
    assert not {"TransformBroadcaster", "StaticTransformBroadcaster"} & sf.imported(vslam)

    # One literal topic name, written on both sides of the contract (read from the sources: this
    # file never imports a ROS node, and rclpy is not installed here).
    node = sf.assignments(sf.tree(f"{NODES}/sensor_pack.py"))
    assert node["SENSOR_DATA_TOPIC"] == "'/rtabmap/sensor_data'"
    assert sf.assignments(vslam)["SENSOR_DATA_TOPIC"] == node["SENSOR_DATA_TOPIC"]
    assert "('sensor_data', SENSOR_DATA_TOPIC)" in sf.unparsed(vslam, ast.Tuple)
    assert "pepin_bringup.sensor_pack" in sf.strings(vslam)
    assert "_after_ghost('/sensor_pack')" in sf.unparsed(vslam, ast.Call)

    # The arguments: the five ros/laptop.sh passes are all still declared, and every argument
    # that used to SELECT A MODE is gone (World R: one database, one grid, one arrangement).
    # camera_only survives as one flag of one node, not as a table.
    arguments = {ast.unparse(c.args[0]) for c in sf.calls_to(vslam, "DeclareLaunchArgument")}
    laptop = (REPO / "ros/laptop.sh").read_text()
    start = laptop.index("ros2 launch pepin_bringup vslam.launch.py")
    command = laptop[start : laptop.index(">/dev/null", start)]
    passed = {f"'{name}'" for name in re.findall(r'"?(\w+):=\$', command)}
    assert len(passed) == 5, passed
    assert passed <= arguments, (
        f"ros/laptop.sh passes what this launch no longer declares: {passed - arguments}"
    )
    assert "'sensor_pack'" in arguments and "'memory'" in arguments
    for gone in ("'graph_odom'", "'slam'", "'resume'", "'world_map'", "'room'", "'map_source'"):
        assert gone not in arguments, f"{gone} selected a mode that no longer exists"
    passes_sources = [s for s in sf.unparsed(vslam, ast.JoinedStr) if "sources:=" in s]
    assert len(passes_sources) == 1, "camera_only reaches exactly one node's sources flag"
    assert {"camera", "camera,lidar"} <= sf.strings(vslam), "the two rosters it picks between"


def test_the_laptop_localizer_matches_the_camera_where_the_camera_is() -> None:
    """The other half of the same rule, on the laptop: /depth_scan and /contact_scan are
    subscribed LOCALLY (they are published on this machine — no bridge hop), matched in a small
    window around the board's belief carried to the scan's stamp, and published as one JSON
    measurement. BOTH HALVES MATCH ON ONE GRID — the map the board's tracker is on — because a fan
    matched against a slice of the volume it helped paint is a loop with no gauge in it."""
    node = sf.tree(f"{NODES}/laptop_localizer.py")
    assert "super().__init__('laptop_localizer')" in sf.unparsed(node, ast.Call)
    assert {"/depth_scan", "/contact_scan", "/localization/measurement"} <= sf.strings(node)
    assert "/map_camera" not in sf.strings(node), "nothing localises against the volume"
    assert "TRACKED_MAP_TOPIC" in sf.imported(node), "the grid the tracker itself adopted"
    assert {"RemoteMeasurement", "Localizer", "OdomHistory"} <= sf.imported(node)
    calls = sf.calls(node)
    assert {"localizer.measure", "RemoteMeasurement.of", "remote.to_json"} <= calls
    assert {"apply_motion", "relative_motion"} <= calls, "the belief is carried to the scan"
    flags = load_table(REPO / NODES / "laptop_localizer.py")
    assert flags["camera_sources"] == ("depth", "contact")
    assert flags["camera_match_hz"] == 5.0 and flags["camera_min_fit"] == 0.25
    assert flags["camera_window_m"] == 0.09 and flags["camera_window_deg"] == 9.0
    assert flags["global_watch"] is True, "the watchdog half is untouched"


def test_the_tracker_adopts_one_map_and_never_treats_a_new_picture_as_a_kidnap() -> None:
    """ONE MAP ON ONE TOPIC (World R): RTAB-Map's loop-closed grid on /map, and there is no flag
    left that chooses between pictures of the room because there is no second picture. What is
    still a decision is WHEN a re-rendered grid is adopted, and that is pepin.mapping.MapChoice's,
    not the node's."""
    from pepin.mapping import MAP_TOPIC

    tracker = load_table(REPO / NODES / "relocalizer.py")
    assert "map_topic" not in dict(tracker.as_dict()), "one topic: nothing to choose"
    assert MAP_TOPIC == "map", "and both ends spell it here"
    assert tracker["map_refresh_s"] == 2.0, (
        "a newer grid is adopted no faster than the board can afford: an adoption is ~65 ms on an"
        " A53 and the rebuild duty budget allows one every 1.9 s (scratch/map_adoption_cost.py),"
        " while RTAB-Map offers one a second (map_always_update at Rtabmap/DetectionRate 1.0)"
    )
    assert tracker["map_fallback_s"] == 10.0, "the patience before a cold boot takes its cache"
    assert tracker["carry_pose_across_maps"] is True, "a new picture of the room is not a kidnap"
    relocalizer = sf.tree(f"{NODES}/relocalizer.py")
    assert "MapChoice" in sf.imported(relocalizer), "the decision lives in pepin, not in the node"
    assert "map_shift" in sf.imported(relocalizer), "and so does what a bend did to the map"
    assert "self._choice.offer" in sf.calls(relocalizer)


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
    # arrive empty. Two arguments default to empty and each is resolved BEFORE it is used:
    # ``database`` to a path (``... or DATABASE``), which reaches no node as an override at all,
    # and ``camera`` to the rig ``camera_rig`` answers with — a name from config/camera.json,
    # which is never empty because that reader raises instead of shrugging.
    empty_by_default = {
        ast.unparse(c.args[0]).strip("'")
        for c in sf.calls_to(sf.tree(VSLAM_LAUNCH), "DeclareLaunchArgument")
        if ast.unparse(sf.keywords(c).get("default_value", ast.Constant(None))) == "''"
    }
    assert empty_by_default == {"database", "camera"}
    assert not [ln for ln in overrides if "database:=" in ln]
    assert "rig = camera_rig(" in launch, "the rig is resolved in the launch, once"
    assert 'f"camera:={rig}"' in launch, "and it is the resolved word that reaches the node"


def test_a_reset_empties_the_volume_and_nothing_falls_back_to_a_picture() -> None:
    """``/fusion/reset`` and the self-heal empty the volume, and with no pgm in the loop "empty"
    means empty. Nothing outside this node reads it, so emptying it costs no tracker anything — it
    costs the surface cloud until the sensors have painted one again."""
    src = (REPO / NODES / "depth_fusion.py").read_text()
    assert src.count("WorldMap(self._spec, self._mount") == 3, (
        "the volume is built in three places only: the placeholder __init__ holds until the"
        " starting state is known, the starting state's own, and the reset's"
    )
    assert src.count("self._world = self._fresh_world()") == 2, "the reset and the self-heal"
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
    assert backend.kind == "choice" and set(backend.choices) == {"remote", "local", "auto"}
    assert backend.default == "local" and backend.env == "PEPIN_DEPTH_BACKEND" and backend.live
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
    told = flags_sh("flag", "depth_stream", "wall_anchor")
    assert told.returncode == 0, told.stderr
    assert told.stdout.startswith("depth_stream/wall_anchor: bool, default on\n")
    for label in ("What:", "Default:", "On when:", "Off when:"):
        assert f"\n{label}" in told.stdout, label
    assert "19.1 -> 16.1 %" in told.stdout, "the measured reason, with its numbers"
    readme = (REPO / "ros/README.md").read_text()
    assert "ros/flags.sh" in readme[readme.index("## Feature flags") :]


def test_the_floor_anchors_the_depth_and_leans_with_the_imu() -> None:
    """The depth node snaps floor pixels to the floor plane (switchable), the plane leans with
    the cart, and the IMU mount the laptop would apply is the one the board publishes
    (roll +90 deg: the chip's Y up)."""
    node = sf.tree(f"{NODES}/depth_stream.py")
    flags = load_table(REPO / NODES / "depth_stream.py")
    assert flags["floor_anchor"] is True, "default on"
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
    assert mount["roll_deg"] == 90.0 and mount["pitch_deg"] == 0.0 and mount["z_m"] == 0.10
    # The launch publishes that file, not a copy of its numbers: every rotation and
    # translation value of its static transforms is a name, never a literal.
    robot = sf.tree(ROBOT_LAUNCH)
    assert sf.assignments(robot)["IMU"] == "MOUNTS.imu.transform()"
    entries = sf.dict_items(robot)
    assert "ix" in entries["rotation.x"] and "imu_z" in entries["translation.z"]
    for key, values in entries.items():
        if key.startswith(("rotation.", "translation.")):
            assert all(value.isidentifier() for value in values), (key, values)


def test_the_laptop_mounts_the_library_live_not_a_copy() -> None:
    """A copy of src/pepin went stale whenever the bridge watch restarted a container: the
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
    nav = sf.tree(NAV_LAUNCH)
    container = sf.keywords(sf.calls_to(nav, "ComposableNodeContainer")[0])
    assert ast.literal_eval(container["respawn"]) is True
    assert 0.0 < float(ast.literal_eval(container["respawn_delay"])) <= 5.0


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
        modules = [s for s in strings if s.startswith("pepin_bringup.")]
        # a node's ghost-wait prefix names the wait module too: the node is what the call runs
        others = [m for m in modules if m != "pepin_bringup.ghost_wait"]
        module = next(iter(others), next(iter(modules), None))
        key = module.removeprefix("pepin_bringup.") if module else str(keywords["executable"])
        found[key] = keywords
    return found


def _respawning(launch: dict[str, dict[str, object]]) -> set[str]:
    return {name for name, keywords in launch.items() if keywords.get("respawn") is True}


def test_a_node_comes_back_by_itself_but_the_watches_exit_on_purpose() -> None:
    """A code change is one kicked process, not a container restart: the launches respawn our
    nodes (and Foxglove's bridge) two seconds after they exit. The watches must not: their exit
    is the signal (bridge_watch -> Shutdown, ghost_wait -> OnProcessExit starts the nodes, and
    respawned it would start every node twice). RTAB-Map stays out too: the graph is its state
    and its crash must stay visible. Its database is never wiped by the launch (no ``-d``): the
    bridge watch restarts the container whenever the board's bridge is new, and a launch that
    wiped it lost the map each time; an empty start is ``ros/laptop.sh vslam --fresh``."""
    vslam_launch = sf.tree(VSLAM_LAUNCH)
    for callee in ("Node", "ExecuteProcess"):
        assert not any("arguments" in sf.keywords(c) for c in sf.calls_to(vslam_launch, callee))
    laptop = (REPO / "ros/laptop.sh").read_text()
    fresh = laptop[laptop.index("    vslam)") : laptop.index("    start)")]
    assert "--fresh) FRESH=true ;;" in fresh and 'rm -f "$HERE"/maps/rtabmap.db' in fresh
    assert fresh.index("rm -f") < fresh.index("docker run")
    vslam = _launch_processes("vslam.launch.py")
    nav = _launch_processes("nav.launch.py")
    assert _respawning(vslam) == {
        "camera_stream",
        "depth_stream",
        "contact_scan",
        "depth_fusion",
        "laptop_localizer",
        # The one input RTAB-Map reads (2026-09-19). It keeps nothing across a restart but a
        # second of each source's stamps, which is what it takes to measure a period again.
        "sensor_pack",
        "rtabmap_frame",
        # The room's vocabulary: its book is on disk beside the database, so a restart loses
        # nothing but the moment before the first graph arrives.
        "places",
        "foxglove_bridge",
        # The camera as a third odometry: rtabmap's node and ours. Neither carries state the
        # way RTAB-Map's graph does — rgbd_odometry's is the last frame, and a restart of it is
        # a jump pepin.visual_odometry.VoGate drops.
        "rgbd_odometry",
        "visual_odometry",
    }
    assert _respawning(nav) == {"relocalizer", "run_recorder", "goal_server", "slam_frame"}
    # The board's sensor launch runs one of our processes too: the neck node (ros/feature.sh
    # neck on). The drivers around it are ROS packages the container restarts with the launch.
    robot = _launch_processes("robot.launch.py")
    # tof_bridge since 2026-09-21: it died once on the robot and stayed dead, and a near-field
    # sensor that silently never comes back is worse than one that was never on.
    assert _respawning(robot) == {"neck_state", "tof_bridge"}
    for launch in (vslam, nav):
        for watch in ("ghost_wait", "bridge_watch"):
            assert "respawn" not in launch[watch], watch
    for launch in (vslam, nav, robot):
        for name, keywords in launch.items():
            if keywords.get("respawn"):
                assert 0.0 < float(str(keywords["respawn_delay"])) <= 5.0, name
    assert "respawn" not in nav["link_watch"] and "respawn" not in vslam["rtabmap"]
    assert "on_exit" in vslam["bridge_watch"] and "on_exit" in nav["bridge_watch"]


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


def test_the_hook_checks_the_shell_scripts_and_the_board_s_books_come_home() -> None:
    """Every ros/*.sh and board/*.sh must parse before a commit; the pose the tracker writes on
    the laptop's mount is not a tracked file; the places books the board edits are fetched, not
    pushed (the tracked copies went stale), and turn_full refuses to turn under a goal."""
    hook = (REPO / ".githooks/pre-commit").read_text()
    assert 'for script in ros/*.sh board/*.sh; do bash -n "$script"; done' in hook
    ignored = (REPO / ".gitignore").read_text().splitlines()
    assert "ros/maps/last_pose.json" in ignored
    fetch = (REPO / "ros/fetch.sh").read_text()
    assert "--include='*.places.yaml' --exclude='*'" in fetch
    assert "root@$BOARD:/root/pepin-ros/maps/" in fetch
    sync = (REPO / "ros/sync.sh").read_text()
    assert "--exclude 'maps/*.places.yaml'" in sync, "the board's book is the truth: never pushed"
    assert "--exclude 'maps/map_cache.json'" in sync, (
        "the board's own map cache exists only there: a deploy with --delete must not remove it"
    )
    assert "--exclude 'maps/last_pose.json'" in sync, "nor the pose the tracker wrote down"
    tracked = subprocess.run(
        ["git", "ls-files", "ros/maps"], capture_output=True, text=True, cwd=REPO, check=True
    ).stdout
    assert ".places.yaml" in tracked, "the books stay tracked; fetch.sh refreshes them"
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
    link = sf.tree(f"{NODES}/link_watch.py")
    assert any(
        isinstance(n, ast.For) and ast.unparse(n.iter) == "BOARD_ACTIONS" for n in ast.walk(link)
    )
    assert "f'/{action}/_action/cancel_goal'" in sf.unparsed(link, ast.JoinedStr)
    thin = (REPO / "ros/thin.sh").read_text()
    status = thin[thin.rindex("    *)") :]
    assert status.rstrip().endswith('echo" ;;\nesac'), "the states are printed, never judged"


def test_one_node_can_be_kicked_without_a_container_restart() -> None:
    """laptop.sh kick / thin.sh kick: SIGINT to one process (never -9, never the container),
    then the log is tailed for the line the node prints once up — a line its source really
    contains. The names a kick knows are exactly the nodes the launches respawn, and an unknown
    name is refused before any host is touched."""
    import os

    known: dict[str, set[str]] = {}
    for script in ("laptop.sh", "thin.sh"):
        src = (REPO / "ros" / script).read_text()
        kick = src[src.index("    kick)") :].split("\n    *)")[0]
        assert "pkill -INT -f" in kick and "pkill -9" not in kick, script
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
    # Foxglove's bridge and rtabmap's rgbd_odometry are not ours to kick: the kick sends SIGINT
    # to a "pepin_bringup.<module>" command line, and neither is one.
    kickable = (_respawning(vslam) - {"foxglove_bridge", "rgbd_odometry"}) | {"goal_server"}
    assert known["laptop.sh"] == kickable
    # Everything of ours the board respawns is kickable: the navigation half and the neck node
    # of the sensor launch (a code change on the board is one kicked process, never a restart).
    assert known["thin.sh"] == _respawning(nav) | _respawning(robot)


def test_the_graphs_grid_is_the_one_map_and_the_tracker_is_the_one_owner_of_map_to_odom() -> None:
    """World R, as one arrangement. RTAB-Map on the laptop IS the map: the EKF's odometry under
    it, its grid remapped onto /map unconditionally (transient local, which the board's tracker
    adopts and republishes for the costmaps), ONE database that is never wiped here. It publishes
    no transform in any situation — the board's tracker owns map -> odom, always — and the retired
    owner of that edge (slam_frame, from /map_odom) is behind nav.launch.py's slam argument, off,
    so the two can never both hold it."""
    vslam = sf.tree(VSLAM_LAUNCH)
    table = _rtabmap("RTABMAP")
    assert table["odom_frame_id"] == "odom" and table["map_frame_id"] == "map"
    # One owner, named by PEPIN_LOCALIZER: publish_tf False is the tracker's stack (the board
    # owns the edge), True comes from PUBLISH_MAP_TO_ODOM under localizer rtabmap. Both are in
    # this file, and rtabmap_parameters picks one.
    assert sf.dict_items(vslam)["publish_tf"] == {"False", "True"}
    # The grid IS the map — once it is assembled from the graph RTAB-Map LOADED. It leaves RTAB-Map
    # on its own topic and pepin_bringup.rtabmap_frame relays it onto /map (grid_needs_tie): before
    # the first recognition the grid is one scan where the odometry puts the cart, and a tracker
    # that adopts it matches itself (2026-09-19: fit 1.00, 1.26 m off).
    assert "remappings.append(('map', '/rtabmap/grid'))" in sf.unparsed(vslam, ast.Call), (
        "unconditionally: there is no second map left for it to make way for"
    )
    assert "('map', '/map')" not in sf.unparsed(vslam, ast.Tuple), "/map has one publisher"
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert {"/rtabmap/grid", "/map", "grid_needs_tie"} <= set(sf.strings(frame))
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
    assert {"Grid/MaxGroundHeight", "Grid/MaxObstacleHeight", "Grid/NormalsSegmentation"} <= set(
        table
    )
    assert table["RGBD/OptimizeFromGraphEnd"] == "false", "the jump belongs in map -> odom"
    # The RETIRED correction path, kept whole and reachable (CLAUDE.md rule 19): the laptop's
    # message, the board's transform, one publisher each, and a route for it over the bridge.
    board_frame = sf.tree(f"{NODES}/slam_frame.py")
    assert "super().__init__('slam_frame')" in sf.unparsed(board_frame, ast.Call)
    assert sf.assignments(board_frame)["FRAMES"] == "('map', 'odom')"
    assert sf.assignments(board_frame)["CORRECTION_TOPIC"] == "'/map_odom'"
    assert "TransformBroadcaster" in sf.imported(board_frame)
    assert "MapGraph" not in sf.imported(board_frame), "the board's image carries no rtabmap_msgs"
    from pepin.deployment import bridge_allow

    assert re.compile(bridge_allow("laptop", "vision")["publishers"][0]).search("/map_odom")
    # The board: the tracker always, the retired owner only behind the argument, Nav2 the same.
    nav = sf.tree(NAV_LAUNCH)
    calls = sf.unparsed(nav, ast.Call)
    assert "runs_here(side, 'map_server')" in calls, "a served pgm answers to its own argument"
    assert "runs_here(side, 'relocalizer', slam_frame, owner)" in calls, (
        "the tracker answers to BOTH switches: the retired slam_frame argument and the localiser"
    )
    assert "localizer()" in calls, "and the launch resolves PEPIN_LOCALIZER once, at the top"
    assert "runs_here(side, 'slam_frame', slam_frame)" in calls
    assert "pepin_bringup.slam_frame" in sf.strings(nav)
    assert "_after_ghost(admin, '/slam_frame')" in calls
    # The grid the board plans on: the static layer takes the tracker's own, latched, and every
    # planner may route through what nobody has looked at yet — a map that is still growing.
    assert _p("global_costmap")["static_layer"]["map_subscribe_transient_local"] is True
    planners = _p("planner_server")
    for name in planners["planner_plugins"]:
        assert planners[name]["allow_unknown"] is True, name


def test_one_gesture_per_side_brings_the_stack_up_and_one_saves_the_map() -> None:
    """Two gestures, and neither of them names a mode any more (World R). ros/thin.sh puts the
    board on a side and gives it a bridge; every branch of it deletes PEPIN_SLAM, because the
    retired frame owner is a line an operator writes by hand and never something a mode turns on.
    The laptop learns the board's side once, on the only path that talks to it, and records it
    (ros/.mode) for the subcommand that never does. ros/map.sh save freezes the grid into the pair
    map_server would read."""
    thin = _case_blocks((REPO / "ros/thin.sh").read_text())
    assert "slam" not in thin, "the SLAM mode went with World R (2026-09-19)"
    for branch in ("on", "vision", "off"):
        assert "/^PEPIN_SLAM=/d" in thin[branch], f"{branch} leaves the retired owner behind"
        assert "board-slam" not in thin[branch], branch
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "Environment=PEPIN_SLAM_TOOLBOX=false" in unit
    assert "slam:=${PEPIN_SLAM} slam_toolbox:=${PEPIN_SLAM_TOOLBOX}" in unit, (
        "the board's own way to the retired frame owner, by hand in /etc/default/pepin-ros"
    )
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    args = {ast.unparse(c.args[0]) for c in sf.calls_to(bringup, "DeclareLaunchArgument")}
    assert {"'slam'", "'slam_toolbox'", "'nav'"} <= args
    includes = sf.dict_items(bringup)
    assert "LaunchConfiguration('slam')" in includes["slam"]
    # The bridge unit waits for the stack's last node: the tracker, or the retired owner.
    bridge = (REPO / "board/pepin-bridge.service").read_text()
    assert 'pgrep -f "pepin_bringup.(relocalizer|slam_frame)"' in bridge
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "CONFIG=zenoh-bridge-laptop.json" in laptop and "laptop-slam" not in laptop
    assert 'printf \'%s\\n\' "$MODE" > "$HERE/.mode"' in laptop
    assert "ros/.mode" in (REPO / ".gitignore").read_text().splitlines()
    vslam_run = next(
        c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-vslam" in c
    )
    for passed in ('"camera_only:=$CAMERA_ONLY"', '"resume_volume:=$RESUME_VOLUME"'):
        assert passed in vslam_run, passed
    for gone in ("slam:=", "resume:=", "world_map:=", "room:="):
        assert gone not in vslam_run, f"{gone} selected a mode that no longer exists"
    save = (REPO / "ros/map.sh").read_text()
    assert "map_saver_cli" in save and "-t /map" in save and "pepin-vslam" in save
    assert "save_map_timeout" in save
    layout = json.loads((REPO / "ros/foxglove/pepin_slam.json").read_text())
    room = layout["configById"]["3D!slam"]
    assert room["topics"]["/map"]["visible"] and room["topics"]["/fusion/surface"]["visible"]
    assert "/rtabmap/mapPath" in room["topics"]


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
    flags = {ast.unparse(c.args[0]) for c in sf.calls_to(runner, "parser.add_argument")}
    assert {"'--board'", "'--square'", "'--no-window'", "'--print'", "'--images'"} <= flags
    # the mount's tilt was fitted together with the field of view the checkerboard replaces, so
    # the run that writes the intrinsics must say the tilt is now stale — in the log and in the
    # README, the two places someone calibrating ever looks.
    assert any("pitch" in text and "depth_fit_models" in text for text in sf.strings(runner))
    readme = (REPO / "ros/README.md").read_text()
    assert "## Camera calibration" in readme and "mount.pitch_deg" in readme


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
    that never arrived, goto refusing with "not localized". Its one precondition is the C++
    bridge, which is what hands over the transform; the way back to no filter is its own switch.
    """
    robot = sf.tree(ROBOT_LAUNCH)
    condition = ast.unparse(sf.keywords(_node_named(robot, "ekf_filter_node"))["condition"])
    assert "'ekf'" in condition and "use_cpp" in condition
    assert "'imu'" not in condition, "the gyro is a source of this filter, never its switch"
    # The bridge hands over odom -> base_link to whoever publishes it: keyed on the filter, so
    # `imu off` can never leave the edge unpublished nor let both publish it.
    assert "ekf_on = LaunchConfiguration('ekf').perform(context).lower() == 'true'" in sf.unparsed(
        robot, ast.Assign
    )
    assert sf.dict_items(robot)["publish_tf"] == {"not ekf_on", "True"}, (
        "the C++ bridge follows the filter; the Python bridge, which has no filter, keeps it"
    )
    assert sf.dict_items(robot)["imu_enable"] == {"imu_on"}, "the gyro keeps its own switch"
    # On by default, and reachable end to end: one operator gesture, one env var, one argument.
    ekf_arg = next(
        c for c in sf.calls_to(robot, "DeclareLaunchArgument") if ast.unparse(c.args[0]) == "'ekf'"
    )
    assert ast.unparse(sf.keywords(ekf_arg)["default_value"]) == "'true'"
    bringup = sf.tree("ros/pepin_bringup/launch/bringup.launch.py")
    assert "LaunchConfiguration('ekf')" in sf.unparsed(bringup, ast.Call)
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "Environment=PEPIN_EKF=true" in unit and "ekf:=${PEPIN_EKF}" in unit
    assert "ekf) VAR=PEPIN_EKF ;;" in (REPO / "ros/feature.sh").read_text()
    # And without the gyro the heading has to come from somewhere: the wheels' own yaw rate.
    ekf_yaml = yaml.safe_load((REPO / "ros/params/ekf.yaml").read_text())["ekf_filter_node"][
        "ros__parameters"
    ]
    assert ekf_yaml["odom0_config"][11] is True, "the wheels carry the heading when the gyro dies"


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
        "robot_localization subscribes RELIABLE at odom1_queue_size: the laptop's publisher and"
        " the bridge route must agree, or the route's QoS is decided by a race"
    )
    assert bridged_qos("/odom") == ("reliable", 10), (
        "the rest watch reads the board's wheels over the bridge, and base_bridge.cpp writes"
        " /odom RELIABLE ten deep: a best-effort endpoint here would let the route's QoS be"
        " decided by a race no one can see losing"
    )
    source = (REPO / "ros/pepin_bringup/pepin_bringup/visual_odometry.py").read_text()
    for topic in ("VO_TOPIC", "WHEELS_TOPIC"):
        assert f"bridged_qos_profile({topic})" in source, (
            f"{topic} crosses the bridge: its endpoint takes the pinned QoS, not one of its own"
        )


def test_the_visual_odometry_runs_on_the_laptop_behind_one_launch_switch() -> None:
    """CLAUDE.md rule 20: what consumes the camera lives on the laptop, and the board gets a
    finished measurement. Nothing new runs there — the EKF reads one more topic. On this side
    both processes are one argument (``vo``), they start after the ghost wait like every other
    node of this launch, rgbd_odometry publishes no transform (the EKF owns odom -> base_link)
    and takes no guess from TF (a visual odometry seeded with the filter's own answer is not a
    third opinion)."""
    vslam = sf.tree(VSLAM_LAUNCH)
    argument = next(
        c for c in sf.calls_to(vslam, "DeclareLaunchArgument") if ast.unparse(c.args[0]) == "'vo'"
    )
    assert ast.unparse(sf.keywords(argument)["default_value"]) == "'true'"
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
    gate = next(
        c
        for c in sf.calls_to(vslam, "ExecuteProcess")
        if "pepin_bringup.visual_odometry" in ast.unparse(c)
    )
    assert "LaunchConfiguration('vo')" in ast.unparse(sf.keywords(gate)["condition"])
    started = _started_after_ghost_wait(vslam)
    assert {"rgbd_odometry", "vo"} <= started, "both wait for the bridge to forget their ghosts"
    from pepin.deployment import LAPTOP_SLAM_NODES

    assert {"/rgbd_odometry", "/visual_odometry"} <= set(LAPTOP_SLAM_NODES), (
        "a ghost of either would strand the next launch's routes"
    )


def test_the_camera_odometry_crosses_to_the_board_and_never_back() -> None:
    """/vo is published on the laptop and subscribed on the board in every bridge mode; a topic
    allowed as a publisher on both sides loops until nothing crosses at all."""
    import re

    from pepin.deployment import BRIDGE_MODES, bridge_allow

    for mode in BRIDGE_MODES:
        board, laptop = bridge_allow("board", mode), bridge_allow("laptop", mode)
        assert re.compile(laptop["publishers"][0]).search("/vo"), mode
        assert re.compile(board["subscribers"][0]).search("/vo"), mode
        assert not re.compile(board["publishers"][0]).search("/vo"), f"{mode}: /vo would loop"
        for block in (*board.values(), *laptop.values()):
            assert not re.compile(block[0]).search("/vo/raw"), (
                f"{mode}: rgbd_odometry's raw output stays on the laptop"
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


def test_goto_judges_a_drive_on_the_fusion_s_sigma_and_never_on_one_sensor_s_fit() -> None:
    """2026-09-15: goto cancelled its own camera-only drives — "localization lost for 19 s while
    the wheels travelled 1.0 m" — because it read /localization_fit, the LIDAR's scan-to-map
    metric, which is 0.00 wherever no lidar scan scored the pose. The rule it reads now is the
    tracker's fused uncertainty (pepin.watch), the same one the goal server reads, and the
    thresholds live there and nowhere else: this file may not grow a second copy of them."""
    goto = sf.tree("ros/tools/goto_ros.py")
    assert {"SIGMA_TOPIC", "Sigma", "Preflight", "BlindDriveWatch", "source_words"} <= sf.imported(
        goto
    ), "the readings and the rule both come from pepin.watch"
    assert "0.15" not in sf.assignments(goto).values(), "the sigma thresholds live in pepin.watch"
    assert "0.25" not in sf.assignments(goto).values()
    assert "blind.observe" in sf.calls(goto) and "certainty.sigma" in sf.calls(goto)
    # ...and the drive's own extra condition survives: a poor reading while the wheels stand
    # still is a stuck cart, and the recoveries (odom frame) can still work it free.
    armed = [line for line in sf.unparsed(goto, ast.IfExp) if "BlindDriveWatch(" in line]
    assert armed and armed[0].endswith("else None"), (
        "online SLAM publishes neither a sigma nor a fit: a watch that read that silence as 0.00"
        " would cut every healthy drive of the mode, so it is not armed there at all"
    )
    assert sf.assignments(goto)["LOST_TRAVEL_M"] == "1.0"
    assert sf.assignments(goto)["LOST_FOR_S"] == "15.0"
    assert "travelled_while_lost" in sf.calls(goto)


def test_goto_prints_a_preflight_before_it_sends_a_goal() -> None:
    """Three checks, one line each, and a refusal names the reading behind it. The old gate
    printed "localized: fit 0.78" or "not localized" — a verdict with no evidence in it, and on
    a camera drive the number it came from was 0.00 by construction."""
    goto = sf.tree("ros/tools/goto_ros.py")
    defined = {f.name for f in ast.walk(goto) if isinstance(f, ast.FunctionDef)}
    assert {"preflight", "cancel_all"} <= defined and "Certainty" in sf.names(goto)
    gate = next(
        f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "ensure_localized"
    )
    body = ast.unparse(gate)
    assert "preflight(nav, certainty)" in body, "the tracker stack is judged by the preflight"
    assert "tracker_here(nav)" in body, "...and online SLAM still by the map frame, as before"
    flight = next(
        f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "preflight"
    )
    printed = ast.unparse(flight)
    assert "check.line()" in printed, "one line per check, whatever the verdict"
    script = (REPO / "ros/goto.sh").read_text()
    assert "preflight certainty" in script, "the header shows the operator what to expect"


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
    assert float(sf.assignments(goto)["CANCEL_CONFIRM_S"]) <= 3.0, "confirmed while he watches"
    main = next(f for f in ast.walk(goto) if isinstance(f, ast.FunctionDef) and f.name == "main")
    lines = ast.unparse(main).splitlines()
    cancelled = next(i for i, line in enumerate(lines) if "cancel_all(node)" in line)
    built = next(i for i, line in enumerate(lines) if "BasicNavigator()" in line)
    assert cancelled < built, "no commander is built for a cancel: building one is what crashed"
    assert "rclpy.ok()" in ast.unparse(main), "a context already shut down refuses every call"
    assert "stop.sh" in " ".join(sf.strings(goto)), "the hard stop is named where cancel can fail"


def test_rmw_switch_defaults_to_zenoh_and_cyclone_stays_reachable() -> None:
    """NEW RULE (2026-09-20): PEPIN_RMW unset is rmw_zenoh — measured CPU-neutral on the board,
    order-free at start and self-healing over restarts — and PEPIN_RMW=cyclone is the whole old
    stack (CycloneDDS plus the two bridges), kept and startable. The default is the same in the
    shell, in the three units and in Python, and every container is TOLD which one it is."""
    assert rmw_is_zenoh({}), "no PEPIN_RMW means rmw_zenoh and the routers"
    assert rmw_is_zenoh({"PEPIN_RMW": "zenoh"})
    assert not rmw_is_zenoh({"PEPIN_RMW": "cyclone"}), "the bridges are one variable away"
    lib = (REPO / "ros/lib.sh").read_text()
    assert 'PEPIN_RMW="${PEPIN_RMW:-zenoh}"' in lib, "the shell default is zenoh too"
    for name in ("pepin-ros.service", "pepin-zrouter.service", "pepin-bridge.service"):
        unit = (REPO / "board" / name).read_text()
        assert "Environment=PEPIN_RMW=zenoh" in unit, name
        assert "EnvironmentFile=-/etc/default/pepin-ros" in unit, "the value survives a reboot"
    run = (REPO / "ros/run.sh").read_text()
    assert 'RMWENV="-e PEPIN_RMW=cyclone"' in run, "a cyclone container is told it is one"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "-e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PEPIN_RMW=cyclone" in laptop
    for build in ("ros/build.sh", "ros/laptop-build.sh"):
        assert ":zenoh" in (REPO / build).read_text(), "a rebuild keeps the default's tag current"


def test_rmw_zenoh_swaps_the_image_and_cyclone_starts_the_old_one() -> None:
    """ros/run.sh starts the image with rmw_zenoh in it by default, and the plain image under
    cyclone, whose own ENV is rmw_cyclonedds_cpp."""
    run = (REPO / "ros/run.sh").read_text()
    assert 'IMAGE="${PEPIN_IMAGE:-pepin-ros}"' in run, "cyclone starts pepin-ros, as before"
    assert 'IMAGE="${PEPIN_IMAGE:-pepin-ros:zenoh}"' in run, (
        "zenoh starts the image with rmw_zenoh in it"
    )
    assert "RMW_IMPLEMENTATION=rmw_zenoh_cpp" in run
    # Start order must not be able to kill a node: the router may not be up yet.
    assert "ZENOH_ROUTER_CHECK_ATTEMPTS=0" in run


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


def test_zenoh_runs_no_bridge_anywhere() -> None:
    """Under zenoh nothing bridge-shaped starts: not the sidecar, not the watch of it."""
    bridge_unit = (REPO / "board/pepin-bridge.service").read_text()
    assert bridge_unit.count('[ "$PEPIN_RMW" != zenoh ]') == 4, (
        "every guard of the bridge unit — both ExecStartPre, ExecStart and ExecStartPost — "
        "stands the unit down under zenoh"
    )
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "zrouter_up" in laptop, "the laptop starts its own router instead of the bridge"
    assert "if pepin_rmw_is_zenoh; then" in laptop
    # The bridge path itself is untouched and still reachable under the default.
    assert "settle_bridge" in laptop, "the cyclone path keeps its bridge choreography"
    assert "eclipse/zenoh-bridge-ros2dds:1.7.0" in laptop
    for launch in ("nav.launch.py", "vslam.launch.py"):
        src = (REPO / "ros/pepin_bringup/launch" / launch).read_text()
        assert "rmw_is_zenoh()" in src, (
            f"{launch} must not start a watch of a bridge that is not there"
        )
        assert "pepin_bringup.bridge_watch" in src, f"{launch} keeps the watch for the cyclone path"


def test_the_camera_rig_decides_who_measures_the_depth() -> None:
    """One choice, the rig's name: a stereo head gets the matched eyes as its depth source, the
    webcam gets the network, and the depth node is told at launch (it decides its subscriptions)."""
    launch = (REPO / VSLAM_LAUNCH).read_text()
    assert 'f"depth_source:={depth_source(rig)}"' in launch
    assert 'return "stereo" if CameraConfig.load(config_file("camera.json"), rig).stereo' in launch
    config = json.loads((REPO / "config/camera.json").read_text())
    assert "rig" in config["stereo"] and "rig" not in config["overview"]
