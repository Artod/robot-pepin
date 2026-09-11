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
    """One cell is a mark the planner squeezes past; the sensor cannot say where in the cone."""
    for costmap in ("local_costmap", "global_costmap"):
        for sensor in ("front", "left", "right"):
            layer = _p(costmap)[f"tof_{sensor}_layer"]
            assert layer["inflate_cone"] == 1.0
            assert layer["phi"] <= 0.5, "phi must model the real 27 degree cone"


def test_each_tof_owns_its_own_layer() -> None:
    """2026-09-08: one shared layer let the dead front sensor clear the side sensors' marks."""
    for costmap in ("local_costmap", "global_costmap"):
        layers = [name for name in _p(costmap)["plugins"] if name.startswith("tof_")]
        assert len(layers) == 3, f"{costmap}: the three ToF must not share a probability grid"
        for sensor, layer in zip(("front", "left", "right"), layers, strict=True):
            assert _p(costmap)[layer]["topics"] == [f"/tof/{sensor}"]


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
        "feature.sh",
        "laptop.sh",
        "thin.sh",
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
    for costmap in ("local_costmap", "global_costmap"):
        for sensor in ("front", "left", "right"):
            assert _p(costmap)[f"tof_{sensor}_layer"]["no_readings_timeout"] == 0.0


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


def test_new_objects_get_a_berth_in_both_costmaps() -> None:
    """The rings around unexplained returns are derived from the hull, never typed twice: a point
    planner's ring makes up the width its 6 cm band lacks plus the toes, a footprint planner's
    only the toes; nothing is ringed so close that a ring could touch the cart's own outline.
    Both costmaps mark the rings and never raytrace-clear through them."""
    from pepin.dynamic import COSTMAP_CELL_M, berth_for, near_exclusion_m, point_planner_ring_m
    from pepin.footprint import HULL

    point = berth_for("GridBased")
    assert point.ring_m == point_planner_ring_m(HULL) >= HULL.half_width_m - HULL.inscribed_radius_m
    assert point.near_m == near_exclusion_m(point.ring_m, HULL)
    for costmap in ("local_costmap", "global_costmap"):
        params = _p(costmap)
        assert params["resolution"] == COSTMAP_CELL_M
        layer = params["obstacle_layer"]
        assert "dynamic" in layer["observation_sources"].split()
        assert layer["dynamic"]["marking"] is True and layer["dynamic"]["clearing"] is False


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


def test_the_camera_slam_lives_beside_the_tracker_never_over_it() -> None:
    """RTAB-Map on the laptop publishes its own frame and no map -> odom: the board's tracker
    keeps the reflexes' frame; the camera is nominal until calibrated and says so."""
    vslam = sf.tree(VSLAM_LAUNCH)
    params = sf.dict_items(vslam)
    assert params["publish_tf"] == {"False"} and params["map_frame_id"] == {"'rtabmap'"}
    rtabmap = sf.keywords(_node_named(vslam, "rtabmap"))
    assert ast.unparse(rtabmap["namespace"]) == "'rtabmap'", (
        "its relative 'map' output must not land on /map"
    )
    assert params["subscribe_scan"] == {"True"} and params["Reg/Strategy"] == {"'1'"}
    assert "pepin_bringup.camera_stream" in sf.strings(vslam)
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "vslam.launch.py" in laptop and "config:/ws/config" in laptop
    # a mono camera cannot seed a loop-closure transform: ICP from identity does
    assert params["RGBD/LoopClosureIdentityGuess"] == {"'true'"}


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
        "/map",
        "/odometry/filtered",
        "/tof/front",
        "/tracker_pose",
        "/dynamic_obstacles",
        "/pepin/run_status",
    ):
        assert pub_b.search(name) and sub_l.search(name), name
        assert not re.compile(laptop["publishers"][0]).search(name), f"{name} would loop"
    for name in ("/plan", "/pepin/run", "/laptop/heartbeat", "/planner_selector", "/rtabmap/map"):
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
    unit = (REPO / "board/pepin-bridge.service").read_text()
    assert "Environment=PEPIN_BRIDGE_CONFIG=zenoh-bridge-board.json" in unit
    assert "/root/pepin-ros/$PEPIN_BRIDGE_CONFIG:/config.json:ro" in unit
    thin = (REPO / "ros/thin.sh").read_text()
    vision = thin[thin.index("    vision)") : thin.index("    off)")]
    assert "PEPIN_BRIDGE_CONFIG=zenoh-bridge-board-vision.json" in vision
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
    assert laptop.count("--restart unless-stopped") == 2
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
    assert "identity.observe(zid, time.monotonic())" in sf.unparsed(watch, ast.Call)
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
    for module in ("camera_stream", "depth_stream", "goal_server"):
        node = sf.tree(f"{NODES}/{module}.py")
        assert f"super().__init__('{module}')" in sf.unparsed(node, ast.Call), module
    assert laptop_launch_nodes("slam") == (
        "/camera_stream",
        "/depth_stream",
        "/depth_fusion",
        "/rtabmap/rtabmap",
        "/rtabmap_frame",
        "/foxglove_bridge",
    )
    nav = sf.tree(NAV_LAUNCH)
    composed = {ast.unparse(sf.keywords(c)["name"]) for c in sf.calls_to(nav, "ComposableNode")}
    assert "f'lifecycle_manager_navigation_{side}'" in composed
    container = sf.keywords(sf.calls_to(nav, "ComposableNodeContainer")[0])
    assert ast.unparse(container["name"]).startswith("f'nav2_container_{side}' if side != 'all'")
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "docker stop -t 15" in laptop and "docker rm -f pepin-vslam" not in laptop
    assert "docker rm -f pepin-laptop" not in laptop
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

    for name in ("nav.launch.py", "vslam.launch.py"):
        src = sf.tree(f"ros/pepin_bringup/launch/{name}")
        assert any(isinstance(n, ast.FunctionDef) and n.name == "_after_ghost" for n in src.body)
        assert "pepin_bringup.ghost_wait" in sf.strings(src), name
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
        "depth_fusion",
        "rtabmap_frame",
        "foxglove_bridge",
    ):
        assert f"_after_ghost('/{node}')" in sf.unparsed(vslam, ast.Call), node
    wait = sf.tree(f"{NODES}/ghost_wait.py")
    assert {"os.execvp(command[0], command)", "rest.index('--')"} <= sf.unparsed(wait, ast.Call)
    assert any("not waiting" in s for s in sf.strings(wait))
    # The board's container shares the host's network: 127.0.0.1:8000 is its bridge's admin.
    run = (REPO / "ros/run.sh").read_text()
    assert "--network host" in run


def test_the_camera_is_a_depth_sensor_scaled_by_the_lidar() -> None:
    """The laptop turns the camera's frames into depth images (a network on CPU, the lidar sets
    the scale), RTAB-Map builds its grid from the lidar and that depth in 3D, the cloud is drawn
    by the operator's Foxglove connected to the laptop, and the image carries the weights."""
    vslam = sf.tree(VSLAM_LAUNCH)
    params = sf.dict_items(vslam)
    assert "pepin_bringup.depth_stream" in sf.strings(vslam)
    assert params["subscribe_depth"] == {"True"}
    assert "('depth/image', '/camera/depth')" in sf.unparsed(vslam, ast.Tuple)
    assert params["Grid/Sensor"] == {"'2'"} and params["Grid/3D"] == {"'true'"}
    assert "Grid/MaxObstacleHeight" in params
    assert {"camera", "depth", "rtabmap", "frame", "foxglove"} <= _started_after_ghost_wait(vslam)
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert "/camera/depth" in sf.strings(node) and {"beam_pairs", "AffineScale"} <= sf.imported(
        node
    )
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
    # the two floor grids that fought each other stay out; the local costmap is asked for
    assert not shown & {"/map", "/rtabmap/map", "/global_costmap/costmap"}
    laptop = (REPO / "ros/laptop.sh").read_text()
    vslam_run = next(
        c for c in sf.shell_commands(laptop) if "docker run -d --name pepin-vslam" in c
    )
    assert "-p 8765:8765" in vslam_run


def test_the_camera_s_depth_reaches_the_costmap_and_its_frame_follows_the_graph() -> None:
    """The depth folded onto the plane goes to the board as /depth_scan and marks the local
    costmap; the camera stamps frames with the board's capture time; map -> rtabmap comes
    from RTAB-Map's own correction on the laptop, not from a fixed identity on the board."""
    from pepin.deployment import LAPTOP_PUBLISHES

    assert "depth_scan" in LAPTOP_PUBLISHES
    layer = _p("local_costmap")["obstacle_layer"]
    assert "depth_scan" in layer["observation_sources"].split()
    source = layer["depth_scan"]
    assert source["topic"] == "/depth_scan" and source["data_type"] == "LaserScan"
    assert source["inf_is_valid"] is True
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert "/depth_scan" in sf.strings(node) and "depth_to_scan" in sf.calls(node)
    camera = sf.tree(f"{NODES}/camera_stream.py")
    assert "capture_time" in sf.calls(camera) and "cv2.VideoCapture" not in sf.calls(camera)
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.rtabmap_frame" in sf.strings(vslam)
    nav = sf.tree(NAV_LAUNCH)
    assert not any("odom_to_rtabmap" in s for s in sf.strings(nav) | sf.names(nav))
    frame = sf.tree(f"{NODES}/rtabmap_frame.py")
    assert "/rtabmap/mapGraph" in sf.strings(frame)
    assert "('map', 'rtabmap')" in sf.unparsed(frame, ast.Tuple)
    # RTAB-Map's odometry is the tracker's pose
    assert sf.dict_items(vslam)["odom_frame_id"] == {"'map'"}
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    assert room["topics"]["/depth_scan"]["visible"]
    assert room["topics"]["/local_costmap/costmap"]["visible"]


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
    transform from the same file, and ros/sync.sh puts config/ where the board's container
    sees it (beside the library: /ws/pepin_src/config, pepin.deployment.config_file)."""
    robot = sf.tree(ROBOT_LAUNCH)
    assert sf.assignments(robot)["MOUNTS"] == "Mounts.load()"
    assert sf.assignments(robot)["LASER"] == "MOUNTS.lidar.transform()"
    declared = {ast.unparse(c.args[0]) for c in sf.calls_to(robot, "DeclareLaunchArgument")}
    assert not declared & {"'laser_roll'", "'laser_yaw'"} and "MOUNT" not in sf.imported(robot)
    assert "quaternion(laser_roll, laser_pitch, laser_yaw)" in sf.unparsed(robot, ast.Call)
    camera = sf.tree(f"{NODES}/camera_stream.py")
    assert "self._tf('base_link', 'laser', lx, ly, lz, lroll, lpitch, lyaw)" in sf.unparsed(
        camera, ast.Call
    )
    assert "LidarMount.from_json" in sf.calls(camera)
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


def test_the_frames_are_fused_into_one_surface_beside_rtabmap_s_cloud() -> None:
    """The SLAM launch runs the fusion node after the ghost wait; the node loads the grid from
    config/fusion.json and offers its switches as parameters; the 3D layout shows the fused
    surface and hides RTAB-Map's concatenated cloud by default (both stay available)."""
    vslam = sf.tree(VSLAM_LAUNCH)
    assert "pepin_bringup.depth_fusion" in sf.strings(vslam)
    assert "fusion" in _started_after_ghost_wait(vslam)
    node = sf.tree(f"{NODES}/depth_fusion.py")
    assert sf.assignments(node)["CONFIG"] == "'/ws/config/fusion.json'"
    # The four are live switches of the kit (node_kit.Switches), so ros2 param set reaches
    # them and their state is printed in the node's report line (CLAUDE.md rule 19).
    assert {"enabled", "align", "min_weight", "surface_hz"} <= set(sf.dict_items(node))
    assert "Switches" in sf.imported(node) and "self._switches.state" in sf.calls(node)
    assert {"/fusion/reset", "/fusion/surface"} <= sf.strings(node)
    for name in ("pepin_3d.json", "pepin_nav.json"):
        layout = json.loads((REPO / "ros/foxglove" / name).read_text())
        panel = next(v for k, v in layout["configById"].items() if k.startswith("3D"))
        assert panel["topics"]["/fusion/surface"]["visible"] is True
        assert panel["topics"]["/fusion/surface"]["colorMode"] == "rgb"
        assert panel["topics"]["/rtabmap/cloud_map"]["visible"] is False


def test_the_floor_anchors_the_depth_and_leans_with_the_imu() -> None:
    """The depth node snaps floor pixels to the floor plane (switchable), the plane leans with
    the accelerometer, and the IMU mount the laptop would apply is the one the board publishes
    (roll +90 deg: the chip's Y up)."""
    node = sf.tree(f"{NODES}/depth_stream.py")
    assert sf.dict_items(node)["floor_anchor"] == {"True"}, "a live switch, default on"
    assert "/imu/data_raw" in sf.strings(node)
    assert {"floor_anchor", "floor_depth"} <= sf.calls(node)
    # The mount is not read here by hand: one loader for every sensor's place on the cart.
    assert "Mounts" in sf.imported(node) and "Mounts.load" in sf.calls(node)
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
    assert '[ "${2:-}" = --fresh ]' in fresh and 'rm -f "$HERE"/maps/rtabmap.db' in fresh
    assert fresh.index("rm -f") < fresh.index("docker run")
    vslam = _launch_processes("vslam.launch.py")
    nav = _launch_processes("nav.launch.py")
    assert _respawning(vslam) == {
        "camera_stream",
        "depth_stream",
        "depth_fusion",
        "rtabmap_frame",
        "foxglove_bridge",
    }
    assert _respawning(nav) == {"relocalizer", "run_recorder", "goal_server"}
    for launch in (vslam, nav):
        for watch in ("ghost_wait", "bridge_watch"):
            assert "respawn" not in launch[watch], watch
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
    for module in ("camera_stream", "depth_stream", "depth_fusion", "rtabmap_frame"):
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
    assert known["laptop.sh"] == (_respawning(vslam) - {"foxglove_bridge"}) | {"goal_server"}
    assert known["thin.sh"] == _respawning(nav)
