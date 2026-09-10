"""Regression guards for the navigation configuration and the operator scripts.

Each assertion is a lesson paid for on the robot: the value it pins was the cause of a failed run.
"""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())


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
    assert '"$HERE/stop.sh"' in goto or "stop.sh" in goto
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
    import json

    from pepin.deployment import BASE_MAX_ANGULAR_RAD_S, BASE_MAX_LINEAR_M_S

    cfg = json.loads((REPO / "config/base.json").read_text())
    base = cfg["max_speed_m_s"]
    assert (base, cfg["max_yaw_rate_rad_s"]) == (BASE_MAX_LINEAR_M_S, BASE_MAX_ANGULAR_RAD_S)
    launch = (REPO / "ros/pepin_bringup/launch/robot.launch.py").read_text()
    assert '"max_linear_m_s": BASE_MAX_LINEAR_M_S' in launch, "the C++ bridge keeps its 0.25 cap"
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
    launch = (REPO / "ros/pepin_bringup/launch/robot.launch.py").read_text()
    assert re.search(r"HULL = \(?\s*hull_box\(", launch), (
        "the lidar hull filter must use the contact band"
    )
    bridge = (REPO / "ros/pepin_bringup/pepin_bringup/tof_bridge.py").read_text()
    assert "_MIN_RANGE_M = CONTACT_BAND_M" in bridge, "ToF readings inside the band must be dropped"
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
    launch = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    assert '"autostart": autostart_for(side)' in launch
    assert '"side": side' in launch, "the goal server must know it is the laptop half"
    server = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    assert "next_transition(" in server and "ChangeState" in server


def test_the_recorder_is_its_own_node_on_the_board_side() -> None:
    """The split's first tapes were written on the laptop and had no scans: the recorder lives
    where the sensors are, and the goal server only sends it a command."""
    launch = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    assert 'runs_here(side, "run_recorder")' in launch and "pepin_bringup.run_recorder" in launch
    setup = (REPO / "ros/pepin_bringup/setup.py").read_text()
    assert "run_recorder = pepin_bringup.run_recorder:main" in setup
    server = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    assert "RunRecorder(" not in server and "curl" not in server
    assert "RUN_COMMAND_TOPIC" in server and "RUN_STATUS_TOPIC" in server


def test_the_camera_slam_lives_beside_the_tracker_never_over_it() -> None:
    """RTAB-Map on the laptop publishes its own frame and no map -> odom: the board's tracker
    keeps the reflexes' frame; the camera is nominal until calibrated and says so."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vslam_launch", REPO / "ros/pepin_bringup/launch/vslam.launch.py"
    )
    src = (REPO / "ros/pepin_bringup/launch/vslam.launch.py").read_text()
    assert '"publish_tf": False' in src and '"map_frame_id": "rtabmap"' in src
    assert 'namespace="rtabmap"' in src, "its relative 'map' output must not land on /map"
    assert '"subscribe_scan": True' in src and '"Reg/Strategy": "1"' in src
    assert "pepin_bringup.camera_stream" in src and spec is not None
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "vslam.launch.py" in laptop and "config:/ws/config" in laptop
    # a mono camera cannot seed a loop-closure transform: ICP from identity does
    assert '"RGBD/LoopClosureIdentityGuess": "true"' in src


def test_the_bridge_routes_only_what_the_split_needs_and_only_one_way() -> None:
    """Routing every node's parameter services wedged the link; routing a topic both ways looped
    it (a bridge finds its own writer and calls it a publisher). Each side's config lists its own
    publishers and the other side's subscribers, generated from pepin.deployment."""
    import json
    import re

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
    unit = (REPO / "board/pepin-bridge.service").read_text()
    assert (
        "zenoh-bridge-board.json" in unit
        and "zenoh-bridge-laptop.json" in (REPO / "ros/laptop.sh").read_text()
    )


def test_the_laptop_half_restarts_with_the_board_s_bridge() -> None:
    """A subscription does not follow a bridge through its restart: both laptop launches carry
    the bridge watch (exit -> launch shutdown -> container restart), the containers restart on
    their own, and the board's bridge is settled BEFORE the containers start, never after."""
    for launch in ("nav.launch.py", "vslam.launch.py"):
        src = (REPO / "ros/pepin_bringup/launch" / launch).read_text()
        assert "pepin_bringup.bridge_watch" in src and "Shutdown(" in src, launch
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert laptop.count("--restart unless-stopped") == 2
    assert laptop.index("settle_bridge  # BEFORE") < laptop.index(
        "docker run -d --name pepin-laptop"
    )
    after = laptop.split("docker run -d --name pepin-laptop")[1]
    assert "\nsettle_bridge" not in after and "        settle_bridge" not in after


def test_the_laptop_halves_start_their_nodes_only_after_their_ghosts_are_gone() -> None:
    """The bridge keeps routes by node name: a half restarted within its predecessor's DDS lease
    loses its routes when the ghost expires. Both laptop launches start their ROS nodes on the
    ghost wait's exit, the names they wait for are the names they create, and laptop.sh lets a
    container leave properly before replacing it."""
    from pepin.deployment import laptop_launch_nodes

    for launch, half in (("nav.launch.py", "nav"), ("vslam.launch.py", "slam")):
        src = (REPO / "ros/pepin_bringup/launch" / launch).read_text()
        assert "pepin_bringup.ghost_wait" in src
        assert re.search(r"OnProcessExit\(\s*target_action=ghost_wait", src), launch
        assert f'laptop_launch_nodes("{half}")' in src, launch
    vslam = (REPO / "ros/pepin_bringup/launch/vslam.launch.py").read_text()
    assert 'name="rtabmap"' in vslam and 'namespace="rtabmap"' in vslam
    camera = (REPO / "ros/pepin_bringup/pepin_bringup/camera_stream.py").read_text()
    assert 'super().__init__("camera_stream")' in camera
    depth = (REPO / "ros/pepin_bringup/pepin_bringup/depth_stream.py").read_text()
    assert 'super().__init__("depth_stream")' in depth and 'name="foxglove_bridge"' in vslam
    assert laptop_launch_nodes("slam") == (
        "/camera_stream",
        "/depth_stream",
        "/rtabmap/rtabmap",
        "/rtabmap_frame",
        "/foxglove_bridge",
    )
    nav = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    goal = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    assert 'name=f"lifecycle_manager_navigation_{side}"' in nav
    assert 'name=f"nav2_container_{side}"' in nav and 'super().__init__("goal_server")' in goal
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "docker stop -t 15" in laptop and "docker rm -f pepin-vslam" not in laptop
    assert "docker rm -f pepin-laptop" not in laptop


def test_the_camera_is_a_depth_sensor_scaled_by_the_lidar() -> None:
    """The laptop turns the camera's frames into depth images (a network on CPU, the lidar sets
    the scale), RTAB-Map builds its grid from the lidar and that depth in 3D, the cloud is drawn
    by the operator's Foxglove connected to the laptop, and the image carries the weights."""
    vslam = (REPO / "ros/pepin_bringup/launch/vslam.launch.py").read_text()
    assert "pepin_bringup.depth_stream" in vslam and '"subscribe_depth": True' in vslam
    assert '("depth/image", "/camera/depth")' in vslam
    for key in ('"Grid/Sensor": "2"', '"Grid/3D": "true"', '"Grid/MaxObstacleHeight"'):
        assert key in vslam, key
    gated = vslam[vslam.index("OnProcessExit(") :].split(")")[0]
    for name in ("camera", "depth", "rtabmap", "frame", "foxglove"):
        assert name in gated.split("on_exit=")[1], name
    node = (REPO / "ros/pepin_bringup/pepin_bringup/depth_stream.py").read_text()
    assert '"/camera/depth"' in node and "scale_from_samples" in node and "DepthScale" in node
    image = (REPO / "ros/Dockerfile.laptop").read_text()
    assert (
        "whl/cpu" in image and "HF_HUB_OFFLINE=1" in image and "ros-jazzy-foxglove-bridge" in image
    )
    assert "Depth-Anything-V2-Metric-Indoor-Small-hf" in image
    import json

    layout = json.loads((REPO / "ros/foxglove/pepin_nav.json").read_text())
    assert layout["configById"]["3D!nav"]["topics"]["/rtabmap/cloud_map"]["colorMode"] == "rgb"
    # RTAB-Map's own grid fights the static map for the floor: off by default in the nav view
    assert layout["configById"]["3D!nav"]["topics"]["/rtabmap/map"]["visible"] is False
    # the 3D view: the cloud, the path, the scan and the robot; no grid lies over the voxels
    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    shown = {t for t, c in room["topics"].items() if c.get("visible")}
    assert "/rtabmap/cloud_map" in shown and "/rtabmap/mapPath" in shown and "/scan" in shown
    # the two floor grids that fought each other stay out; the local costmap is asked for
    assert not shown & {"/map", "/rtabmap/map", "/global_costmap/costmap"}
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "-p 8765:8765" in laptop.split("docker run -d --name pepin-vslam")[1].split("\n")[0]


def test_the_camera_s_depth_reaches_the_costmap_and_its_frame_follows_the_graph() -> None:
    """The depth folded onto the plane goes to the board as /depth_scan and marks the local
    costmap; the camera stamps frames with the board's capture time; map -> rtabmap comes
    from RTAB-Map's own correction on the laptop, not from a fixed identity on the board."""
    from pepin.deployment import LAPTOP_PUBLISHES

    assert "depth_scan" in LAPTOP_PUBLISHES
    params = (REPO / "ros/params/nav2_params.yaml").read_text()
    local = params[params.index("local_costmap:") : params.index("global_costmap:")]
    assert "observation_sources: scan dynamic depth_scan" in local
    source = local[local.index("        depth_scan:") :]
    for line in ("topic: /depth_scan", 'data_type: "LaserScan"', "inf_is_valid: True"):
        assert line in source, line
    node = (REPO / "ros/pepin_bringup/pepin_bringup/depth_stream.py").read_text()
    assert '"/depth_scan"' in node and "depth_to_scan" in node
    camera = (REPO / "ros/pepin_bringup/pepin_bringup/camera_stream.py").read_text()
    assert "capture_time(headers)" in camera and "cv2.VideoCapture" not in camera
    vslam = (REPO / "ros/pepin_bringup/launch/vslam.launch.py").read_text()
    assert "pepin_bringup.rtabmap_frame" in vslam
    nav = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    assert "odom_to_rtabmap" not in nav
    frame = (REPO / "ros/pepin_bringup/pepin_bringup/rtabmap_frame.py").read_text()
    assert '"/rtabmap/mapGraph"' in frame and '"map", "rtabmap"' in frame
    assert '"odom_frame_id": "map"' in vslam  # RTAB-Map's odometry is the tracker's pose
    import json

    room = json.loads((REPO / "ros/foxglove/pepin_3d.json").read_text())["configById"]["3D!room"]
    assert room["topics"]["/depth_scan"]["visible"]
    assert room["topics"]["/local_costmap/costmap"]["visible"]


def test_the_drive_ends_on_position_and_the_goal_server_turns_to_the_heading() -> None:
    """RPP with reversing cannot rotate in place: the tree's FollowPath judges xy only, and the
    goal server pivots the residual heading with the behaviour server's Spin before "done"."""
    params = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
    controller = params["controller_server"]["ros__parameters"]
    assert "xy_only_goal_checker" in controller["goal_checker_plugins"]
    xy_only = controller["xy_only_goal_checker"]
    assert xy_only["xy_goal_tolerance"] == controller["general_goal_checker"]["xy_goal_tolerance"]
    assert xy_only["yaw_goal_tolerance"] >= 3.14
    tree = (REPO / "ros/params/pepin_nav_to_pose.xml").read_text()
    assert 'goal_checker_id="xy_only_goal_checker"' in tree
    server = (REPO / "ros/pepin_bringup/pepin_bringup/goal_server.py").read_text()
    assert 'ActionClient(self, Spin, "spin")' in server and "_pivot_to(yaw_deg" in server
    assert "PIVOT_TOLERANCE_DEG = 11.5" in server  # the general checker's 0.20 rad


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
    launch = (REPO / "ros/pepin_bringup/launch/nav.launch.py").read_text()
    container = launch[launch.index("container = ComposableNodeContainer(") :]
    container = container[: container.index(")\n")]
    assert "respawn=True" in container and "respawn_delay=" in container
