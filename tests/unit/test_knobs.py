"""config/knobs.json: the nodes' numbers, read at start and kept as live parameters.

A knob is a number with a default, a range and one line of note (pepin.flags.load_knobs); a node
passes its own to node_kit.Switches beside its FLAGS, so ``ros2 param set`` and ros/flags.sh reach
a knob the way they reach a flag. Where a library or a node already names the number, the file
repeats it and this test holds the two equal: one value, written twice, never allowed to drift.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import source_facts as sf

from pepin.camera_grid import GRID_RESOLUTION_M, GRID_SIZE_M
from pepin.contact import CONTACT_MAX_RANGE
from pepin.depth import SCALE_CEILING
from pepin.flags import FlagSet, knob, knobs_of, load_knobs, load_table, read_knobs, with_knobs
from pepin.gaze import GazeSettings
from pepin.global_descriptor import MAX_NULL_SHARE
from pepin.graphmode import PNP_REPROJ_PX, PNP_REPROJ_RANGE_PX
from pepin.lean import LEAN_QUALITY_FLOOR, SCAN_LEAN_GATE_DEG
from pepin.marks_audit import MATCH_CELLS, RADIUS_M
from pepin.path_gaze import PathGazeLaw, ReverseLaw
from pepin.snapshot import PAIR_PERIODS
from pepin.stall_look import AHEAD_M as STALL_AHEAD_M
from pepin.stall_look import CLUSTER_M as STALL_CLUSTER_M
from pepin.stall_look import MARGIN_M as STALL_MARGIN_M
from pepin.tsdf import band_half_z_m
from pepin.volume_scan import MARKS_MAX_Z_M, MARKS_MIN_Z_M

REPO = Path(__file__).resolve().parents[2]
NODES = "ros/pepin_bringup/pepin_bringup"


def _node_constant(node: str, name: str) -> float:
    return float(ast.literal_eval(sf.assignments(sf.tree(f"{NODES}/{node}.py"))[name]))


# (node, knob) -> the number the library or the node already names
NAMED = {
    ("contact_scan", "max_range"): CONTACT_MAX_RANGE,
    ("depth_fusion", "lean_gate_deg"): SCAN_LEAN_GATE_DEG,
    ("depth_fusion", "lean_min_quality"): LEAN_QUALITY_FLOOR,
    ("depth_fusion", "marks_min_z"): MARKS_MIN_Z_M,
    ("depth_fusion", "grid_size_m"): GRID_SIZE_M,
    ("depth_fusion", "grid_resolution_m"): GRID_RESOLUTION_M,
    ("depth_fusion", "band_half_z"): band_half_z_m(REPO / "config/fusion.json"),
    ("depth_stream", "scale_ceiling"): SCALE_CEILING,
    ("depth_stream", "lean_min_quality"): LEAN_QUALITY_FLOOR,
    ("marks_audit", "radius_m"): RADIUS_M,
    ("marks_audit", "match_cells"): MATCH_CELLS,
    ("rtabmap_frame", "pnp_reproj_px"): PNP_REPROJ_PX,
    ("rtabmap_frame", "descriptor_null_share"): MAX_NULL_SHARE,
    ("sensor_pack", "pair_periods"): PAIR_PERIODS,
    # the gaze arbiter: its library's defaults, and the marks' band its columns are read in
    ("gaze", "frames"): GazeSettings().frames,
    ("gaze", "settle_tol_deg"): GazeSettings().settle_tol_deg,
    ("gaze", "move_timeout_s"): GazeSettings().move_timeout_s,
    ("gaze", "frame_period_s"): GazeSettings().frame_period_s,
    ("gaze", "ttl_navigation_s"): GazeSettings().ttl_for(1),
    ("gaze", "ttl_person_s"): GazeSettings().ttl_for(2),
    ("gaze", "ttl_driving_s"): GazeSettings().ttl_for(4),
    ("gaze", "stall_ahead_m"): STALL_AHEAD_M,
    ("gaze", "stall_margin_m"): STALL_MARGIN_M,
    ("gaze", "stall_cluster_m"): STALL_CLUSTER_M,
    ("gaze", "stall_match_cells"): MATCH_CELLS,
    ("gaze", "stall_column_bottom_m"): MARKS_MIN_Z_M,
    ("gaze", "stall_column_top_m"): MARKS_MAX_Z_M,
    ("gaze", "path_lookahead_s"): PathGazeLaw().lookahead_s,
    ("gaze", "path_deadband_deg"): PathGazeLaw().deadband_deg,
    ("gaze", "path_pan_clamp_deg"): PathGazeLaw().pan_clamp_deg,
    ("gaze", "reverse_pan_deg"): ReverseLaw().pan_deg,
    ("gaze", "reverse_tilt_deg"): ReverseLaw().tilt_deg,
    ("gaze", "reverse_min_s"): ReverseLaw().min_s,
    ("gaze", "reverse_rear_m"): ReverseLaw().rear_m,
}
NODE_NAMED = {
    ("depth_stream", "tf_dead_s"): "TF_DEAD_S",
    ("depth_stream", "depth_reach_m"): "DEPTH_REACH_M",
    ("sensor_pack", "pack_hz"): "DETECTION_RATE_HZ",
    ("sensor_pack", "place_timeout_s"): "PLACE_TIMEOUT_S",
}


def test_every_knob_loads_as_a_live_bounded_number_of_a_node_that_passes_it() -> None:
    blocks = read_knobs(REPO / "config/knobs.json")
    assert blocks, "the file carries the nodes' knobs"
    for node in blocks:
        path = REPO / NODES / f"{node}.py"
        assert path.is_file(), f"{node}: no such node"
        knobs = knobs_of(blocks, node)
        assert len(knobs) and all(k.kind == "number" and k.live and k.range for k in knobs)
        assert all(k.description for k in knobs), f"{node}: every knob carries its note"
        flags = load_table(path)
        with_knobs(flags, knobs)  # no knob shadows a flag of the same node
        assert f"load_knobs('{node}')" in sf.unparsed(sf.tree(f"{NODES}/{node}.py"), ast.Call), (
            f"{node} must hand its knobs to Switches, or they are dead numbers"
        )
    assert load_knobs("no_such_node", REPO / "config/knobs.json").names == ()


def test_a_knob_the_code_already_names_is_the_same_number() -> None:
    for (node, name), value in NAMED.items():
        assert load_knobs(node, REPO / "config/knobs.json")[name] == value, f"{node}.{name}"
    for (node, name), constant in NODE_NAMED.items():
        knob_value = load_knobs(node, REPO / "config/knobs.json")[name]
        assert knob_value == _node_constant(node, constant), f"{node}.{name} vs {constant}"
    pnp = load_knobs("rtabmap_frame", REPO / "config/knobs.json").flag("pnp_reproj_px")
    assert pnp.range == PNP_REPROJ_RANGE_PX


def test_a_knob_is_refused_outside_its_range_and_an_int_counts() -> None:
    top_k = knob("xfeat_top_k", {"default": 2048, "range": [256, 8192], "note": "corners"})
    assert top_k.integer and top_k.parse("4096") == 4096
    with pytest.raises(ValueError, match="outside"):
        top_k.parse(9000)
    with pytest.raises(ValueError, match="a knob needs"):
        knob("x", {"default": 1.0})
    with pytest.raises(ValueError, match="a knob is a number"):
        knob("x", {"default": True, "range": [0, 1]})
    table = with_knobs(FlagSet(), FlagSet(top_k))
    assert table.set("xfeat_top_k", "1024") == 2048 and table["xfeat_top_k"] == 1024


def test_a_key_starting_with_an_underscore_is_a_note_not_a_node() -> None:
    data = json.loads((REPO / "config/knobs.json").read_text())
    assert "_note" in data and "_note" not in read_knobs(REPO / "config/knobs.json")
