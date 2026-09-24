"""The XFeat + LighterGlue visual registration: the two adapters RTAB-Map loads by path
(ros/xfeat), the parameter sets that travel with the visual strategy (pepin.graphmode) and the
launch table they are set against.

The adapters' arrays are read by RTAB-Map's C++ row by row with no strides, so their shape, dtype
and contiguity are the contract; a live parameter set is only seen for a name the launch table
overrides, so every name a set may carry must be in that table, at the value the set starts from.
The models themselves are exercised by the slow test at the end, when the pinned checkout is here.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import source_facts as sf

from pepin.graphmode import (
    CONFIRM_AGGRESSIVE,
    CONFIRM_PARAMETERS,
    CONFIRM_SINGLE,
    CONFIRM_STOCK,
    FEATURE_PARAMETERS,
    FEATURES_ORB,
    FEATURES_XFEAT,
    PNP_REPROJ_PX,
    PNP_REPROJ_RANGE_PX,
    PROXIMITY_STOCK,
    REGISTRATION_PARAMETERS,
    STRATEGY_ICP,
    STRATEGY_VIS,
    XFEAT_DETECTOR_PATH,
    XFEAT_MATCHER_PATH,
    visual_parameters,
)

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = REPO / "ros" / "xfeat"
VSLAM_LAUNCH = "ros/pepin_bringup/launch/vslam.launch.py"


def _adapter(name: str) -> ModuleType:
    """One adapter module, loaded from its file as RTAB-Map loads it (by path, not by package)."""
    spec = importlib.util.spec_from_file_location(name, ADAPTERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rtabmap_table() -> dict[str, object]:
    """vslam.launch.py's RTAB-Map table, read from the sources (no `launch` package here)."""
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))["RTABMAP"]))


# ---- the parameter sets -------------------------------------------------------------------------
def test_the_visual_strategy_carries_the_flag_s_feature_set() -> None:
    """xfeat is RTAB-Map's Python detector and matcher, fed by re-extraction from the stored
    pictures; orb is RTAB-Map's own defaults."""
    assert visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0) == {
        "Vis/FeatureType": "15",
        "Vis/CorNNType": "6",
        "RGBD/LoopClosureReextractFeatures": "true",
        "Vis/PnPReprojError": "2",
        "Rtabmap/LoopThr": "0.11",
        "RGBD/MaxOdomCacheSize": "10",
        "RGBD/ProximityBySpace": "true",
    }
    assert visual_parameters(STRATEGY_VIS, FEATURES_ORB, 4.0) == {
        "Vis/FeatureType": "8",
        "Vis/CorNNType": "1",
        "RGBD/LoopClosureReextractFeatures": "false",
        "Vis/PnPReprojError": "4",
        "Rtabmap/LoopThr": "0.11",
        "RGBD/MaxOdomCacheSize": "10",
        "RGBD/ProximityBySpace": "true",
    }


def test_the_confirmation_travels_with_the_visual_strategy_while_localising() -> None:
    """Parked under the lamps the second, confirming try never reached Rtabmap/LoopThr 0.11 and
    0 of 244 updates were accepted: aggressive keeps it at 0.05, single accepts the first."""
    aggressive = visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, confirm=CONFIRM_AGGRESSIVE)
    assert (aggressive["Rtabmap/LoopThr"], aggressive["RGBD/MaxOdomCacheSize"]) == ("0.05", "10")
    single = visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, confirm=CONFIRM_SINGLE)
    assert (single["Rtabmap/LoopThr"], single["RGBD/MaxOdomCacheSize"]) == ("0.11", "0")
    for confirm in CONFIRM_PARAMETERS:
        # ICP's weak hypotheses are not tried from the identity guess, and a mapping database
        # keeps RTAB-Map's own rule
        assert visual_parameters(STRATEGY_ICP, FEATURES_XFEAT, 2.0, confirm=confirm) == (
            visual_parameters(STRATEGY_ICP, FEATURES_ORB, 2.0)
        )
        mapping = visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, True, confirm)
        assert mapping.items() >= CONFIRM_PARAMETERS[CONFIRM_STOCK].items()
    with pytest.raises(ValueError):
        visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, confirm="trust")


def test_the_stock_confirmation_is_the_launch_table_s() -> None:
    """The start values must be the table's, or the first set would change them unasked."""
    table = _rtabmap_table()
    for name, value in CONFIRM_PARAMETERS[CONFIRM_STOCK].items():
        assert table[name] == value, name
    assert table["RGBD/ProximityBySpace"] == ("true" if PROXIMITY_STOCK else "false")


def test_the_proximity_search_is_switched_off_only_for_a_localising_camera() -> None:
    """A localised camera's proximity registrations are XFeat matches (1.77 s an update parked
    at the base); the lidar's are a few ms of ICP and most of the graph's links."""
    off = visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, proximity=False)
    assert off["RGBD/ProximityBySpace"] == "false"
    assert visual_parameters(STRATEGY_ICP, FEATURES_XFEAT, 2.0, proximity=False)[
        "RGBD/ProximityBySpace"
    ] == ("true")
    mapping = visual_parameters(STRATEGY_VIS, FEATURES_XFEAT, 2.0, True, proximity=False)
    assert mapping["RGBD/ProximityBySpace"] == "true"


def test_icp_always_carries_orb_whatever_the_flag_says() -> None:
    """Re-extraction also strips a NEW node of its descriptors and 3D (Memory.cpp:6126), and ICP
    is the strategy that maps: a node the lidar teaches must keep ORB words any registration can
    use."""
    assert visual_parameters(STRATEGY_ICP, FEATURES_XFEAT, 2.0) == visual_parameters(
        STRATEGY_ICP, FEATURES_ORB, 2.0
    )
    assert visual_parameters(STRATEGY_ICP, FEATURES_XFEAT, 2.0)["Vis/FeatureType"] == "8"


def test_the_gate_is_a_string_rtabmap_can_read() -> None:
    assert visual_parameters(STRATEGY_VIS, FEATURES_ORB, 2.5)["Vis/PnPReprojError"] == "2.5"
    assert PNP_REPROJ_RANGE_PX[0] <= PNP_REPROJ_PX <= PNP_REPROJ_RANGE_PX[1]
    for table in FEATURE_PARAMETERS.values():
        assert all(isinstance(value, str) for value in table.values())


def test_an_unknown_feature_set_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError):
        visual_parameters(STRATEGY_VIS, "sift", 2.0)


# ---- the launch table ---------------------------------------------------------------------------
def test_every_name_a_live_set_may_carry_is_in_the_launch_table() -> None:
    """CoreWrapper.cpp:362-379: a name the launch never overrode accepts the set and is ignored."""
    table = _rtabmap_table()
    names: set[str] = set()
    for strategy in REGISTRATION_PARAMETERS:
        names |= set(REGISTRATION_PARAMETERS[strategy])
        for features in FEATURE_PARAMETERS:
            for confirm in CONFIRM_PARAMETERS:
                names |= set(visual_parameters(strategy, features, PNP_REPROJ_PX, False, confirm))
    assert names <= set(table), sorted(names - set(table))


def test_the_launch_starts_from_the_set_the_node_believes_is_in_force() -> None:
    """rtabmap_frame re-sends a visual set only when it differs from the last one it believes
    RTAB-Map holds, and it starts believing ORB under ICP at the default gate: that must be the
    table's own values, or a difference would never be sent."""
    table = _rtabmap_table()
    start = visual_parameters(STRATEGY_ICP, FEATURES_ORB, PNP_REPROJ_PX)
    assert {name: table[name] for name in start} == start


def test_the_launch_points_rtabmap_at_the_adapters_the_image_carries() -> None:
    table = _rtabmap_table()
    assert table["PyDetector/Path"] == XFEAT_DETECTOR_PATH == "/opt/xfeat/rtabmap_xfeat.py"
    assert table["PyMatcher/Path"] == XFEAT_MATCHER_PATH == "/opt/xfeat/rtabmap_lighterglue.py"
    assert (
        float(str(table["PyMatcher/Threshold"])) == _adapter("rtabmap_lighterglue").DEFAULT_MIN_CONF
    )
    dockerfile = (REPO / "ros" / "Dockerfile.xfeat").read_text()
    assert "COPY xfeat/rtabmap_xfeat.py xfeat/rtabmap_lighterglue.py /opt/xfeat/" in dockerfile


def _features_kept(tmp_path: Path, apt: list[str], new: list[str]) -> tuple[int, str]:
    """ros/xfeat/build_rtabmap.sh's features_kept on two Version.h excerpts: (status, stderr)."""
    import subprocess

    (tmp_path / "apt").write_text("".join(f"#define RTABMAP_{d}\n" for d in apt))
    (tmp_path / "new").write_text("".join(f"#define RTABMAP_{d}\n" for d in new))
    script = ADAPTERS / "build_rtabmap.sh"
    run = subprocess.run(
        [
            "bash",
            "-c",
            f"eval \"$(sed -n '/^features_kept() {{/,/^}}/p' '{script}')\";"
            f" features_kept '{tmp_path / 'apt'}' '{tmp_path / 'new'}'",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return run.returncode, run.stderr


def test_the_rebuilt_core_must_keep_every_optional_library_the_apt_build_had(
    tmp_path: Path,
) -> None:
    """Optimizer/Strategy and Icp/Strategy are compile-time defaults the launch table does not name
    (GTSAM -> 2, libpointmatcher -> 1): a library the rebuild missed would change the lidar's path
    without a word. The image that exists passes (scratch/xfeat_critic/params_diff.sh: the same 19
    defines plus RTABMAP_PYTHON, all 498 parameter defaults identical)."""
    apt = ["TORO", "G2O", "GTSAM", "POINTMATCHER", "OCTOMAP", "OPENNI2"]
    assert _features_kept(tmp_path, apt, [*apt, "PYTHON"]) == (0, "")
    status, why = _features_kept(tmp_path, apt, [d for d in apt if d != "GTSAM"] + ["PYTHON"])
    assert status == 1 and "RTABMAP_GTSAM" in why
    status, why = _features_kept(tmp_path, apt, apt)
    assert status == 1 and "no RTABMAP_PYTHON" in why
    status, why = _features_kept(tmp_path, ["TORO"], ["TORO", "PYTHON"])
    assert status == 1 and "no RTABMAP_GTSAM" in why, "the floor holds whatever apt's header had"


# ---- the adapters' arrays -----------------------------------------------------------------------
def test_the_detector_answers_in_the_layout_rtabmap_reads() -> None:
    """PyDetector.cpp asserts N x 3 float keypoints and N x dim float descriptors and reads both
    buffers straight: float32, C-contiguous, x/y/score in that order."""
    xfeat = _adapter("rtabmap_xfeat")
    xy = np.array([[10.0, 20.0], [30.5, 40.25]], dtype=np.float64)
    scores = np.array([0.9, 0.1])
    desc = np.arange(2 * 64, dtype=np.float64).reshape(64, 2).T  # a transposed, strided view
    points, descriptors = xfeat.rtabmap_arrays(xy, scores, desc)
    assert points.dtype == np.float32 and points.flags["C_CONTIGUOUS"]
    assert descriptors.dtype == np.float32 and descriptors.flags["C_CONTIGUOUS"]
    assert points.tolist() == [[10.0, 20.0, pytest.approx(0.9)], [30.5, 40.25, pytest.approx(0.1)]]
    assert descriptors.shape == (2, 64) and descriptors[1, 0] == desc[1, 0]


def test_the_detector_answers_nothing_as_empty_arrays_of_the_right_width() -> None:
    xfeat = _adapter("rtabmap_xfeat")
    points, descriptors = xfeat.rtabmap_arrays(np.zeros((0, 2)), np.zeros(0), np.zeros((0, 64)))
    assert points.shape == (0, 3) and descriptors.shape == (0, 64)


def test_the_detector_refuses_counts_that_disagree() -> None:
    with pytest.raises(ValueError):
        _adapter("rtabmap_xfeat").rtabmap_arrays(np.zeros((3, 2)), np.zeros(2), np.zeros((3, 64)))


def test_the_matcher_answers_query_train_pairs_as_int32() -> None:
    """PyMatcher.cpp accepts INT or LONG and turns each row into (queryIdx, trainIdx)."""
    glue = _adapter("rtabmap_lighterglue")
    pairs = glue.as_pairs(np.array([[0, 5], [3, 1]], dtype=np.int64))
    assert (
        pairs.dtype == np.int32
        and pairs.flags["C_CONTIGUOUS"]
        and pairs.tolist() == [[0, 5], [3, 1]]
    )
    assert glue.as_pairs([]).shape == (0, 2)


def test_a_side_with_no_keypoints_is_no_pairs_without_touching_the_model() -> None:
    glue = _adapter("rtabmap_lighterglue")
    none = glue.pairs(
        None, np.zeros((0, 2)), np.ones((4, 2)), np.zeros((0, 64)), np.ones((4, 64)), 800, 600
    )
    assert none.shape == (0, 2)


# ---- the models, where the pinned checkout is ----------------------------------------------------
XFEAT_DIR = Path(os.environ.get("PEPIN_XFEAT_DIR", "/opt/xfeat/accelerated_features"))


@pytest.mark.slow
@pytest.mark.skipif(not (XFEAT_DIR / "weights" / "xfeat.pt").is_file(), reason="no XFeat checkout")
def test_the_adapters_match_a_picture_to_itself_shifted() -> None:
    """The image build runs the same check (ros/Dockerfile.xfeat's last step): tiles, and the
    same tiles 12 px to the left, must match on the shift through RTAB-Map's own two calls."""
    xfeat = _adapter("rtabmap_xfeat")
    glue = _adapter("rtabmap_lighterglue")
    rng = np.random.default_rng(0)
    big = (np.kron(rng.random((62, 82)), np.ones((10, 10))) * 255).astype(np.uint8)
    model = xfeat.load_xfeat(str(XFEAT_DIR))
    a_pts, a_desc = xfeat.features(model, big[:600, :800])
    b_pts, b_desc = xfeat.features(model, big[:600, 12:812])
    matcher = glue.load_lighterglue(str(XFEAT_DIR))
    found = glue.pairs(matcher, b_pts[:, :2], a_pts[:, :2], b_desc, a_desc, 800, 600)
    shift = a_pts[found[:, 1], 0] - b_pts[found[:, 0], 0]
    assert int(np.sum(np.abs(shift - 12.0) < 2.0)) > 50
