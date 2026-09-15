"""The depth node under the ROS stubs: the pipeline and the frame poser wired in.

rclpy is faked (``ros_stubs``), the network is a test's scripted frames, TF is the stub's
buffer; the node is built and its frames processed here as on the laptop. The proof that
matters: with the default flags the published depth and scan are, to the bit, what the node's
own chain (edges -> beam pairs -> law -> drop edges -> scan -> floor anchor, as ``_process``
stood before the pipeline) published — held here as the reference. Then the camera pose from
TF at the frame's stamp, the config as the fallback while TF has no edge, the flags reaching
the stages, and the scan's source when the wall correction is on.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from camera_configs import CALIBRATION, camera_config  # noqa: E402
from pepin_bringup.depth_stream import FLAGS, SCAN_RANGE_M, DepthStream  # noqa: E402
from pepin_bringup.msgs import (  # noqa: E402
    image_from_array,
    pose_from_transform,
    scan_from_ranges,
    stamp_seconds,
)

from pepin.camera import (  # noqa: E402
    OPTICAL_RPY,
    CameraConfig,
    camera_info_arrays,
    mount_transform,
)
from pepin.depth import (  # noqa: E402
    LAW_VERSION,
    UP_LEVEL,
    AffineScale,
    CameraPose,
    Intrinsics,
    apply_affine,
    beam_pairs,
    depth_to_scan,
    drop_edges,
    edge_mask,
    floor_anchor,
    floor_depth,
    optical_heading,
    plane_in_view_from,
    project,
    quaternion_from_matrix,
    save_law,
    scan_points,
    to_base,
)
from pepin.depth_pipeline import (  # noqa: E402
    LIDAR_SIGMA_M,
    PARALLAX_WEIGHT,
    PIPELINE_DEFAULTS,
    FloorPairs,
    FrameContext,
    FrameLaw,
    LidarAnchor,
    ParallaxAnchor,
    grid_of,
    standard_pipeline,
)
from pepin.mounts import load_lidar_mount, rotation_from_rpy  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAMERA_JSON = str(REPO / "config" / "camera.json")
WIDTH, HEIGHT = 640, 360  # what camera_stream publishes at scale 0.5
# The mount and the nominal field of view: the two numbers a calibration never rewrites (it
# writes an intrinsics block beside them). The nodes under test are given a config of their own,
# with the optics pinned to this nominal pinhole — see the build fixture.
CONFIG = CameraConfig.load(CAMERA_JSON)
K, _D, _R, _P = camera_info_arrays(WIDTH, HEIGHT, CONFIG.hfov_deg)
INTR = Intrinsics.from_camera_info(K, WIDTH, HEIGHT)
CONFIG_CAM = CameraPose(*mount_transform(CONFIG)[:3], mount_transform(CONFIG)[4])
# The mount is config/lidar.json's, not a number retyped here: the beams these fakes place
# must sit where the node projects them from.
LIDAR_Z_M = load_lidar_mount().z_m
LIDAR_MOUNT = RigidPose(np.eye(3), np.array([0.0, 0.0, LIDAR_Z_M]))
SCAN_MAX_RANGE = 3.0  # the node's default
WALLS = (1.0, 1.5, 2.5, 3.5, 2.0, 3.0)  # views enough for POOL_MIN_SAMPLES pairs and a spread
LAW = (1.3, 0.03)  # the network's own error: 1 / z = a / D + b


class Param:
    """What ``ros2 param set`` hands the node's callback."""

    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value


class FakeNet:
    """The depth backend as a script: the frames it will answer with, in order, and the two
    attributes the node reads of the real switch."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.mode = "local"
        self.status = "fake"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.shape == (HEIGHT, WIDTH, 3)
        return self.frames.pop(0)


# ---- the synthetic room ------------------------------------------------------------------
def _scene(cam: CameraPose, wall_x: float) -> np.ndarray:
    """The true optical depth, per pixel, of a wall ``wall_x`` ahead in base_link and the floor
    under it, seen from ``cam`` (NaN where a ray meets neither)."""
    rows, _cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    lift = -(rows - INTR.cy) / INTR.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    fwd, up = c + s * lift, -s + c * lift
    with np.errstate(divide="ignore"):
        depth = np.minimum((wall_x - cam.x) / fwd, np.where(up < 0, -cam.z / up, np.inf))
    return np.where(np.isfinite(depth), depth, np.nan)


def _network(z: np.ndarray, seed: int, noise: float = 0.03) -> np.ndarray:
    """What the network says about ``z``: through LAW with relative noise, float32 like the
    real one."""
    rng = np.random.default_rng(seed)
    a, b = LAW
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / ((1.0 / z - b) / a)
    return (d * (1.0 + noise * rng.standard_normal(z.shape))).astype(np.float32)


def _stamp(k: int) -> Any:
    return ros_stubs.Time(sec=1_000 + k, nanosec=250_000_000)


def _image(stamp: Any) -> Any:
    return ros_stubs.Image(
        header=ros_stubs.Header(stamp=stamp, frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        encoding="bgr8",
        step=WIDTH * 3,
        data=bytes(WIDTH * HEIGHT * 3),
    )


def _info(stamp: Any) -> Any:
    return ros_stubs.CameraInfo(
        header=ros_stubs.Header(stamp=stamp, frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        k=list(K),
    )


def _scan(wall_x: float, stamp: Any) -> Any:
    """The lidar on the wall: one beam a degree, the wall within 60 degrees of straight ahead,
    nothing farther round."""
    angles = np.radians(np.arange(-180, 180, dtype=float))
    r = np.where(np.abs(angles) < math.radians(60.0), wall_x / np.cos(angles), np.inf)
    return ros_stubs.LaserScan(
        header=ros_stubs.Header(stamp=stamp, frame_id="laser"),
        angle_min=float(angles[0]),
        angle_max=float(angles[-1]),
        angle_increment=math.radians(1.0),
        range_min=0.1,
        range_max=12.0,
        ranges=[float(v) for v in r],
    )


def _transform(translation: tuple[float, float, float], rotation: np.ndarray) -> Any:
    x, y, z = translation
    qx, qy, qz, qw = quaternion_from_matrix(rotation)
    return ros_stubs.TransformStamped(
        transform=ros_stubs.Transform(
            translation=ros_stubs.Vector3(x=x, y=y, z=z),
            rotation=ros_stubs.Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
    )


def _optical_edge(pitch_deg: float, pan_deg: float = 0.0, z: float = 1.2) -> Any:
    """``base_link -> camera_optical`` as TF composes it: the neck's pitch and pan, then the
    optical turn."""
    link = rotation_from_rpy(0.0, math.radians(pitch_deg), math.radians(pan_deg))
    return _transform((0.05, 0.0, z), link @ rotation_from_rpy(*OPTICAL_RPY))


# ---- the reference: the node's chain as it stood ------------------------------------------
def legacy_process(
    depth32: np.ndarray,
    scan: Any,
    cam: CameraPose,
    law: AffineScale,
    stamp: Any,
    pan: float = 0.0,
) -> tuple[bytes, list[float]] | None:
    """``DepthStream._process`` before the pipeline, line for line: the edge mask on the raw
    depth, the beams of the scan projected through the static mounts (the cart stood still:
    the carry is the identity), the beam pairs into the law, the law applied, the edges
    dropped, the scan before the floor anchor, the floor anchored, both published; ``None``
    when the law did not exist yet. Returns the published image's bytes and scan's ranges.

    ``pan`` is the head's heading the fan is folded through (``scan_honours_pan``): 0 is the
    chain of before 2026-09-15, and a reference built against a TF edge passes that edge's own
    yaw, which a quaternion round trip leaves non-zero even for a head pointing straight."""
    edge = edge_mask(depth32)
    xy = scan_points(np.asarray(scan.ranges), scan.angle_min, scan.angle_increment, SCAN_RANGE_M)
    samples = project(to_base(xy, LIDAR_MOUNT.rotation, LIDAR_MOUNT.translation), cam, INTR)
    a, b = law.observe(beam_pairs(depth32, samples, edge))
    if not law.ready:
        return None
    metric = apply_affine(depth32, a, b)
    metric, _dropped = drop_edges(metric, edge)
    angle_min, step, ranges = depth_to_scan(metric, INTR, cam, pan=pan, max_range=SCAN_MAX_RANGE)
    published_scan = scan_from_ranges(
        ranges, float(angle_min), float(step), stamp, "base_link", 0.1, SCAN_MAX_RANGE
    )
    metric, _anchored = floor_anchor(metric, floor_depth(INTR, cam, UP_LEVEL), cam.z)
    image = image_from_array(metric, "32FC1", stamp, "camera_optical")
    return bytes(image.data), list(published_scan.ranges)


# ---- building the node ----------------------------------------------------------------------
Build = Callable[..., tuple[DepthStream, FakeNet]]


@pytest.fixture
def build(tmp_path: Path) -> Iterator[Build]:
    """A depth node over the stubs: the config's camera, a scripted network, the lidar's mount
    and a still odometry in TF, the camera's optical edge when asked for, a saved law when
    given (in its own file unless ``law_file`` names one two nodes are to share); every node
    built is closed when the test ends.

    Its config is a copy of config/camera.json with the optics pinned to the nominal pinhole —
    the INTR this file's synthetic room is built with — so the day the robot's camera is
    calibrated does not move the reference."""
    made: list[DepthStream] = []
    nominal_dir = tmp_path / "nominal"
    nominal_dir.mkdir()
    nominal = camera_config(nominal_dir)

    def make(
        camera_edge: Any = None,
        odom: bool = True,
        law: tuple[float, float] | None = None,
        law_file: Path | None = None,
        **params: Any,
    ) -> tuple[DepthStream, FakeNet]:
        law_file = law_file or tmp_path / f"depth_law_{len(made)}.json"
        if law is not None:
            save_law(law_file, law[0], law[1], 500, time.time())
        with ros_stubs.parameters(config=nominal, law_file=str(law_file), **params):
            node = DepthStream()
        made.append(node)
        net = FakeNet()
        node._net = net  # type: ignore[assignment]
        edges = node._tf.buffer.transforms
        edges[("base_link", "laser")] = _transform((0.0, 0.0, LIDAR_Z_M), np.eye(3))
        if odom:
            edges[("odom", "base_link")] = ros_stubs.TransformStamped()
        if camera_edge is not None:
            edges[("base_link", "camera_optical")] = camera_edge
        node.subs["/camera/camera_info"][1](_info(_stamp(0)))
        return node, net

    yield make
    for node in made:
        node.close()


def frame(node: DepthStream, net: FakeNet, cam: CameraPose, wall_x: float, k: int) -> Any:
    """One frame through the node: the network's answer scripted, the lidar's scan of the same
    moment delivered, the image processed. Returns the stamp."""
    stamp = _stamp(k)
    net.frames.append(_network(_scene(cam, wall_x), seed=k))
    node.subs["/scan"][1](_scan(wall_x, stamp))
    node._process(_image(stamp))
    return stamp


def published(node: DepthStream) -> tuple[list[Any], list[Any]]:
    return node.pubs["/camera/depth"].sent, node.pubs["/depth_scan"].sent


# ---- bit for bit ---------------------------------------------------------------------------
def test_the_default_flags_publish_today_s_depth_and_scan_bit_for_bit(build: Build) -> None:
    """Six views of walls at 1-3.5 m through a noisy network, no camera edge in TF (the config
    is the pose, as it was): after every frame the node's law is the reference's law to the
    last bit, it withholds exactly the frames the reference withheld, and every published
    image and scan is byte for byte the reference's."""
    node, net = build(
        range_law=False,
        frame_law=False,
        lidar_sigma_m=0.0,
        fan_floor_gate="off",
        floor_pairs=False,
        parallax_anchor=False,
    )
    # the LIDAR's affine law alone, every beam weighing the same, and the fan's floor gate off:
    # this reference is the chain of before 2026-09-15, and every switch that has moved it since
    # is named here — the floor's pixels and the parallax corners fit the same law now
    assert node._pipeline.switches == {name: FLAGS[name] for name in node._pipeline.names} | {
        "range_law": False,
        "frame_law": False,
        "floor_pairs": False,
        "parallax_anchor": False,
    }, "the flags' defaults are the chain's, bar the four switched here"

    reference = AffineScale()
    withheld = 0
    for k, wall_x in enumerate(WALLS):
        stamp = frame(node, net, CONFIG_CAM, wall_x, k)
        depth32 = _network(_scene(CONFIG_CAM, wall_x), seed=k)
        expected = legacy_process(depth32, _scan(wall_x, stamp), CONFIG_CAM, reference, stamp)
        depths, scans = published(node)
        assert (node._law.a, node._law.b, node._law.pooled) == (
            reference.a,
            reference.b,
            reference.pooled,
        )
        if expected is None:
            withheld += 1
            assert len(depths) == k + 1 - withheld
            continue
        image, scan = depths[-1], scans[-1]
        assert len(depths) == k + 1 - withheld
        assert bytes(image.data) == expected[0] and image.encoding == "32FC1"
        assert image.header.stamp == stamp and image.header.frame_id == "camera_optical"
        assert np.array_equal(np.asarray(scan.ranges), np.asarray(expected[1]), equal_nan=True)
        assert scan.header.frame_id == "base_link" and scan.header.stamp == stamp
        if wall_x < SCAN_MAX_RANGE - 0.5:  # a wall well within the scan's range marks it
            assert np.isfinite(scan.ranges).sum() > 20, "the wall marks the scan"
    assert withheld >= 1 and node._law.fitted
    assert node._law.a == pytest.approx(LAW[0], rel=0.05)
    node._report()
    line = node.logger.texts("info")[-1]
    assert line.startswith("depth: ") and f"{withheld} withheld" in line
    assert "edge_filter on [step 8%]" in line and "affine_law on [a 1." in line
    assert "floor_pairs off" in line and "wall_anchor off" in line and "wall_correct off" in line
    assert (
        f"camera pose from config {len(WALLS)} frames (no TF edge; fan pan from the mount)" in line
    )  # no edge to read a pan off: the fan is folded straight ahead, as the mount looks
    assert "backend fake (CPU model not loaded)" in line
    assert (
        "flags: edge_filter=on lidar_anchor=on floor_pairs=off wall_anchor=off"
        " parallax_anchor=off affine_law=on ray_law=off range_law=off frame_law=off"
        " wall_correct=off"
        " floor_anchor=on"
        " depth_backend=local" in line
    )
    assert "ms median/max: network" in line and "pipeline" in line
    saved = json.loads(node._law_file.read_text())
    assert saved["pooled"] == node._law.pooled and saved["a"] == node._law.a
    assert node._pipeline.stats["affine_law"].frames == 0, "the stage totals are the window's"


def test_one_table_of_defaults_reaches_both_the_node_s_flags_and_the_stages(build: Build) -> None:
    """A default is written once. Every switch and knob of pepin.depth_pipeline's
    PIPELINE_DEFAULTS is the node's flag default and the chain standard_pipeline builds, the
    node hands the knobs to the stages that own them at start, and a live change reaches
    them."""
    for name, value in PIPELINE_DEFAULTS.items():
        assert name in FLAGS, f"{name}: the pipeline's default has no flag of that name"
        assert FLAGS.flag(name).default == value, f"{name}: the flag's default is not the table's"
    chain = standard_pipeline()
    for name, on in chain.switches.items():
        if name in PIPELINE_DEFAULTS:
            assert on is PIPELINE_DEFAULTS[name], f"{name}: the chain is not the table"
    node, _net = build()
    field = node._pipeline.stage("frame_law")
    floor = node._pipeline.stage("floor_pairs")
    assert isinstance(field, FrameLaw) and isinstance(floor, FloorPairs)
    assert field.field.grid == grid_of(str(PIPELINE_DEFAULTS["field_grid"]))
    assert field.field.prior == PIPELINE_DEFAULTS["field_prior"]
    assert field.field.carry == PIPELINE_DEFAULTS["field_carry"]
    assert field.field.carry_tau_s == PIPELINE_DEFAULTS["field_carry_tau_s"]
    assert floor.sigma_pitch_deg == PIPELINE_DEFAULTS["floor_sigma_pitch_deg"]
    assert floor.normal_tol_deg == PIPELINE_DEFAULTS["floor_normal_tol_deg"]
    node._switches.set("field_grid", "1x1")  # the old single law, live
    node._switches.set("field_prior", 7.5)
    node._switches.set("floor_normal_tol_deg", 2.0)
    assert field.field.grid == (1, 1) and field.field.prior == 7.5
    assert floor.normal_tol_deg == 2.0


# ---- the camera pose from TF ---------------------------------------------------------------
def test_the_camera_pose_is_tf_s_at_the_frame_s_stamp_and_the_config_only_without_it(
    build: Build,
) -> None:
    """The head sits at pitch 31.5 deg (the neck's live edge) while config/camera.json says
    26: the node projects, scans and anchors with TF's pose — the published frame is the
    reference's with that pose, and not the reference's with the config's — and the report
    line no longer counts a config fallback. A head panned 20 deg is counted as such."""
    edge = _optical_edge(31.5)
    node, net = build(
        camera_edge=edge,
        law=LAW,
        range_law=False,
        frame_law=False,
        fan_floor_gate="off",
        floor_pairs=False,
        parallax_anchor=False,
    )  # the gate off and the two picture-fed anchors with it: this test is about WHICH pose the
    # chain uses, and the reference beside it is the lidar's law alone
    on_neck = pose_from_transform(edge)
    tf_cam = CameraPose.from_optical(on_neck.rotation, on_neck.translation)
    assert tf_cam.pitch == pytest.approx(math.radians(31.5)) and tf_cam.z == 1.2
    assert tf_cam != CONFIG_CAM
    stamp = frame(node, net, tf_cam, 2.0, 0)
    depths, scans = published(node)
    assert len(depths) == 1, "a seeded law publishes at once"
    depth32 = _network(_scene(tf_cam, 2.0), seed=0)
    seeded = AffineScale()
    seeded.seed(*LAW)
    tf_pan = optical_heading(on_neck.rotation)[1]  # a head pointing straight, to the last bit
    with_tf = legacy_process(depth32, _scan(2.0, stamp), tf_cam, seeded, stamp, pan=tf_pan)
    assert with_tf is not None
    assert bytes(depths[0].data) == with_tf[0]
    assert np.array_equal(np.asarray(scans[0].ranges), np.asarray(with_tf[1]), equal_nan=True)
    seeded_config = AffineScale()
    seeded_config.seed(*LAW)
    with_config = legacy_process(depth32, _scan(2.0, stamp), CONFIG_CAM, seeded_config, stamp)
    assert with_config is not None and bytes(depths[0].data) != with_config[0]
    node._report()
    line = node.logger.texts("info")[-1]
    assert "camera pose from config" not in line and "head panned" not in line
    assert "1 lidar verdicts" in line
    # the head turned: the pose is still taken (pitch kept), the pan counted in the report
    node._tf.buffer.transforms[("base_link", "camera_optical")] = _optical_edge(31.5, pan_deg=20.0)
    turned, edge = node._camera_at(_stamp(1))
    assert turned.pitch == pytest.approx(math.radians(31.5)) and turned.z == 1.2
    # the pose drops the pan, the edge beside it keeps it: the parallax anchor triangulates
    # against the edge, and its baseline would point 20 deg wrong without it
    assert edge is not None
    assert optical_heading(edge.rotation)[1] == pytest.approx(math.radians(20.0))
    frame(node, net, turned, 2.0, 1)  # the second count: the frame's own lookup
    node._report()
    assert "head panned 2 frames (projected with the pan)" in node.logger.texts("info")[-1]
    node._switches.set("scan_honours_pan", False)  # the old fan, and the report line says so
    frame(node, net, turned, 2.0, 2)
    node._report()
    assert "head panned 1 frames (projected as if not)" in node.logger.texts("info")[-1]


def test_without_a_camera_edge_the_config_pose_stands_in_and_is_counted(build: Build) -> None:
    node, _net = build()
    assert node._camera_at(_stamp(0)) == (CONFIG_CAM, None)
    assert node._tally.take().counts["camera_from_config"] == 1
    assert "camera pose from TF" in node.logger.texts("info")[-1]
    assert (
        f"{math.degrees(CONFIG_CAM.pitch):.1f} deg while TF has no edge"
        in (node.logger.texts("info")[-1])
    )


# ---- the flags and the stages ----------------------------------------------------------------
def test_a_flag_switches_its_stage_and_a_launch_override_reaches_it(build: Build) -> None:
    node, _net = build(floor_pairs=True)
    assert node._pipeline.on("floor_pairs"), "the override reached the stage"
    assert node.set_parameters([Param("floor_anchor", False)])[0].successful
    assert not node._pipeline.on("floor_anchor") and not node._switches.on("floor_anchor")
    assert node.set_parameters([Param("wall_correct", True), Param("edge_filter", False)])[
        0
    ].successful
    assert node._pipeline.on("wall_correct") and not node._pipeline.on("edge_filter")
    refused = node.set_parameters([Param("no_such_stage", True)])[0]
    assert not refused.successful and "not a flag of this node" in refused.reason
    assert node.set_parameters([Param("depth_backend", "remote")])[0].successful
    assert node._net.mode == "remote"
    node._report()
    line = node.logger.texts("info")[-1]
    assert "floor_anchor off [tolerance 4 cm]" in line and "wall_correct on [" in line
    assert "flags: edge_filter=off lidar_anchor=on floor_pairs=on" in line


def test_the_two_rulers_weights_are_live_flags_and_the_report_prints_them(build: Build) -> None:
    """What a beam is trusted to (lidar_sigma_m) and how loudly the corners vote
    (parallax_weight) are live parameters that reach their stages, and both stages print what
    they are set to in the report line."""
    node, net = build()
    beams, corners = node._pipeline.stage("lidar_anchor"), node._pipeline.stage("parallax_anchor")
    assert isinstance(beams, LidarAnchor) and isinstance(corners, ParallaxAnchor)
    assert beams.sigma_m == LIDAR_SIGMA_M == 0.0, "a beam ships weighing a flat 1"
    assert corners.weight == PARALLAX_WEIGHT
    assert node.set_parameters([Param("lidar_sigma_m", 0.015), Param("parallax_weight", 0.0)])[
        0
    ].successful
    assert beams.sigma_m == 0.015 and corners.weight == 0.0
    assert node.set_parameters([Param("lidar_sigma_m", 0.03), Param("parallax_weight", 2.0)])[
        0
    ].successful
    assert beams.sigma_m == 0.03 and corners.weight == 2.0
    for k, wall_x in enumerate(WALLS):
        frame(node, net, CONFIG_CAM, wall_x, k)
    node._report()
    line = node.logger.texts("info")[-1]
    assert "lidar_anchor on [weight 1 / sigma^2, sigma 3.0 cm]" in line
    assert "weight 2 / sigma^2" in line, "the parallax anchor says what its pairs vote with"
    assert "lidar_sigma_m=0.03 " in line and "parallax_weight=2.0 " in line
    assert "rulers: lidar " in line, "the frame law says whose weight fitted it"


def test_a_parallax_corner_is_a_track_and_the_knobs_reach_the_stage(build: Build) -> None:
    """The shape of a parallax measurement is live: how many frames a corner must be seen in,
    how far back the window reaches and over how many views, how much parallax those views must
    add up to, what the sigma is taken from and how far a track's two halves may disagree. All
    six reach the anchor, the report line says which shape is running, and
    parallax_track_min_obs 2 puts the old pair back without a restart."""
    node, _net = build()
    corners = node._pipeline.stage("parallax_anchor")
    assert isinstance(corners, ParallaxAnchor)
    assert corners.track_min_obs == 3 and corners.tracking, "a corner ships as a track"
    assert corners.track_max_views == 8 and corners.sigma_model == "covariance"
    assert corners.split_tol_sigma == 0.0, "the split gate ships off: measured, it buys nothing"
    line = corners.describe()
    assert ">= 3 obs over <= 8 views, asks 10 cm total" in line
    assert "sigma from the covariance, halves unchecked" in line
    assert node.set_parameters(
        [
            Param("parallax_track_min_obs", 5),
            Param("parallax_track_window_s", 0.8),
            Param("parallax_min_total_baseline_m", 0.2),
            Param("parallax_track_max_views", 4),
            Param("parallax_sigma_model", "baseline"),
            Param("parallax_split_tol_sigma", 3.0),
        ]
    )[0].successful
    assert corners.track_min_obs == 5 and corners.track_window_s == 0.8
    assert corners.min_total_baseline_m == 0.2 and corners.window_s == 0.8
    assert corners.track_max_views == 4 and corners.sigma_model == "baseline"
    assert corners.split_tol_sigma == 3.0
    assert "over <= 4 views" in corners.describe()
    assert "halves within 3 sigma" in corners.describe()
    refused = node.set_parameters([Param("parallax_sigma_model", "guess")])[0]
    assert not refused.successful, "a sigma model nobody implements is not a silent default"
    assert node.set_parameters([Param("parallax_track_min_obs", 2)])[0].successful
    assert corners.tracking is False and corners.window_s == corners.max_gap_s
    node._report()
    line = node.logger.texts("info")[-1]
    assert "parallax_track_min_obs=2 " in line and "parallax_track_window_s=0.8 " in line
    assert "parallax_track_max_views=4 " in line and "parallax_sigma_model=baseline " in line
    assert "parallax_split_tol_sigma=3.0 " in line


def test_the_scan_is_built_from_the_depth_before_the_floor_anchor(
    build: Build, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By default the law's output; with the wall correction on, the corrected depth — the
    floor anchor never touches what stops the cart."""
    node, net = build(law=LAW)
    seen: dict[str, Any] = {}
    run = node._pipeline.run

    def spy_run(depth: Any, ctx: Any) -> Any:
        seen["result"] = run(depth, ctx)
        return seen["result"]

    as_scan = node._as_scan

    def spy_scan(depth: Any, image: Any, ctx: Any) -> Any:
        seen["scan_depth"] = depth
        return as_scan(depth, image, ctx)

    monkeypatch.setattr(node._pipeline, "run", spy_run)
    monkeypatch.setattr(node, "_as_scan", spy_scan)
    frame(node, net, CONFIG_CAM, 2.0, 0)
    assert seen["scan_depth"] is seen["result"].after["frame_law"]
    node.set_parameters([Param("wall_correct", True)])
    frame(node, net, CONFIG_CAM, 2.0, 1)
    assert seen["scan_depth"] is seen["result"].after["wall_correct"]
    assert seen["result"].verdict("wall_correct").pixels > 0
    assert len(published(node)[0]) == 2


def test_a_scan_the_odometry_cannot_carry_passes_as_it_is_and_is_counted(build: Build) -> None:
    node, net = build(odom=False, law=LAW)
    frame(node, net, CONFIG_CAM, 2.0, 0)
    counts = node._tally.take().counts
    assert counts["uncarried"] == 1 and counts["frames"] == 1 and counts["verdicts"] == 1


def test_a_cart_parked_under_the_lidar_s_plane_is_told_so_and_not_asked_about_scan(
    build: Build,
) -> None:
    """The cart parked 0.4 m from a wall — closer than the distance at which the lidar's plane
    enters the picture, so not one beam lands in the image while the lidar answers at full rate.
    The window must name that, not send the reader to /scan: the counter says how many frames
    and from how far the plane shows, and the warning says to back off or tilt the head.
    """
    node, net = build(law=LAW)
    near = plane_in_view_from(INTR, CONFIG_CAM, LIDAR_Z_M)
    assert near is not None and near > 0.5
    for k, wall_x in enumerate((0.4, 0.45)):
        frame(node, net, CONFIG_CAM, wall_x, k)
    counts = node._tally.take().counts
    assert counts["beams_out_of_frame"] == 2 and counts["verdicts"] == 0
    assert counts["beams_too_few"] == 0 and counts["held"] == 2
    for k, wall_x in enumerate((0.4, 0.45)):  # the window the report reads is its own
        frame(node, net, CONFIG_CAM, wall_x, k)
    node._report()
    warning = node.logger.texts("warning")[-1]
    assert "the lidar's plane is out of the picture" in warning
    assert f"past {near:.2f} m ahead" in warning and "is /scan alive?" not in warning
    line = node.logger.texts("info")[-1]
    assert f"lidar plane out of the picture 2 frames (it shows past {near:.2f} m ahead)" in line


def test_without_a_scan_at_all_the_window_still_asks_whether_scan_is_alive(build: Build) -> None:
    """The other blindness: no scan reaches the node, so no beam is even projected — the
    question about /scan is the right one and stays."""
    node, net = build(law=LAW)
    stamp = _stamp(0)
    net.frames.append(_network(_scene(CONFIG_CAM, 2.0), seed=0))
    node._process(_image(stamp))  # no scan delivered
    node._report()
    counts = node._tally.take().counts
    assert counts["beams_out_of_frame"] == 0 and counts["beams_too_few"] == 0
    assert "is /scan alive?" in node.logger.texts("warning")[-1]


def test_the_imu_leans_the_floor_only_while_something_asks_for_the_lean(build: Build) -> None:
    """A base_link reading seeds the lean as is; with both floor stages off and imu_lean off it
    is not read at all; a reading in another frame with no mount switches the floor stages
    off."""
    node, _net = build()
    reading = ros_stubs.Imu(
        header=ros_stubs.Header(stamp=_stamp(0), frame_id="base_link"),
        linear_acceleration=ros_stubs.Vector3(x=0.0, y=0.0, z=9.81),
    )
    node.subs["/imu/data_raw"][1](reading)
    assert node._lean.estimator is not None and node._lean.up == pytest.approx([0.0, 0.0, 1.0])
    assert "lean +0.0/+0.0 deg" in node._lean.report()
    off, _ = build(floor_anchor=False, floor_pairs=False, imu_lean=False)
    off.subs["/imu/data_raw"][1](reading)
    assert off._lean.estimator is None and "lean none" in off._lean.report()
    alien, _ = build(floor_pairs=True)
    alien._lean._mount = None
    reading.header.frame_id = "imu"
    alien.subs["/imu/data_raw"][1](reading)
    assert alien._lean.estimator is None
    assert not alien._switches.on("floor_anchor") and not alien._switches.on("floor_pairs")
    assert not alien._pipeline.on("floor_anchor") and not alien._pipeline.on("floor_pairs")
    assert "the floor stages are off" in alien.logger.texts("error")[-1]


def test_the_fallback_optics_are_the_calibration_when_the_config_carries_one(
    build: Build, tmp_path: Path
) -> None:
    """Before any camera_info arrives the node projects with the camera config's own optics.
    Those must be the checkerboard's when there is one, scaled to the frame, and the nominal
    pinhole of the field of view when there is not — one reader for both (pepin.camera.optics),
    so a calibration written by ros/calibrate.sh needs no second edit here. Both configs are
    built here: neither provenance is read off the committed file."""
    node, _net = build()
    node._intr = None
    frame = SimpleNamespace(width=WIDTH, height=HEIGHT)
    nominal = node._intr_or_nominal(frame)  # type: ignore[arg-type]
    assert nominal == INTR, "no intrinsics block in the config: the nominal pinhole"

    node._camera_cfg = CameraConfig.load(camera_config(tmp_path, CALIBRATION))
    measured = node._intr_or_nominal(frame)  # type: ignore[arg-type]
    assert (measured.fx, measured.fy) == (450.0, 448.0)  # scaled to the published 640x360
    assert (measured.cx, measured.cy) == (323.0, 177.0)
    assert (measured.width, measured.height) == (WIDTH, HEIGHT)
    assert measured != nominal


# ---- the law file: both laws ------------------------------------------------------------------
PITCHES = (10.0, 26.0, 42.0)  # the head at three tilts: the same wall at three elevations, so
# the ray's angle and the depth are no longer one regressor (pepin.elevation.separable 0.26)


def _tilted_run(node: DepthStream, net: FakeNet) -> None:
    """Twelve frames of the walls with the head tilting between PITCHES: a pool whose angles
    an angular law can be told from (one pitch reads separable 1.00 and fits none)."""
    for k, wall_x in enumerate(WALLS * 2):
        pitch = PITCHES[k % len(PITCHES)]
        node._tf.buffer.transforms[("base_link", "camera_optical")] = _optical_edge(pitch)
        frame(node, net, CameraPose(0.05, 0.0, 1.2, math.radians(pitch)), wall_x, k)


def test_a_seeded_law_publishes_at_once_with_the_ray_law_on(build: Build) -> None:
    """The saved law seeds every law of the chain, not only the affine one: with ``ray_law``
    on, the first frame goes out. The ray law pools and fits on its own, so left unseeded it
    withheld the whole warm-up while the start line said "publishing at once"."""
    node, net = build(law=LAW, ray_law=True, wall_anchor=True)
    assert "publishing at once" in node.logger.texts("info")[0]
    assert node._ray.ready and not node._ray.ray_ready, "the affine seed, no angular law yet"
    frame(node, net, CONFIG_CAM, 2.0, 0)
    depths, scans = published(node)
    assert len(depths) == 1 and len(scans) == 1, "the seeded chain withholds nothing"
    assert node._pipeline.stats["ray_law"].frames == 1


def test_both_laws_go_through_the_file_from_one_run_to_the_next(
    build: Build, tmp_path: Path
) -> None:
    """A run that fits an angular law writes it beside the affine one, and the next start
    restores it and applies it before any live pool."""
    path = tmp_path / "shared_law.json"
    # the wall's pairs are the angular law's source here; the floor's would be a second one,
    # and the frames below are meant to carry no angle of their own at all
    first, net = build(law_file=path, ray_law=True, wall_anchor=True, floor_pairs=False)
    _tilted_run(first, net)
    assert first._ray.ray_fitted and first._ray.gain is not None
    first._report()
    saved = json.loads(path.read_text())
    assert saved["version"] == LAW_VERSION and saved["ray"] == first._ray.gain.state()
    second, _net = build(law_file=path, ray_law=True, wall_anchor=True, floor_pairs=False)
    assert second._ray.ray_ready and not second._ray.ray_fitted, "restored, not refitted"
    assert second._ray.gain is not None and second._ray.gain.state() == saved["ray"]
    assert "ray law ray deg" in second.logger.texts("info")[0]
    for k, wall_x in enumerate(WALLS[:3]):  # frames whose angles carry no angular law of their own
        frame(second, _net, CONFIG_CAM, wall_x, k)
    assert not second._ray.ray_fitted and "(seed)" in second._pipeline.report()
    second._report()
    assert json.loads(path.read_text())["ray"] == saved["ray"], "the seed is kept, not erased"


def test_a_law_file_without_a_ray_record_leaves_the_affine_law_alone(build: Build) -> None:
    """The common case — a file written by a run with the stage off, or by an older build:
    the affine law is seeded, the ray law waits for its own pool, and nothing is invented."""
    node, net = build(law=LAW, ray_law=True, wall_anchor=True)
    assert "no ray law saved" in node.logger.texts("info")[0]
    assert node._ray.gain is None and node._ray.saved_state() is None
    frame(node, net, CONFIG_CAM, 2.0, 0)
    node._report()
    assert "ray" not in json.loads(node._law_file.read_text()), "nothing is invented"


def test_the_imu_lean_flag_places_the_frame_and_carries_the_scan_with_the_body(
    build: Build,
) -> None:
    """With every floor stage off, imu_lean alone is enough to read the IMU, and it is the
    poser that the reading reaches: the flag turns the lean on for the frame\'s pose and turns
    the gyro on inside the estimator; off again, both go back."""
    node, _net = build(floor_anchor=False, imu_lean=True)
    assert node._poser.apply_lean and node._lean.use_gyro
    node.subs["/imu/data_raw"][1](
        ros_stubs.Imu(
            header=ros_stubs.Header(stamp=_stamp(0), frame_id="base_link"),
            linear_acceleration=ros_stubs.Vector3(x=0.0, y=0.0, z=9.81),
        )
    )
    assert node._lean.estimator is not None  # read with no floor stage on at all
    at = stamp_seconds(_stamp(0))
    assert node._poser.lean_at(at) is not None
    node._switches.set("imu_lean", False)
    assert not node._poser.apply_lean and not node._lean.use_gyro
    assert node._poser.lean_at(at) is None


class FakePoser:
    """The frame poser with the odometry replaced: every carry moves the cart by ``step`` metres
    along x, whatever the stamps. The camera's pose is still the real one's."""

    def __init__(self, real: Any, step: float) -> None:
        self._real = real
        self._step = step

    def camera_in_base(self, stamp: float) -> Any:
        return self._real.camera_in_base(stamp)

    def motion(self, from_stamp: float, to_stamp: float) -> Any:
        return SimpleNamespace(rotation=np.eye(3), translation=np.array([self._step, 0.0, 0.0]))


def test_a_carry_the_cart_could_not_have_driven_throws_the_frame_s_beams_away(
    build: Build,
) -> None:
    """2026-09-14: the runaway EKF moved the scan one to two metres over the 25 ms between the
    scan and the frame, the beams landed on the wrong pixels and the law was refitted from
    them (a 1.65 -> 2.05). The frame still publishes; it just judges nothing."""
    node, net = build(law=LAW)
    node._poser = FakePoser(node._poser, 0.05)  # type: ignore[assignment]
    frame(node, net, CONFIG_CAM, 2.0, 0)
    counts = node._tally.take().counts
    assert counts["carry_insane"] == 1 and counts["verdicts"] == 0, "no pair from this frame"
    assert counts["frames"] == 1 and counts["uncarried"] == 0, "the depth still went out"


def test_a_carry_within_the_cart_s_speed_anchors_the_law_as_before(build: Build) -> None:
    node, net = build(law=LAW)
    node._poser = FakePoser(node._poser, 0.0002)  # type: ignore[assignment]
    frame(node, net, CONFIG_CAM, 2.0, 0)
    counts = node._tally.take().counts
    assert counts["carry_insane"] == 0 and counts["verdicts"] == 1


def test_the_insane_carry_is_named_in_the_report_line(build: Build) -> None:
    node, net = build(law=LAW)
    node._poser = FakePoser(node._poser, 0.05)  # type: ignore[assignment]
    frame(node, net, CONFIG_CAM, 2.0, 0)
    node._report()
    line = node.logger.texts("info")[-1]
    assert "carry insane 1 frames (the odometry ran away)" in line
    assert "carry_max_speed_mps=1.0" in line


def test_the_range_law_ships_live_and_goes_through_the_file(build: Build, tmp_path: Path) -> None:
    """The node's live law is the one that follows the range: it fits on the same pooled beams,
    says so in the report line and in the flags, is written beside the affine numbers, and the
    next start applies it before any live pool."""
    path = tmp_path / "range_law.json"
    first, net = build(law=LAW, law_file=path)
    for k, wall_x in enumerate(WALLS * 2):
        frame(first, net, CONFIG_CAM, wall_x, k)
    stage = first._range
    assert stage.fitted and stage.law is not None and stage.law.centres.size >= 2
    first._report()
    line = first.logger.texts("info")[-1]
    assert "range_law on [D" in line and "range_law=on" in line
    saved = json.loads(path.read_text())
    assert saved["range"] == stage.law.state() and saved["a"] == first._law.a
    second, _net = build(law_file=path)
    assert second._range.law is not None and second._range.law.state() == stage.law.state()
    assert second._range.ready and not second._range.fitted, "a seed until the live pool answers"
    assert "range law D" in second.logger.texts("info")[0]


def test_the_shipped_floor_gate_raises_the_fan_s_band_and_says_so(build: Build) -> None:
    """The default is no longer the flat 0.15 m edge: with ``fan_floor_gate`` band the fan marks
    from the floor's own noise upward (pepin.contact.fan_min_z), so a bearing the flat edge
    marked on a noisy floor can come back clear, and ``off`` still reproduces the old fan
    exactly. The chain is otherwise the reference's."""
    gated, net_gated = build(range_law=False, frame_law=False, lidar_sigma_m=0.0)
    assert str(gated._switches["fan_floor_gate"]) == "band", "the shipped gate"
    plain, net_plain = build(
        range_law=False, frame_law=False, lidar_sigma_m=0.0, fan_floor_gate="off"
    )
    for k, wall_x in enumerate(WALLS):
        frame(gated, net_gated, CONFIG_CAM, wall_x, k)
        frame(plain, net_plain, CONFIG_CAM, wall_x, k)
    _d, gated_scans = published(gated)
    _p, plain_scans = published(plain)
    assert len(gated_scans) == len(plain_scans) and gated_scans
    marked_gated = sum(np.count_nonzero(np.isfinite(np.asarray(s.ranges))) for s in gated_scans)
    marked_plain = sum(np.count_nonzero(np.isfinite(np.asarray(s.ranges))) for s in plain_scans)
    assert marked_gated <= marked_plain  # a gate removes marks, it never invents them


# ---- the fan and the neck's pan ---------------------------------------------------------------
def _post(range_m: float = 2.0) -> np.ndarray:
    """A depth image of one narrow post straight ahead of the LENS at ``range_m`` and nothing
    else (NaN): whatever the neck does, this post sits on the camera's optical axis, so where
    the published fan marks it is the whole answer about the pan. Its rows sit below the
    principal point, which puts the post 0.9-1.2 m above the floor — inside the fan's band."""
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    u, v = int(INTR.cx), int(INTR.cy)
    depth[v + 4 : v + 52, u - 2 : u + 3] = range_m
    return depth


def _marked_deg(scan: Any) -> np.ndarray:
    """The bearings, in degrees, a LaserScan actually marks: the bins holding a finite range."""
    r = np.asarray(scan.ranges, dtype=float)
    angles = scan.angle_min + scan.angle_increment * np.arange(r.size)
    return np.degrees(angles[np.isfinite(r)])


def _fan_of(node: DepthStream, pan_deg: float) -> Any:
    """The published fan of :func:`_post` with the neck's edge at ``pan_deg``, through the
    node's own ``_camera_at`` — so the pan reaches the fan the way a live frame's does."""
    node._tf.buffer.transforms[("base_link", "camera_optical")] = _optical_edge(0.0, pan_deg)
    cam, cam_optical = node._camera_at(_stamp(0))
    ctx = FrameContext(INTR, cam, cam_optical=cam_optical)
    return node._as_scan(_post(), _image(_stamp(0)), ctx)


def test_the_fan_s_bearings_turn_with_the_neck_s_pan(build: Build) -> None:
    """A post on the optical axis marks bearing 0 with the head straight, +10 deg with the head
    turned 10 deg left and -25 with it turned right: the fan's bearings are the cart's, not the
    camera's. The window turns with them — angle_min is pan - 40 deg — and the frame stays
    base_link, which is what Nav2's obstacle layer is handed."""
    node, _net = build(fan_floor_gate="off")  # this test is about bearings, not the floor
    for pan_deg in (0.0, 10.0, -25.0):
        scan = _fan_of(node, pan_deg)
        marked = _marked_deg(scan)
        assert marked.size, f"the post marks the fan at pan {pan_deg}"
        # the post is 5 px wide, so it fills a bin or two; the 5 cm the lens sits ahead of
        # base_link's origin shifts its bearing by under a quarter of a degree at 2 m
        assert float(np.mean(marked)) == pytest.approx(pan_deg, abs=0.6)
        assert math.degrees(scan.angle_min) == pytest.approx(pan_deg - 40.0, abs=1e-6)
        assert math.degrees(scan.angle_max) == pytest.approx(pan_deg + 40.0, abs=1e-6)
        assert scan.header.frame_id == "base_link"


def test_scan_honours_pan_off_folds_the_fan_as_if_the_head_looked_ahead(build: Build) -> None:
    """The old projection is one live parameter away: with ``scan_honours_pan`` off the same
    post marks bearing 0 whatever the neck does, and the fan's window sits on base_link's x —
    which is the costmap of before 2026-09-15, wrong by the whole pan."""
    node, _net = build(fan_floor_gate="off")
    assert node.set_parameters([Param("scan_honours_pan", False)])[0].successful
    assert not node._switches.on("scan_honours_pan"), "the flag reaches the node live"
    for pan_deg in (0.0, 10.0, -25.0):
        scan = _fan_of(node, pan_deg)
        assert float(np.mean(_marked_deg(scan))) == pytest.approx(0.0, abs=0.6)
        assert math.degrees(scan.angle_min) == pytest.approx(-40.0, abs=1e-6)
    node.set_parameters([Param("scan_honours_pan", True)])  # and back, without a restart
    assert float(np.mean(_marked_deg(_fan_of(node, 10.0)))) == pytest.approx(10.0, abs=0.6)
