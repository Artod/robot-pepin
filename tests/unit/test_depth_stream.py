"""The depth node under the ROS stubs: the pipeline and the frame poser wired in.

rclpy is faked (``ros_stubs``), the network is a test's scripted frames, TF is the stub's
buffer; the node is built and its frames processed here as on the laptop. The proof that
matters: with the flags' defaults, bar the switches that have moved since and are named one by
one at the build, the published depth and scan are, to the bit, what the node's own chain
(edges -> beam pairs -> law -> drop edges -> scan -> floor anchor, as ``_process``
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
import stereo_scenes as scenes

RCLPY = ros_stubs.install()

from camera_configs import CALIBRATION, camera_config  # noqa: E402
from pepin_bringup.depth_stream import (  # noqa: E402
    CARRY_WAIT_S,
    DEPTH_REACH_M,
    FLAGS,
    SCAN_RANGE_M,
    DepthStream,
)
from pepin_bringup.msgs import (  # noqa: E402
    array_from_image,
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
    WallAnchor,
    grid_of,
    standard_pipeline,
)
from pepin.mounts import load_lidar_mount, rotation_from_rpy  # noqa: E402
from pepin.stereo_depth import StereoMatcher  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAMERA_JSON = str(REPO / "config" / "camera.json")
WIDTH, HEIGHT = 640, 360  # what camera_stream publishes at scale 0.5
# The mount and the nominal field of view: the two numbers a calibration never rewrites (it
# writes an intrinsics block beside them). The nodes under test are given a config of their own,
# with the optics pinned to this nominal pinhole — see the build fixture.
CONFIG = CameraConfig.load(CAMERA_JSON, name="overview")
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


def _transform(
    translation: tuple[float, float, float], rotation: np.ndarray, stamp: Any = None
) -> Any:
    """One TF edge, stamped at the first frame's moment unless a test says otherwise: a live
    route republishes its edges, and the node's guard reads that stamp to tell a route that
    still runs from one that has died (``tf_dead_s``)."""
    x, y, z = translation
    qx, qy, qz, qw = quaternion_from_matrix(rotation)
    return ros_stubs.TransformStamped(
        header=ros_stubs.Header(stamp=stamp or _stamp(0)),
        transform=ros_stubs.Transform(
            translation=ros_stubs.Vector3(x=x, y=y, z=z),
            rotation=ros_stubs.Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
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
            edges[("odom", "base_link")] = _transform((0.0, 0.0, 0.0), np.eye(3))
        if camera_edge is not None:
            edges[("base_link", "camera_optical")] = camera_edge
        node.subs["/camera/camera_info"][1](_info(_stamp(0)))
        return node, net

    yield make
    for node in made:
        node.close()


def frame(node: DepthStream, net: FakeNet, cam: CameraPose, wall_x: float, k: int) -> Any:
    """One frame through the node: the network's answer scripted, the lidar's scan of the same
    moment delivered, TF's edges republished at this moment as a live route does, the image
    processed. Returns the stamp."""
    stamp = _stamp(k)
    for edge in node._tf.buffer.transforms.values():
        edge.header.stamp = stamp
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
        floor_pairs=False,
        wall_anchor=False,
        parallax_anchor=False,
        lidar_sigma_m=0.0,
        fan_floor_gate="off",
        depth_reach=False,
    )
    # the affine law alone on the beams alone, every beam weighing the same, the fan's floor gate
    # off and the camera's reach not yet gated: this reference is the chain of before 2026-09-15,
    # and every switch that has moved it since is named here — the floor's, the wall's and the
    # parallax's pairs all ship on today, and so does depth_reach (2026-09-19)
    assert node._pipeline.switches == {name: FLAGS[name] for name in node._pipeline.names} | {
        "range_law": False,
        "frame_law": False,
        "floor_pairs": False,
        "wall_anchor": False,
        "parallax_anchor": False,
    }, "the flags' defaults are the chain's, bar the five switched here"

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
        " parallax_anchor=off affine_law=on range_law=off frame_law=off"
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
    assert field.pairs_cap == PIPELINE_DEFAULTS["field_pairs_cap"]
    assert floor.sigma_pitch_deg == PIPELINE_DEFAULTS["floor_sigma_pitch_deg"]
    assert floor.normal_tol_deg == PIPELINE_DEFAULTS["floor_normal_tol_deg"]
    node._switches.set("field_grid", "1x1")  # the old single law, live
    node._switches.set("field_prior", 7.5)
    node._switches.set("field_pairs_cap", 0)  # the uncapped fit of before, live
    node._switches.set("floor_normal_tol_deg", 2.0)
    assert field.field.grid == (1, 1) and field.field.prior == 7.5
    assert field.pairs_cap == 0 and floor.normal_tol_deg == 2.0


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
        floor_pairs=False,
        parallax_anchor=False,
        fan_floor_gate="off",
        depth_reach=False,
    )  # the gate off, and the floor's and the parallax's pairs off (on by default since
    # 2026-09-16, and they would move the seeded law off the reference's): this test is about
    # WHICH pose the chain uses, not about the fan's floor nor about the rulers of the law
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


# ---- a TF route that has died ---------------------------------------------------------------
class StoppedTf:
    """TF whose route died ``age_s`` before the first frame: nothing covers a frame's stamp,
    the newest edge of every pair is that old, and the blocking ask — the one the guard exists
    to refuse — sleeps ``wait_s`` and counts itself in ``waits``."""

    def __init__(self, age_s: float, wait_s: float = CARRY_WAIT_S) -> None:
        self.stamp = stamp_seconds(_stamp(0)) - age_s
        self.wait_s = wait_s
        self.waits = 0

    def pose_at(self, stamp: float, frame: str, fixed: str) -> Any:
        self.waits += 1
        time.sleep(self.wait_s)
        return None

    def pose_at_nowait(self, stamp: float, frame: str, fixed: str) -> Any:
        return None

    def latest_pose(self, frame: str, fixed: str) -> Any:
        return RigidPose(np.eye(3), np.zeros(3)), self.stamp


def stopped_tf(node: DepthStream, age_s: float, wait_s: float = CARRY_WAIT_S) -> StoppedTf:
    """Put a dead TF route under both of the node's guards — the one that may wait (the camera
    pose, the scan's carry) and the one the pipeline asks per view — so every lookup of a
    frame's path meets the stopped route."""
    fake = StoppedTf(age_s, wait_s)
    node._history.history = fake
    node._frame_history.history = fake
    return fake


def test_a_dead_neck_edge_gives_the_config_pose_at_once_instead_of_waiting(build: Build) -> None:
    """2026-09-16: the board's TF route died, base_link <- camera_optical stopped 344 s back,
    and every frame still spent CARRY_WAIT_S on a lookup no publisher was going to answer
    (pose 212/226 ms, 0.9-3 frames/s). An edge older than tf_dead_s is dead: the config mount
    at once, counted as such, and no wait at all."""
    node, _net = build()
    fake = stopped_tf(node, age_s=10.0)
    started = time.perf_counter()
    assert node._camera_at(_stamp(0)) == (CONFIG_CAM, None)
    spent_ms = (time.perf_counter() - started) * 1e3
    assert fake.waits == 0, "no lookup waited for an edge that has stopped"
    assert spent_ms < 1.0, f"the dead route cost the frame {spent_ms:.1f} ms"
    counts = node._tally.take()
    assert counts.counts["neck_edge_dead"] == 1 and counts.counts["camera_from_config"] == 1
    assert counts.samples["neck_edge_stale_s"] == pytest.approx([10.0])


def test_an_edge_younger_than_tf_dead_s_is_still_waited_for(build: Build) -> None:
    """The live case the wait exists for: the neck publishes, its newest edge is behind the
    frame's stamp and does not cover it yet. That wait stays — the guard refuses only the
    lookups that cannot be answered. Both ways round the newest-edge shortcut: an edge half a
    second old with ``camera_tf_latest`` off (the old ask at the exact stamp), and one two
    seconds old, which that shortcut refuses (CAMERA_TF_MAX_AGE_S) and the wait then takes."""
    old_ask, _net = build(camera_tf_latest=False)
    fake = stopped_tf(old_ask, age_s=0.5, wait_s=0.0)
    assert old_ask._camera_at(_stamp(0)) == (CONFIG_CAM, None)
    assert fake.waits == 1, "a young edge is still waited for at the frame's stamp"
    assert old_ask._tally.take().counts["neck_edge_dead"] == 0
    node, _net = build()
    older = stopped_tf(node, age_s=2.0, wait_s=0.0)
    assert node._camera_at(_stamp(0)) == (CONFIG_CAM, None)
    assert older.waits == 1 and node._tally.take().counts["neck_edge_dead"] == 0


def test_tf_dead_s_zero_waits_on_a_dead_edge_as_the_node_used_to(build: Build) -> None:
    """The flag's off position is the behaviour of before: every lookup waits, however old the
    edge is."""
    node, _net = build(tf_dead_s=0.0)
    fake = stopped_tf(node, age_s=10.0, wait_s=0.0)
    assert node._camera_at(_stamp(0)) == (CONFIG_CAM, None)
    assert fake.waits == 1 and node._tally.take().counts["neck_edge_dead"] == 0


def test_a_dead_odometry_edge_leaves_the_scan_uncarried_instead_of_waiting(build: Build) -> None:
    """The same rule on the lidar's carry: with no odometry edge within tf_dead_s the scan
    passes as it is (counted uncarried) rather than costing the frame two more waits."""
    node, _net = build(law=LAW)
    stamp = _stamp(0)
    node.subs["/scan"][1](_scan(2.0, stamp))
    fake = stopped_tf(node, age_s=10.0)
    started = time.perf_counter()
    points = node._lidar_points(_image(stamp))
    spent_ms = (time.perf_counter() - started) * 1e3
    assert points is not None and fake.waits == 0
    assert spent_ms < 1.0, f"the dead odometry cost the frame {spent_ms:.1f} ms"
    counts = node._tally.take().counts  # two asks: the carry looks the cart up at both stamps
    assert counts["uncarried"] == 1 and counts["odom_edge_dead"] == 2


def test_a_live_odometry_edge_still_carries_the_scan(build: Build) -> None:
    """The guard judges the edge, not the lookup: with TF republishing, the carry runs as it
    always did and nothing is counted dead."""
    node, net = build(law=LAW)
    frame(node, net, CONFIG_CAM, 2.0, 0)
    counts = node._tally.take().counts
    assert counts["uncarried"] == 0 and counts["odom_edge_dead"] == 0
    assert counts["verdicts"] == 1


def test_no_lookup_of_a_frame_s_path_waits_while_tf_is_dead(build: Build) -> None:
    """The whole frame path, the parallax anchor included: with the route dead, not one of the
    node's lookups — the camera pose, the carry, the parallax's motion between two views —
    pays a wait. The frames still publish on the seeded law."""
    node, net = build(law=LAW, parallax_anchor=True)
    fake = stopped_tf(node, age_s=10.0)
    for k in range(3):
        frame(node, net, CONFIG_CAM, 2.0, k)
    assert fake.waits == 0, "a dead route was waited for somewhere on the frame's path"
    assert len(published(node)[0]) == 3


def test_the_pipeline_asks_its_motion_of_a_tf_that_cannot_wait(
    build: Build, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two posers over one TF: the camera pose and the scan's carry may spend CARRY_WAIT_S on a
    stamp the buffer does not cover yet, the pipeline's may spend nothing. The parallax anchor
    asks once per stored view, so a wait there is paid per view — 834 ms of one frame offline —
    and the ask that cannot be answered simply loses that view."""
    node, net = build(law=LAW)
    assert node._history.history.timeout_s == CARRY_WAIT_S
    assert node._frame_history.history.timeout_s == 0.0
    seen: dict[str, Any] = {}
    run = node._pipeline.run

    def spy_run(depth: Any, ctx: Any) -> Any:
        seen["ctx"] = ctx
        return run(depth, ctx)

    monkeypatch.setattr(node._pipeline, "run", spy_run)
    frame(node, net, CONFIG_CAM, 2.0, 0)
    assert seen["ctx"].motion is node._frame_poser
    assert node._frame_poser is not node._poser
    node._switches.set("lean_min_quality", 0.75)  # a live flag reaches both posers
    node._switches.set("imu_lean", False)
    assert node._poser.min_lean_quality == 0.75 and node._frame_poser.min_lean_quality == 0.75
    assert not node._poser.apply_lean and not node._frame_poser.apply_lean


def test_the_report_line_names_a_dead_neck_edge_and_how_stale_it_is(build: Build) -> None:
    """A dead route reads differently from an edge TF never had: the report says which, how
    many frames it cost and how far behind the edge is."""
    node, net = build(law=LAW)
    stopped_tf(node, age_s=10.0, wait_s=0.0)
    frame(node, net, CONFIG_CAM, 2.0, 0)
    node._report()
    line = node.logger.texts("info")[-1]
    assert "camera pose from config 1 frames (neck edge dead 1 frames (10 s stale)" in line
    assert "no TF edge" not in line
    assert "scans uncarried 1" in line and "odom edge dead 2 asks (10 s stale)" in line
    assert "tf_dead_s=3.0" in line


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
    wall = node._pipeline.stage("wall_correct")
    assert isinstance(wall, WallAnchor)
    wall.min_walk_m = 0.0  # this room's 3 % of network noise trips the slope gate within 10 cm
    # of the beams and the climb gate would then refuse every column; what is under test here is
    # WHERE the published scan is taken from, not how far a wall is walked
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
    # both floor stages off: floor_pairs reads the up vector too and ships on since 2026-09-16
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


# ---- the law file: a record of a law that no longer exists --------------------------------------
def test_a_law_file_that_still_carries_a_retired_law_is_read_and_says_so(
    build: Build, tmp_path: Path
) -> None:
    """A file written by an older build carries the ray law's record beside the affine numbers.
    That law is gone (2026-09-15): the file still reads and seeds the affine law, the start
    line names the record it ignores, and the next save writes the file without it."""
    path = tmp_path / "old_law.json"
    save_law(path, LAW[0], LAW[1], 500, time.time())
    record = json.loads(path.read_text())
    record["ray"] = {"alpha": [1.0, 0.1], "beta": 0.0, "lo": -0.3, "hi": 0.2, "pairs": 900}
    path.write_text(json.dumps(record))
    node, net = build(law_file=path)
    start = node.logger.texts("info")[0]
    assert "publishing at once" in start and "ignoring the retired ray law record" in start
    assert node._law.ready and (node._law.a, node._law.b) == LAW
    for k, wall_x in enumerate(WALLS * 2):
        frame(node, net, CONFIG_CAM, wall_x, k)
    assert node._law.fitted
    node._report()
    saved = json.loads(path.read_text())
    assert "ray" not in saved and saved["version"] == LAW_VERSION, "the next save drops it"


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


# ---- the camera answers for its own depth ---------------------------------------------------
def test_the_reach_is_one_number_for_the_image_and_for_the_scan() -> None:
    """The published depth's reach and /depth_scan's cap are the same physical claim, so they are
    the same constant: two literals would drift, and the costmap's obstacle_max_range of 2.5 m has
    to stay under both."""
    assert DEPTH_REACH_M == 3.0
    assert FLAGS["depth_reach_m"] == 3.0
    assert FLAGS["depth_reach"] is True
    assert SCAN_MAX_RANGE == DEPTH_REACH_M, "this file's own reference uses the node's default"


def test_the_published_depth_is_nan_past_the_reach_and_the_scan_is_untouched(
    build: Build,
) -> None:
    """Two nodes, the same seeded law and the same frame of a wall 3.5 m off: the one with
    ``depth_reach`` on publishes NaN exactly where the other published more than 3 m, and nothing
    else moves — same finite pixels to the bit, same /depth_scan to the bit."""
    # Every ruler off and the law seeded: the two nodes then publish the SAME pixels bit for bit,
    # because a law that still moves moves at so much per SECOND of wall time (``law_slew``) and
    # two nodes never see the same wall time.
    frozen = dict(
        law=LAW,
        fan_floor_gate="off",
        lidar_anchor=False,
        range_law=False,
        frame_law=False,
        floor_pairs=False,
        wall_anchor=False,
        parallax_anchor=False,
    )
    off, off_net = build(**frozen)
    on, on_net = build(**frozen)
    assert off.set_parameters([Param("depth_reach", False)])[0].successful
    assert off._switches.on("depth_reach") is False and on._switches.on("depth_reach") is True
    frame(off, off_net, CONFIG_CAM, 3.5, 0)
    frame(on, on_net, CONFIG_CAM, 3.5, 0)
    raw = np.asarray(array_from_image(published(off)[0][0]), dtype=float)
    gated = np.asarray(array_from_image(published(on)[0][0]), dtype=float)
    assert raw.shape == gated.shape == (HEIGHT, WIDTH)
    beyond = raw > 3.0
    assert beyond.any() and (~beyond & np.isfinite(raw)).any(), "the frame has both halves"
    assert np.all(np.isnan(gated[beyond])), "past 3 m the camera says nothing"
    assert np.array_equal(gated[~beyond], raw[~beyond], equal_nan=True), "and nothing else moves"
    assert np.array_equal(
        np.asarray(published(off)[1][0].ranges),
        np.asarray(published(on)[1][0].ranges),
        equal_nan=True,
    ), "/depth_scan is built before the gate and is unchanged"
    on._report()
    line = on.logger.texts("info")[-1]
    share = int(beyond.sum()) / (HEIGHT * WIDTH) * 100.0
    assert f"published NaN past 3.0 m over {share:.1f}% of the pixels" in line
    assert "depth_reach=on depth_reach_m=3.0" in line
    off._report()
    assert "published NaN past" not in off.logger.texts("info")[-1]


def test_the_reach_moves_live_and_a_nearer_one_says_less(build: Build) -> None:
    """The flag is a number a drive may move without a restart: at 1.5 m the same frame of a wall
    3.5 m off keeps only what stands within 1.5 m, and the report line says over how much of the
    picture."""
    node, net = build(law=LAW, fan_floor_gate="off", lidar_anchor=False, parallax_anchor=False)
    frame(node, net, CONFIG_CAM, 3.5, 0)
    wide = np.asarray(array_from_image(published(node)[0][0]), dtype=float)
    assert node.set_parameters([Param("depth_reach_m", 1.5)])[0].successful
    frame(node, net, CONFIG_CAM, 3.5, 1)
    near = np.asarray(array_from_image(published(node)[0][1]), dtype=float)
    assert np.nanmax(near) <= 1.5
    assert np.count_nonzero(np.isnan(near)) > np.count_nonzero(np.isnan(wide))


# ---- the stereo head as the second source ---------------------------------------------------
BASELINE_M = 0.063  # the module's nominal; the node learns it from the right eye's P[0,3]


class FakeMatcher(StereoMatcher):
    """A matcher whose answer the test writes: the disparities of a known scene, in order, and
    the shapes it was handed so a test can prove the right eye reached it."""

    def __init__(self) -> None:
        super().__init__()
        self.disparities: list[np.ndarray] = []
        self.seen: list[tuple[Any, Any]] = []

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.seen.append((np.asarray(left).shape, np.asarray(right).shape))
        return self.disparities.pop(0)


def _right_info(stamp: Any, baseline_m: float = BASELINE_M) -> Any:
    """The right eye's camera_info: the same rectified pinhole, its baseline in ``P[0, 3]``."""
    p = [0.0] * 12
    p[0], p[5], p[2], p[6] = INTR.fx, INTR.fy, INTR.cx, INTR.cy
    p[3] = -INTR.fx * baseline_m
    return ros_stubs.CameraInfo(
        header=ros_stubs.Header(stamp=stamp, frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        k=list(K),
        p=p,
    )


def _right_image(stamp: Any) -> Any:
    """The right eye as the camera node publishes it: rectified mono8, the left one's stamp."""
    return ros_stubs.Image(
        header=ros_stubs.Header(stamp=stamp, frame_id="camera_optical"),
        height=HEIGHT,
        width=WIDTH,
        encoding="mono8",
        step=WIDTH,
        data=bytes(WIDTH * HEIGHT),
    )


def _disparity(z: np.ndarray, baseline_m: float = BASELINE_M) -> np.ndarray:
    """The disparity a rig of this fx and baseline would measure for a true depth image."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(z) & (z > 0), INTR.fx * baseline_m / z, np.nan).astype(
            np.float32
        )


def rig(node: DepthStream) -> FakeMatcher:
    """Tell the node's stereo source what rig it is on (the right eye's camera_info) and put a
    scripted matcher in front of it; returns the matcher for the test to load."""
    node.subs["/camera/right/camera_info"][1](_right_info(_stamp(0)))
    matcher = FakeMatcher()
    assert node._stereo is not None
    node._stereo.matcher = matcher
    return matcher


def stereo_frame(
    node: DepthStream,
    matcher: FakeMatcher,
    cam: CameraPose,
    wall_x: float,
    k: int,
    right: bool = True,
) -> Any:
    """One stereo frame: the true scene as a disparity for the matcher, the right eye delivered
    (unless ``right`` says otherwise), the scan of the moment, the left picture processed."""
    stamp = _stamp(k)
    for edge in node._tf.buffer.transforms.values():
        edge.header.stamp = stamp
    matcher.disparities.append(_disparity(_scene(cam, wall_x)))
    if right:
        node.subs["/camera/right/image"][1](_right_image(stamp))
    node.subs["/scan"][1](_scan(wall_x, stamp))
    node._process(_image(stamp))
    return stamp


def test_the_network_is_the_default_source_and_subscribes_to_no_right_eye(build: Build) -> None:
    """The head on the robot today is one camera: nothing about this node changes until a launch
    says ``depth_source: stereo``, and it does not sit on topics nobody publishes."""
    node, _net = build()
    assert node._source.name == "network" and node._stereo is None
    assert "/camera/right/image" not in node.subs
    assert "/camera/right/camera_info" not in node.subs
    assert node.declared["law_file"] == "/maps/depth_law.json"
    assert node.declared["depth_source"] == "network"


def test_a_stereo_node_keeps_its_own_law_file_and_seeds_the_identity_law(build: Build) -> None:
    """A stereo depth is metres already. With no law of its own saved it starts from a 1.00
    b +0.000 and publishes at once, instead of withholding until the lidar has pooled enough
    beams to grant permission — and it never reads the mono network's law file, whose a 1.28
    would put every obstacle a quarter too far."""
    node, _net = build(depth_source="stereo")
    assert node.declared["law_file"] == "/maps/depth_law_stereo.json"
    assert node._law.ready and (node._law.a, node._law.b) == (1.0, 0.0)
    assert any("identity law" in text for text in node.logger.texts("info"))
    plain, _net2 = build()
    assert not plain._law.ready, "the mono node still waits for its beams"


def test_a_stereo_head_runs_the_cleaning_stages_and_only_watches_the_lidar_law(
    build: Build,
) -> None:
    """The rig decides the chain with no flag to remember: under stereo the scale-recovering
    stages are off and the affine law watches; the mono node's chain is what it always was; and
    every one of them is still a flag a launch or a person can turn back."""
    node, _net = build(depth_source="stereo")
    switches = node._pipeline.switches
    for off in ("floor_pairs", "wall_anchor", "parallax_anchor", "range_law", "frame_law"):
        assert not switches[off], off
    for on in ("edge_filter", "lidar_anchor", "affine_law", "floor_anchor"):
        assert switches[on], on
    assert node._law.watching
    plain, _net2 = build()
    assert not plain._law.watching
    assert plain._pipeline.switches["range_law"] and plain._pipeline.switches["parallax_anchor"]
    back, _net3 = build(depth_source="stereo", law_watch=False, range_law=True)
    assert not back._law.watching and back._pipeline.switches["range_law"]


def test_the_stereo_source_pairs_the_right_eye_by_its_exact_stamp(build: Build) -> None:
    """Both eyes come out of ONE transport frame with ONE stamp, so the pairing is exact: the
    right eye of this picture's stamp reaches the matcher, and the published depth is the
    metres the rig measured, at the left picture's own stamp and frame."""
    node, _net = build(depth_source="stereo", stereo_reach_m=10.0, lidar_anchor=False)
    matcher = rig(node)
    assert node._source.name == "stereo"
    assert node._stereo is not None and node._stereo.geometry is not None
    assert node._stereo.geometry.baseline_m == pytest.approx(BASELINE_M)
    stamp = stereo_frame(node, matcher, CONFIG_CAM, 2.0, 0)
    assert matcher.seen == [((HEIGHT, WIDTH, 3), (HEIGHT, WIDTH))], "both eyes reached it"
    depths, scans = published(node)
    assert len(depths) == 1 and len(scans) == 1
    assert depths[0].header.stamp == stamp and depths[0].header.frame_id == "camera_optical"
    assert scans[0].header.stamp == stamp and scans[0].header.frame_id == "base_link"
    measured = np.asarray(array_from_image(depths[0]), dtype=float)
    truth = _scene(CONFIG_CAM, 2.0)
    both = np.isfinite(measured) & np.isfinite(truth)
    assert both.sum() > 10_000
    assert float(np.median(np.abs(measured[both] - truth[both]) / truth[both])) < 0.01


def test_the_fan_never_announces_a_range_the_stereo_head_does_not_measure(build: Build) -> None:
    """The costmap clears an ``inf`` bearing out to the scan's own range_max, so under a stereo
    head that number is the rig's reach and not the mono-era 3 m; a rig that reaches further
    than the fan was ever asked to look changes nothing."""
    node, _net = build(depth_source="stereo", lidar_anchor=False)
    matcher = rig(node)
    assert node._stereo is not None
    reach = float(node._stereo.reach)
    assert 0.5 < reach < 3.0
    stereo_frame(node, matcher, CONFIG_CAM, 1.5, 0)
    _depths, scans = published(node)
    assert scans[0].range_max == pytest.approx(reach)
    far, _net2 = build(depth_source="stereo", stereo_reach_m=10.0, lidar_anchor=False)
    rig(far)
    assert far._scan_max_range == pytest.approx(3.0)


def test_a_left_picture_with_no_right_eye_of_its_stamp_is_dropped_and_counted(
    build: Build,
) -> None:
    """A right eye of a NEIGHBOURING stamp is a different exposure: pairing it would measure the
    cart's own motion as disparity. So the frame is dropped, counted, and named in the report
    line — and the wait it cost is there too, because it is paid on the frame path."""
    node, _net = build(depth_source="stereo", stereo_reach_m=10.0, stereo_pair_wait_s=0.0)
    matcher = rig(node)
    node.subs["/camera/right/image"][1](_right_image(_stamp(7)))  # a stamp no picture will have
    stereo_frame(node, matcher, CONFIG_CAM, 2.0, 0, right=False)
    assert published(node)[0] == [] and matcher.disparities, "nothing measured, nothing published"
    matcher.disparities.clear()
    stereo_frame(node, matcher, CONFIG_CAM, 2.0, 1)
    assert len(published(node)[0]) == 1, "the very next paired frame goes out"
    node._report()
    line = node.logger.texts("info")[-1]
    assert "unpaired 1 frames" in line and "right eye waited" in line
    assert "source stereo: 128px/5px 3way" in line and "% valid" in line


def test_a_frame_before_the_rig_describes_itself_is_lost_and_said_so(build: Build) -> None:
    """fx and the baseline arrive on camera_info. Until the RIGHT eye's has, the node cannot
    turn a disparity into a metre, and it says so rather than publishing a guess."""
    node, _net = build(depth_source="stereo")
    assert node._stereo is not None and node._stereo.geometry is None
    matcher = FakeMatcher()
    node._stereo.matcher = matcher
    matcher.disparities.append(_disparity(_scene(CONFIG_CAM, 2.0)))
    node.subs["/camera/right/image"][1](_right_image(_stamp(0)))
    node._process(_image(_stamp(0)))
    assert published(node)[0] == []
    assert any("cannot answer" in text for text in node.logger.texts("warning"))
    node._report()
    assert "rig unknown 1 frames" in node.logger.texts("info")[-1]


def test_the_matcher_s_holes_survive_the_whole_chain_and_the_fan(build: Build) -> None:
    """The network's depth is dense and a matcher's is not: the left band it cannot search, the
    blank wall, the occlusions. Every stage on, every hole must stay a hole — NaN through the
    edge filter, the laws, the floor anchor and out — and /depth_scan must answer NaN for a
    bearing with no finite pixel at all (unknown: a costmap neither marks nor clears it) rather
    than clearing it to the horizon."""
    node, _net = build(depth_source="stereo", stereo_reach_m=10.0, fan_floor_gate="off")
    matcher = rig(node)
    truth = _scene(CONFIG_CAM, 2.0)
    holed = _disparity(truth)
    holed[:, : WIDTH // 4] = np.nan  # the band no right eye reaches into
    holed[:, WIDTH // 4 : WIDTH // 2] = np.nan  # a blank wall the texture gate refused
    matcher.disparities.append(holed)
    node.subs["/camera/right/image"][1](_right_image(_stamp(0)))
    node.subs["/scan"][1](_scan(2.0, _stamp(0)))
    node._process(_image(_stamp(0)))
    depths, scans = published(node)
    assert len(depths) == 1, "a holed depth is still a depth"
    out = np.asarray(array_from_image(depths[0]), dtype=float)
    assert np.all(np.isnan(out[:, : WIDTH // 2])), "a hole must not be filled by any stage"
    assert np.isfinite(out[:, WIDTH // 2 + 4 :]).any(), "and the rest must survive"
    ranges = np.asarray(scans[0].ranges)
    assert np.isnan(ranges).any(), "a bearing with no finite pixel is unknown, not clear"
    assert np.isfinite(ranges).any(), "and the half that was measured marks the fan"


def test_the_real_matcher_measures_a_rendered_room_through_the_whole_node(build: Build) -> None:
    """One pass with nothing faked but the room: a rectified pair rendered from a known depth
    goes in as bgr8 and mono8, the node's own StereoMatcher runs on it, and what comes out of
    /camera/depth is the room's metres — no lidar, no law fitted, no network anywhere. This is
    the wiring test: the eyes reach OpenCV in the right order and the answer is not mirrored,
    which a disparity of the wrong sign would make look like an empty picture."""
    node, _net = build(
        depth_source="stereo", stereo_reach_m=10.0, lidar_anchor=False, fan_floor_gate="off"
    )
    node.subs["/camera/right/camera_info"][1](_right_info(_stamp(0)))
    truth = np.where(np.isfinite(_scene(CONFIG_CAM, 2.0)), _scene(CONFIG_CAM, 2.0), 6.0)
    texture = scenes.noise_texture((HEIGHT, WIDTH), np.random.default_rng(11))
    left, right = scenes.render_pair(truth, texture, fx_b=INTR.fx * BASELINE_M)
    colour = np.repeat(left[:, :, None], 3, axis=2)
    stamp = _stamp(0)
    node.subs["/camera/right/image"][1](image_from_array(right, "mono8", stamp, "camera_optical"))
    node._process(image_from_array(colour, "bgr8", stamp, "camera_optical"))
    depths, _scans = published(node)
    assert len(depths) == 1
    measured = np.asarray(array_from_image(depths[0]), dtype=float)
    both = np.isfinite(measured) & (truth < 5.0)
    assert both.mean() > 0.3, "the real matcher answered for almost nothing"
    assert float(np.median(np.abs(measured[both] - truth[both]) / truth[both])) < 0.03
    assert not np.isfinite(measured[:, :128]).any(), "the band no right eye reaches is unknown"
