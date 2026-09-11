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
from typing import Any

import numpy as np
import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup.depth_stream import FLAGS, SCAN_RANGE_M, DepthStream  # noqa: E402
from pepin_bringup.msgs import image_from_array, pose_from_transform, scan_from_ranges  # noqa: E402

from pepin.camera import (  # noqa: E402
    OPTICAL_RPY,
    CameraConfig,
    camera_info_arrays,
    mount_transform,
)
from pepin.depth import (  # noqa: E402
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
    project,
    quaternion_from_matrix,
    save_law,
    scan_points,
    to_base,
)
from pepin.mounts import rotation_from_rpy  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAMERA_JSON = str(REPO / "config" / "camera.json")
WIDTH, HEIGHT = 640, 360  # what camera_stream publishes at scale 0.5
CONFIG = CameraConfig.load(CAMERA_JSON)
K, _D, _R, _P = camera_info_arrays(WIDTH, HEIGHT, CONFIG.hfov_deg)
INTR = Intrinsics.from_camera_info(K, WIDTH, HEIGHT)
CONFIG_CAM = CameraPose(*mount_transform(CONFIG)[:3], mount_transform(CONFIG)[4])
LIDAR_MOUNT = RigidPose(np.eye(3), np.array([0.0, 0.0, 0.2]))
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
    depth32: np.ndarray, scan: Any, cam: CameraPose, law: AffineScale, stamp: Any
) -> tuple[bytes, list[float]] | None:
    """``DepthStream._process`` before the pipeline, line for line: the edge mask on the raw
    depth, the beams of the scan projected through the static mounts (the cart stood still:
    the carry is the identity), the beam pairs into the law, the law applied, the edges
    dropped, the scan before the floor anchor, the floor anchored, both published; ``None``
    when the law did not exist yet. Returns the published image's bytes and scan's ranges."""
    edge = edge_mask(depth32)
    xy = scan_points(np.asarray(scan.ranges), scan.angle_min, scan.angle_increment, SCAN_RANGE_M)
    samples = project(to_base(xy, LIDAR_MOUNT.rotation, LIDAR_MOUNT.translation), cam, INTR)
    a, b = law.observe(beam_pairs(depth32, samples, edge))
    if not law.ready:
        return None
    metric = apply_affine(depth32, a, b)
    metric, _dropped = drop_edges(metric, edge)
    angle_min, step, ranges = depth_to_scan(metric, INTR, cam, max_range=SCAN_MAX_RANGE)
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
    given; every node built is closed when the test ends."""
    made: list[DepthStream] = []

    def make(
        camera_edge: Any = None,
        odom: bool = True,
        law: tuple[float, float] | None = None,
        **params: Any,
    ) -> tuple[DepthStream, FakeNet]:
        law_file = tmp_path / f"depth_law_{len(made)}.json"
        if law is not None:
            save_law(law_file, law[0], law[1], 500, time.time())
        with ros_stubs.parameters(config=CAMERA_JSON, law_file=str(law_file), **params):
            node = DepthStream()
        made.append(node)
        net = FakeNet()
        node._net = net  # type: ignore[assignment]
        edges = node._tf.buffer.transforms
        edges[("base_link", "laser")] = _transform((0.0, 0.0, 0.2), np.eye(3))
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
    node, net = build()
    assert node._pipeline.switches == {name: FLAGS[name] for name in node._pipeline.names}, (
        "the flags' defaults are the chain's"
    )
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
    assert f"camera pose from config {len(WALLS)} frames (no TF edge)" in line
    assert "backend fake (CPU model not loaded)" in line
    assert (
        "flags: edge_filter=on lidar_anchor=on floor_pairs=off wall_anchor=off affine_law=on"
        " wall_correct=off floor_anchor=on depth_backend=local" in line
    )
    assert "ms median/max: network" in line and "pipeline" in line
    saved = json.loads(node._law_file.read_text())
    assert saved["pooled"] == node._law.pooled and saved["a"] == node._law.a
    assert node._pipeline.stats["affine_law"].frames == 0, "the stage totals are the window's"


# ---- the camera pose from TF ---------------------------------------------------------------
def test_the_camera_pose_is_tf_s_at_the_frame_s_stamp_and_the_config_only_without_it(
    build: Build,
) -> None:
    """The head sits at pitch 31.5 deg (the neck's live edge) while config/camera.json says
    26: the node projects, scans and anchors with TF's pose — the published frame is the
    reference's with that pose, and not the reference's with the config's — and the report
    line no longer counts a config fallback. A head panned 20 deg is counted as such."""
    edge = _optical_edge(31.5)
    node, net = build(camera_edge=edge, law=LAW)
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
    with_tf = legacy_process(depth32, _scan(2.0, stamp), tf_cam, seeded, stamp)
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
    turned = node._camera_at(_stamp(1))
    assert turned.pitch == pytest.approx(math.radians(31.5)) and turned.z == 1.2
    frame(node, net, turned, 2.0, 1)  # the second count: the frame's own lookup
    node._report()
    assert "head panned 2 frames (projected as if not)" in node.logger.texts("info")[-1]


def test_without_a_camera_edge_the_config_pose_stands_in_and_is_counted(build: Build) -> None:
    node, _net = build()
    assert node._camera_at(_stamp(0)) == CONFIG_CAM
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
    assert seen["scan_depth"] is seen["result"].after["affine_law"]
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


def test_the_imu_leans_the_floor_only_while_a_floor_stage_is_on(build: Build) -> None:
    """A base_link reading seeds the tilt as is; with both floor stages off it is not read;
    a reading in another frame with no mount switches the floor stages off."""
    node, _net = build()
    reading = ros_stubs.Imu(
        header=ros_stubs.Header(stamp=_stamp(0), frame_id="base_link"),
        linear_acceleration=ros_stubs.Vector3(x=0.0, y=0.0, z=9.81),
    )
    node.subs["/imu/data_raw"][1](reading)
    assert node._tilt is not None and node._tilt.up == pytest.approx([0.0, 0.0, 1.0])
    off, _ = build(floor_anchor=False)
    off.subs["/imu/data_raw"][1](reading)
    assert off._tilt is None
    alien, _ = build(floor_pairs=True)
    alien._imu_mount = None
    reading.header.frame_id = "imu"
    alien.subs["/imu/data_raw"][1](reading)
    assert alien._tilt is None
    assert not alien._switches.on("floor_anchor") and not alien._switches.on("floor_pairs")
    assert not alien._pipeline.on("floor_anchor") and not alien._pipeline.on("floor_pairs")
    assert "the floor stages are off" in alien.logger.texts("error")[-1]
