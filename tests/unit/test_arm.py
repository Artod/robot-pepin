"""The arm's self-filter: the SO-101's links posed by its joints and cut out of the volume.

The chain is checked against the URDF by homogeneous matrices written out here; the scenes are
ray-marched step by step against the links as fitted (margin 0) with a containment test of this
file's own — independent of the slab test and the culling the filter uses — and the camera is
posed by the neck's model from config/neck.json, the way the robot places it. The claims: a frame
looking at the arm, in any of four poses, paints nothing inside it while the floor beside it is
painted, and changes no voxel whose ray does not pass through the arm; a ray that enters the arm
carves nothing behind it; a surface painted before the arm moved in, and a lidar return off the
arm, are forgotten; the column a stall look asks over the arm holds none of it.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.arm import (
    ArmMask,
    ArmModel,
    ArmPose,
    JointHistory,
    arm_pose,
    parse_urdf,
)
from pepin.body import OrientedBox, oriented_ray_depth
from pepin.camera import OPTICAL_RPY
from pepin.depth import Intrinsics
from pepin.mounts import rotation_from_rpy
from pepin.neck import NeckAngles, NeckConfig, camera_pose
from pepin.tsdf import DepthLaw, GridSpec, RigidPose
from pepin.volume_scan import column_points, column_window
from pepin.worldmap import PlanarMount, WorldMap

REPO = Path(__file__).resolve().parents[2]
NECK = NeckConfig.from_json(REPO / "config/neck.json")
SHIPPED_DATA = json.loads((REPO / "config/arm.json").read_text())
SHIPPED = ArmModel.from_dict(SHIPPED_DATA)
# The arm of these tests stands where it can be seen and is never the placeholder of the file:
# the top shelf's front, 10 cm right of centre, pointing forward.
MOUNT = {"x_m": 0.0, "y_m": -0.10, "z_m": 0.78, "yaw_deg": 0.0, "measured": True}
MODEL = ArmModel.from_dict({**SHIPPED_DATA, "mount": MOUNT})
SOLID = dataclasses.replace(MODEL, margin_m=0.0)  # the links as fitted: the scene's geometry
INTR = Intrinsics(fx=40.0, fy=40.0, cx=39.5, cy=29.5, width=80, height=60)  # ~90 deg, small
EYE = Intrinsics(fx=494.6, fy=494.6, cx=399.5, cy=299.5, width=800, height=600)
PARKED = {name: math.degrees(v) for name, v in (SHIPPED.pose or {}).items()}
# Four poses, degrees (URDF convention), each with a head pose (pan, tilt) that sees it.
POSES: dict[str, tuple[dict[str, float], tuple[float, float]]] = {
    "parked": (PARKED, (-30.0, 60.0)),
    "zero": (dict.fromkeys(PARKED, 0.0), (-20.0, 50.0)),
    "reaching down": (
        {**dict.fromkeys(PARKED, 0.0), "shoulder_lift": 60.0, "elbow_flex": 10.0},
        (-15.0, 55.0),
    ),
    "panned right": (
        {**dict.fromkeys(PARKED, 0.0), "shoulder_pan": 70.0, "shoulder_lift": 30.0},
        (-50.0, 50.0),
    ),
}


def radians(pose_deg: dict[str, float]) -> dict[str, float]:
    return {name: math.radians(v) for name, v in pose_deg.items()}


def pose_of(pose_deg: dict[str, float]) -> ArmPose:
    """An :class:`ArmPose` from the file (the joints in the model's order)."""
    angles = radians(pose_deg)
    return ArmPose(tuple((n, angles[n]) for n in MODEL.joint_names), "config")


def camera(pan_deg: float, pitch_deg: float) -> RigidPose:
    """``base_link <- camera_optical`` at these neck angles, through the neck's model."""
    x, y, z, _roll, pitch, yaw = camera_pose(
        NECK, NeckAngles(math.radians(pan_deg), math.radians(pitch_deg))
    )
    rotation = rotation_from_rpy(0.0, pitch, yaw) @ rotation_from_rpy(*OPTICAL_RPY)
    return RigidPose(rotation, np.array([x, y, z]))


def within(boxes: tuple[OrientedBox, ...], points: np.ndarray) -> np.ndarray:
    """Which (n, 3) points lie in one of ``boxes``: this file's own test, not the module's."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    hit = np.zeros(len(p), dtype=bool)
    for box in boxes:
        local = np.einsum("ij,ni->nj", box.rotation, p - box.centre)
        hit |= np.all(np.abs(local) <= box.half, axis=1)
    return hit


def march(intr: Intrinsics, pose: RigidPose, boxes: tuple[OrientedBox, ...]) -> np.ndarray:
    """The optical depth each pixel sees of ``boxes`` on a floor at z = 0, by marching the ray in
    1 cm steps (NaN where nothing within 2.5 m)."""
    v, u = np.mgrid[0 : intr.height, 0 : intr.width]
    rays = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones(u.shape)], -1)
    direction = rays.reshape(-1, 3) @ pose.rotation.T
    depth = np.full(direction.shape[0], np.nan)
    for z in np.arange(0.05, 2.5, 0.01):
        open_ = np.isnan(depth)
        if not open_.any():
            break
        points = pose.translation + z * direction[open_]
        hit = within(boxes, points) | (points[:, 2] <= 0.0)
        depth[np.flatnonzero(open_)[hit]] = z
    return depth.reshape(intr.height, intr.width).astype(np.float32)


def room() -> GridSpec:
    """A 5 cm grid around the cart, floor to above the head."""
    return GridSpec(origin=(-1.0, -1.2, -0.1), shape=(44, 48, 30), range_max_m=4.0)


def voxel_centres(spec: GridSpec) -> np.ndarray:
    """Every voxel's centre, (n, 3) in the grid's C order (the order of ``weight.ravel``)."""
    index = np.stack(np.meshgrid(*(np.arange(n) for n in spec.shape), indexing="ij"), axis=-1)
    return (index.reshape(-1, 3) + 0.5) * spec.voxel_m + np.array(spec.origin)


def homogeneous(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation_from_rpy(*rpy)
    out[:3, 3] = xyz
    return out


def about_z(q: float) -> np.ndarray:
    out = np.eye(4)
    out[:2, :2] = [[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]]
    return out


# ---- the model ---------------------------------------------------------------------------------
def test_the_vendored_urdf_is_the_so101_chain_with_its_limits() -> None:
    """Six revolute joints with the names lerobot's so101_follower reads (one fixed frame
    beside them), every axis z, the limits upstream wrote."""
    joints = {
        j.name: j
        for j in parse_urdf((REPO / "src/pepin/vendor/so101/so101_new_calib.urdf").read_text())
    }
    assert set(MODEL.joint_names) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
    assert joints["gripper_frame_joint"].kind == "fixed"
    assert all(np.allclose(joints[n].axis, [0, 0, 1]) for n in MODEL.joint_names)
    assert joints["elbow_flex"].upper == pytest.approx(1.69)
    assert joints["shoulder_pan"].parent == "base_link"
    assert MODEL.chain.root == "base_link"


def test_forward_kinematics_is_the_product_of_the_urdf_s_transforms() -> None:
    """The gripper's frame by homogeneous matrices written out from the URDF's numbers, against
    the module's chain, at an arbitrary pose; the mount is the outermost factor."""
    q = {"shoulder_pan": 0.3, "shoulder_lift": -0.7, "elbow_flex": 1.1, "wrist_flex": -0.4}
    chain = [
        ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159), "shoulder_pan"),
        ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0), "shoulder_lift"),
        ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708), "elbow_flex"),
        ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708), "wrist_flex"),
    ]
    expected = homogeneous((0.0, -0.10, 0.78), (0.0, 0.0, 0.0))
    for xyz, rpy, name in chain:
        expected = expected @ homogeneous(xyz, rpy) @ about_z(q[name])
    angles = {**dict.fromkeys(MODEL.joint_names, 0.0), **q}
    wrist = MODEL.link_poses(angles)["wrist_link"]
    assert np.allclose(wrist.translation, expected[:3, 3], atol=1e-4)
    assert np.allclose(wrist.rotation, expected[:3, :3], atol=1e-4)


def test_the_shipped_arm_loads_with_the_parked_pose_read_from_the_encoders() -> None:
    """The file's joints are the encoders' parked pose, 11 boxes on the 7 moving links, a 3 cm
    margin, and a mount measured (fitted to the head's depth) on the top basket's right front."""
    assert SHIPPED.source == "config" and SHIPPED.topic == "/arm/joint_states"
    assert PARKED["shoulder_lift"] == pytest.approx(-102.4)
    assert PARKED["elbow_flex"] == pytest.approx(96.8)
    assert len(SHIPPED.links) == 11 and SHIPPED.margin_m == 0.03 and SHIPPED.stride_px == 4
    assert {link.link for link in SHIPPED.links} == {j.child for j in SHIPPED.chain.joints} - {
        "gripper_frame_link"
    } | {"base_link"}
    assert SHIPPED.mount_measured
    x, y, z = SHIPPED.mount.translation
    assert -0.30 < x < 0.027 and y < 0.0 and 0.70 < z < 0.85


def test_the_parked_arm_stands_folded_on_its_base() -> None:
    """The sign check the encoders allow without a move: with lerobot's signs the folded arm's
    links stay above its base's bottom face (boxes are hulls, so within 3 cm) and within 20 cm
    of the pan axis; with shoulder_lift and elbow_flex flipped it would reach under the base."""
    here = {**MOUNT, "z_m": 0.0, "x_m": 0.0, "y_m": 0.0}
    arm = dataclasses.replace(ArmModel.from_dict({**SHIPPED_DATA, "mount": here}), margin_m=0.0)
    parked = {**radians(PARKED), "shoulder_pan": 0.0}
    corners = np.vstack([b.corners() for b in arm.boxes_at(parked)])
    assert corners[:, 2].min() > -0.03
    assert np.hypot(corners[:, 0] - 0.0388, corners[:, 1]).max() < 0.20
    flipped = {**parked, "shoulder_lift": -parked["shoulder_lift"]}
    flipped["elbow_flex"] = -parked["elbow_flex"]
    assert np.vstack([b.corners() for b in arm.boxes_at(flipped)])[:, 2].min() < -0.10


def test_a_config_that_is_not_an_arm_is_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="a mount"):
        ArmModel.from_dict({"links": []})
    with pytest.raises(ValueError, match="not in the URDF"):
        ArmModel.from_dict({**SHIPPED_DATA, "links": [{**SHIPPED_DATA["links"][0], "link": "x"}]})
    with pytest.raises(ValueError, match="none of"):
        ArmModel.from_dict({**SHIPPED_DATA, "joints": {"source": "dream"}})
    with pytest.raises(ValueError, match=r"needs joints\.pose_deg"):
        ArmModel.from_dict({**SHIPPED_DATA, "joints": {"source": "config"}})
    with pytest.raises(ValueError, match="positive half"):
        box = {**SHIPPED_DATA["links"][0], "half_m": [0.01, 0.0, 0.01]}
        ArmModel.from_dict({**SHIPPED_DATA, "links": [box]})


def test_the_joints_come_from_the_topic_near_the_stamp_and_from_the_file_otherwise() -> None:
    """source topic: the sample nearest the stamp within max_age_s; none, a stale one or one that
    lacks a joint gives the file's pose, named stale. source config: the file, always."""
    topic = dataclasses.replace(MODEL, source="topic")
    history = JointHistory()
    names = list(MODEL.joint_names)
    silent = arm_pose(topic, history, 10.0)
    assert silent is not None and silent.source == "stale"
    assert silent.angles == arm_pose(topic, None, 10.0).angles  # type: ignore[union-attr]
    history.add(10.0, names, [0.1] * len(names))
    history.add(10.2, names, [0.2] * len(names))
    pose = arm_pose(topic, history, 10.15)
    assert pose is not None and pose.source == "topic" and pose.gap_s == pytest.approx(0.05)
    assert pose.mapping["elbow_flex"] == pytest.approx(0.2)
    stale = arm_pose(topic, history, 11.0)
    assert stale is not None and stale.source == "stale"
    assert stale.mapping == pytest.approx(MODEL.pose)
    history.add(12.0, names[:2], [0.3, 0.3])  # a message that names two joints only
    assert arm_pose(topic, history, 12.0).source == "stale"  # type: ignore[union-attr]
    assert arm_pose(MODEL, history, 10.2).source == "config"  # type: ignore[union-attr]
    assert history.newest() == 12.0 and history.heard == 3
    for t in np.arange(13.0, 16.0, 0.1):
        history.add(float(t), names, [0.0] * len(names))
    assert history.nearest(10.0)[0] > 13.0  # type: ignore[index]  # two seconds are kept


# ---- the rays ------------------------------------------------------------------------------------
def test_an_oriented_box_is_entered_where_the_geometry_says() -> None:
    """A 0.1 m cube turned 45 degrees about z, 1 m below a camera looking straight down: the
    centre ray meets its top at 0.95; a ray landing at x 0.06 on the top enters the turned cube
    (its corner reaches 0.0707) and misses the unturned one (its side is at 0.05); a box around
    the lens is skipped and named; the culled grid equals the whole-picture test on every ray."""
    down = RigidPose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.0]))  # optical z = -z
    unturned = OrientedBox("cube", np.zeros(3), np.eye(3), np.full(3, 0.05))
    turned = dataclasses.replace(unturned, rotation=rotation_from_rpy(0.0, 0.0, math.pi / 4))
    intr = Intrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=11, height=11)
    rays = oriented_ray_depth((turned,), intr, down, 1)
    assert rays.z[5, 5] == pytest.approx(0.95)
    aside = np.array([[0.06 / 0.95, 0.0, 1.0]]) @ down.rotation.T
    assert turned.entry(down.translation, aside)[0] == pytest.approx(0.95)
    assert np.isinf(unturned.entry(down.translation, aside)[0])
    around = OrientedBox("around", np.array([0.0, 0.0, 1.0]), np.eye(3), np.full(3, 0.2))
    both = oriented_ray_depth((turned, around), intr, down, 1)
    assert both.skipped == ("around",) and both.z[5, 5] == pytest.approx(0.95)
    v, u = np.mgrid[0:11, 0:11]
    d = np.stack([(u - 5.0) / 10.0, (v - 5.0) / 10.0, np.ones(u.shape)], -1).reshape(-1, 3)
    whole = turned.entry(down.translation, d @ down.rotation.T).reshape(11, 11)
    assert np.array_equal(np.isfinite(whole), np.isfinite(rays.z)) and np.isinf(rays.z[0, 0])
    assert np.allclose(whole[np.isfinite(whole)], rays.z[np.isfinite(rays.z)], atol=1e-6)


@pytest.mark.slow  # four scenes marched ray by ray
def test_the_ray_grid_matches_the_marched_arm_in_every_pose() -> None:
    """Per pose, where the marched scene (margin 0, 1 cm steps) says a pixel sees the arm, the
    mask's grid says its ray enters the grown arm no deeper than that; the culling drops no ray
    the whole-picture test would have hit."""
    for name, (pose_deg, head) in POSES.items():
        pose, cam = pose_of(pose_deg), camera(*head)
        boxes = MODEL.boxes_at(pose.mapping)
        grid = oriented_ray_depth(boxes, INTR, cam, 1)
        assert 0.01 < grid.share < 0.9, f"{name}: the arm is in view, not all of it"
        v, u = np.mgrid[0 : INTR.height, 0 : INTR.width]
        pixels = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones(u.shape)], -1)
        rays = pixels.reshape(-1, 3) @ cam.rotation.T
        brute = np.min([b.entry(cam.translation, rays) for b in boxes], axis=0)
        unseen = -1.0  # inf == inf compares, but allclose wants finite numbers
        assert np.allclose(
            np.where(np.isinf(brute), unseen, brute).reshape(grid.z.shape),
            np.where(np.isinf(grid.z), unseen, grid.z),
            atol=1e-5,
        ), f"{name}: the culling dropped a ray"
        seen = march(INTR, cam, SOLID.boxes_at(pose.mapping))
        hit_z = cam.translation[2] + seen.reshape(-1) * rays[:, 2]
        on_arm = (np.isfinite(seen.reshape(-1)) & (hit_z > 0.02)).reshape(seen.shape)
        assert on_arm.sum() > 10, f"{name}: the march sees the arm"
        assert np.all(grid.z[on_arm] <= seen[on_arm] + 0.011), name


# ---- the volume --------------------------------------------------------------------------------
def fused(pose_deg: dict[str, float], head: tuple[float, float], filtered: bool) -> WorldMap:
    """One frame of the marched scene (arm as fitted on the floor) integrated into a fresh
    volume at the cart's origin, with the arm's clip and forget or without."""
    pose, cam = pose_of(pose_deg), camera(*head)
    depth = march(INTR, cam, SOLID.boxes_at(pose.mapping))
    world = WorldMap(room(), PlanarMount(), protect_lidar_layer=False)
    mask = ArmMask(dataclasses.replace(MODEL, stride_px=1))
    clip = mask.for_frame(INTR, cam, pose) if filtered else None
    world.integrate_depth(depth, None, INTR, cam, clip=clip)
    if filtered:
        world.forget(mask.boxes(pose))
    return world


@pytest.mark.slow  # four scenes marched ray by ray, two integrations each
def test_a_frame_looking_at_the_arm_paints_nothing_of_it_in_any_pose() -> None:
    """Parked, at zero, reaching down and panned right: unfiltered the frame paints the arm into
    the volume; filtered, not one voxel inside the grown links holds weight, no surface point
    lies on the arm, the floor beside it is painted — and every voxel the filter changed lies
    inside the grown arm or behind it along its own ray."""
    centres = voxel_centres(room())
    for name, (pose_deg, head) in POSES.items():
        angles = radians(pose_deg)
        grown, solid = MODEL.boxes_at(angles), SOLID.boxes_at(angles)
        raw, cut = fused(pose_deg, head, False), fused(pose_deg, head, True)
        shape = raw.volume.weight.shape
        in_solid = within(solid, centres).reshape(shape)
        in_grown = within(grown, centres).reshape(shape)
        assert np.count_nonzero(raw.volume.weight[in_solid]) > 3, f"{name}: unfiltered, painted"
        assert np.count_nonzero(cut.volume.weight[in_grown]) == 0, f"{name}: filtered, nothing"
        points, _ = cut.volume.surface(min_weight=0.1)
        assert not within(solid, points).any(), f"{name}: no surface on the arm"
        assert np.count_nonzero(points[:, 2] < 0.1) > 20, f"{name}: the floor is still there"
        # A pixel on the arm measures no room, so the free space in front of the arm along it is
        # not carved either (the body's rule): what the filter may change is the voxels on a
        # line of sight that meets the arm, before, inside or behind it. Each voxel reads its
        # NEAREST pixel's ray (up to half a pixel, 1.3 cm a metre here), hence 2 cm of slack.
        changed = (raw.volume.weight != cut.volume.weight) | (raw.volume.sdf != cut.volume.sdf)
        moved = centres[changed.ravel()]
        lens = camera(*head).translation
        slack = tuple(box.grown(0.02) for box in grown)
        sight = np.zeros(len(moved), dtype=bool)
        for t in np.linspace(0.0, 3.0, 240)[1:]:
            sight |= within(slack, lens + t * (moved - lens))
        assert len(moved) > 0 and sight.all(), f"{name}: the filter touched only the arm"


@pytest.mark.slow  # the occlusion is marched voxel by voxel
def test_a_ray_that_enters_the_arm_carves_nothing_behind_it() -> None:
    """A dark frame (every pixel NaN) carves its rays to the reach; an obstacle the reaching arm
    hides from the camera survives it, while unfiltered the rays carve through the arm."""
    pose_deg, head = POSES["reaching down"]
    pose, cam = pose_of(pose_deg), camera(*head)
    dark = np.full((INTR.height, INTR.width), np.nan, dtype=np.float32)
    law = DepthLaw(no_depth_free=True, no_depth_weight=1.0, reach_m=3.0)
    clip = ArmMask(dataclasses.replace(MODEL, stride_px=1)).for_frame(INTR, cam, pose)
    assert clip is not None

    def painted() -> WorldMap:
        world = WorldMap(room(), PlanarMount(), protect_lidar_layer=False)
        world.volume.sdf[:] = -0.5
        world.volume.weight[:] = 10.0
        return world

    centres = voxel_centres(room())
    solid, grown = SOLID.boxes_at(pose.mapping), MODEL.boxes_at(pose.mapping)
    hidden = np.zeros(len(centres), dtype=bool)
    for t in np.linspace(0.0, 1.0, 120)[1:-1]:
        hidden |= within(solid, cam.translation + t * (centres - cam.translation))
    hidden &= ~within(grown, centres)
    raw, cut = painted(), painted()
    raw.integrate_depth(dark, None, INTR, cam, law=law)
    cut.integrate_depth(dark, None, INTR, cam, law=law, clip=clip)
    shape = raw.volume.sdf.shape
    hidden = hidden.reshape(shape)
    assert np.count_nonzero((raw.volume.sdf != -0.5) & hidden) > 5, "unfiltered: carved through"
    assert np.count_nonzero((cut.volume.sdf != -0.5) & hidden) == 0, "filtered: nothing behind"


def test_what_was_painted_where_the_arm_now_stands_is_forgotten_and_nothing_else() -> None:
    """A wall painted by the camera and the lidar, then the arm reaches into it: every voxel
    inside a grown link goes back to unobserved on every channel (field, weight, colour, the
    lidar's weight, the views), the rest of the wall is untouched, and a column over the arm, as
    the stall look asks it, holds no point of the arm."""
    world = WorldMap(room(), PlanarMount(), protect_lidar_layer=False)
    centres = voxel_centres(room())
    shape = world.volume.weight.shape
    wall = ((centres[:, 0] > 0.1) & (centres[:, 0] < 0.5)).reshape(shape)
    world.volume.sdf[wall], world.volume.weight[wall] = -0.2, 8.0
    world.volume.colour_weight[wall], world.lidar_weight[wall], world.views[wall] = 1.0, 8.0, 2.0
    world.volume.sdf[~wall], world.volume.weight[~wall] = 1.0, 8.0  # free and seen around it
    angles = radians(POSES["reaching down"][0])
    grown = MODEL.boxes_at(angles)
    in_arm = within(grown, centres).reshape(shape)
    assert np.count_nonzero(in_arm & wall) > 5, "the arm reaches into the wall"
    before = world.volume.weight.copy()
    forgotten = world.forget(grown)
    assert forgotten == np.count_nonzero(in_arm)
    for channel in (
        world.volume.weight,
        world.volume.colour_weight,
        world.lidar_weight,
        world.views,
    ):
        assert not channel[in_arm].any()
    assert np.all(world.volume.sdf[in_arm] == 1.0)
    assert np.array_equal(world.volume.weight[~in_arm], before[~in_arm]), "nothing else moved"
    corners = np.vstack([b.corners() for b in grown])
    lo, hi = corners.min(axis=0) - 0.1, corners.max(axis=0) + 0.1
    window = column_window(world.volume, world.lidar_weight, lo, hi)
    assert window is not None
    column = column_points(*window, lo, hi, 2.0)
    assert len(column.points) > 0, "the wall around the arm is still in the column"
    assert not within(SOLID.boxes_at(angles), column.points).any(), "and none of the arm"


def test_a_lidar_return_off_the_arm_is_forgotten() -> None:
    """A ranging plane through the reaching arm's lowest link (the lidar's, or a whisker's, at
    whatever height the arm comes down to): the beam that hits the link marks a surface inside
    it, and forgetting the arm takes it back out."""
    grown = MODEL.boxes_at(radians(POSES["reaching down"][0]))
    lowest = min(grown, key=lambda b: b.centre[2])
    world = WorldMap(room(), PlanarMount(z_m=float(lowest.centre[2])), protect_lidar_layer=False)
    # one beam straight at the link's centre, at its bearing and range
    bearing = math.atan2(lowest.centre[1], lowest.centre[0])
    reach = float(np.hypot(lowest.centre[0], lowest.centre[1]))
    world.integrate_scan(np.array([bearing]), np.array([reach]), RigidPose(np.eye(3), np.zeros(3)))
    centres = voxel_centres(room())
    in_arm = within(grown, centres).reshape(world.volume.weight.shape)
    assert world.lidar_weight[in_arm].any(), "the return is painted inside the arm"
    world.forget(grown)
    assert not world.lidar_weight[in_arm].any() and not world.volume.weight[in_arm].any()


# ---- the mask's cache and cost -------------------------------------------------------------------
def test_the_mask_is_rebuilt_only_when_the_joints_the_head_or_the_model_move() -> None:
    mask = ArmMask(MODEL)
    pose_deg, head = POSES["zero"]
    pose, cam = pose_of(pose_deg), camera(*head)
    assert mask.for_frame(EYE, cam, pose) is not None
    mask.for_frame(EYE, cam, pose_of({**pose_deg, "elbow_flex": 0.1}))
    assert mask.rebuilds == 1, "a tenth of a degree is the same arm"
    mask.for_frame(EYE, cam, pose_of({**pose_deg, "elbow_flex": 1.0}))
    assert mask.rebuilds == 2, "a degree of elbow is a new grid"
    mask.for_frame(EYE, camera(head[0] + 1.0, head[1]), pose_of({**pose_deg, "elbow_flex": 1.0}))
    assert mask.rebuilds == 3, "a degree of pan is a new grid"
    mask.model = MODEL
    mask.for_frame(EYE, camera(head[0] + 1.0, head[1]), pose_of({**pose_deg, "elbow_flex": 1.0}))
    assert mask.rebuilds == 3, "the same model is no change"
    mask.model = None
    assert mask.for_frame(EYE, cam, pose) is None and mask.boxes(pose) == ()


def test_at_the_working_tilt_looking_ahead_the_parked_arm_is_not_in_the_picture() -> None:
    """Straight ahead at 23.8 deg down the frame's bottom edge passes over the parked arm at
    this file's mount: the mask answers None and the integrator pays nothing."""
    assert ArmMask(MODEL).for_frame(EYE, camera(0.0, 23.8), pose_of(PARKED)) is None
