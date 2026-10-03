"""The self-filter: the cart's own body cut out of every camera frame the volume integrates.

The scenes are ray-marched here, step by step, against the boxes as measured — independent of the
slab test the filter uses — and the camera is posed by the neck's own model (pepin.neck) from
config/neck.json, the way the robot places it. The claims: a frame looking down at the cart's own
shelf paints nothing inside the body; a ray that enters the body carves nothing behind it; a frame
that never sees the body is written bit for bit as without the filter.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.body import BodyBox, BodyMask, BodyModel
from pepin.camera import OPTICAL_RPY
from pepin.depth import Intrinsics
from pepin.mounts import rotation_from_rpy
from pepin.neck import NeckAngles, NeckConfig, angle_limits, camera_pose, pan_pivot
from pepin.tsdf import DepthLaw, GridSpec, RigidPose, Tsdf

REPO = Path(__file__).resolve().parents[2]
NECK = NeckConfig.from_json(REPO / "config/neck.json")
SHIPPED = BodyModel.load(REPO / "config/body.json")
INTR = Intrinsics(fx=40.0, fy=40.0, cx=39.5, cy=29.5, width=80, height=60)  # ~90 deg, small
MODEL = dataclasses.replace(SHIPPED, stride_px=1)  # every ray its own: exact for the asserts
# The rectified left eye of the stereo head: fx 494.6 px at 800x600 (gaze.md section 0).
EYE = Intrinsics(fx=494.6, fy=494.6, cx=399.5, cy=299.5, width=800, height=600)


def camera(pan_deg: float, pitch_deg: float) -> RigidPose:
    """``base_link <- camera_optical`` at these neck angles, through the neck's model."""
    x, y, z, _roll, pitch, yaw = camera_pose(
        NECK, NeckAngles(math.radians(pan_deg), math.radians(pitch_deg))
    )
    rotation = rotation_from_rpy(0.0, pitch, yaw) @ rotation_from_rpy(*OPTICAL_RPY)
    return RigidPose(rotation, np.array([x, y, z]))


def inside(model: BodyModel, points: np.ndarray) -> np.ndarray:
    """Which (n, 3) base_link points lie in one of the model's grown boxes, faces included."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    hit = np.zeros(p.shape[0], dtype=bool)
    for box in model.grown:
        hit |= np.all((p >= np.array(box.lo)) & (p <= np.array(box.hi)), axis=1)
    return hit


def march(intr: Intrinsics, pose: RigidPose, boxes: tuple[BodyBox, ...]) -> np.ndarray:
    """The optical depth each pixel sees in a room of ``boxes`` on a floor at z = 0, by marching
    the ray in 5 mm steps (NaN where nothing within 3 m)."""
    v, u = np.mgrid[0 : intr.height, 0 : intr.width]
    rays = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones(u.shape)], -1)
    direction = rays.reshape(-1, 3) @ pose.rotation.T
    depth = np.full(direction.shape[0], np.nan)
    solid = BodyModel(boxes, margin_m=0.0)
    for z in np.arange(0.05, 3.0, 0.005):
        open_ = np.isnan(depth)
        if not open_.any():
            break
        points = pose.translation + z * direction[open_]
        hit = inside(solid, points) | (points[:, 2] <= 0.0)
        depth[np.flatnonzero(open_)[hit]] = z
    return depth.reshape(intr.height, intr.width).astype(np.float32)


def room() -> GridSpec:
    """A 5 cm grid around the cart, floor to above the head."""
    return GridSpec(origin=(-1.6, -1.2, -0.1), shape=(56, 48, 30), range_max_m=4.0)


def voxel_centres(spec: GridSpec) -> np.ndarray:
    """Every voxel's centre, (n, 3) in the grid's C order (the order of ``Tsdf.weight.ravel``)."""
    index = np.stack(np.meshgrid(*(np.arange(n) for n in spec.shape), indexing="ij"), axis=-1)
    return (index.reshape(-1, 3) + 0.5) * spec.voxel_m + np.array(spec.origin)


# ---- the model ---------------------------------------------------------------------------------
def test_the_shipped_body_loads_and_the_camera_never_stands_inside_it() -> None:
    """At every pose the neck can reach the lens stays outside every grown box: a margin that
    swallowed the lens would cut every pixel of the frame."""
    assert [box.name for box in SHIPPED.boxes] == ["cart", "top_load", "wheels", "mast"]
    assert SHIPPED.margin_m == 0.05 and SHIPPED.stride_px == 4
    (pan_lo, pan_hi), (tilt_lo, tilt_hi) = angle_limits(NECK)
    for pan in np.linspace(pan_lo, pan_hi, 37):
        for tilt in np.linspace(tilt_lo, tilt_hi, 11):
            pose = camera(math.degrees(pan), math.degrees(tilt))
            assert not inside(SHIPPED, pose.translation[None, :]).any(), (pan, tilt)


def test_a_config_that_is_not_a_body_is_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="a box needs"):
        BodyModel.from_dict({"boxes": [{"name": "x"}]})
    with pytest.raises(ValueError, match="not below"):
        BodyModel.from_dict({"boxes": [{"name": "x", "min_m": [0, 0, 1], "max_m": [1, 1, 0]}]})
    with pytest.raises(ValueError, match="out of range"):
        BodyModel.from_dict({"boxes": [], "ray_stride_px": 0})
    data = json.loads((REPO / "config/body.json").read_text())
    assert BodyModel.from_dict(data) == SHIPPED


def test_a_ray_enters_a_box_where_the_geometry_says_and_a_box_around_the_lens_is_skipped() -> None:
    """A camera 1 m above a 0.5 m box looking straight down: the centre ray meets its top 0.5 m
    out; a ray past the box's edge never does. A box around the lens is named and skipped."""
    down = RigidPose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.0]))  # optical z = -z
    intr = Intrinsics(fx=5.0, fy=5.0, cx=1.0, cy=1.0, width=3, height=3)
    box = BodyBox("block", (-0.05, -0.05, 0.0), (0.05, 0.05, 0.5))
    rays = BodyModel((box,), margin_m=0.0, stride_px=1).ray_depth(intr, down)
    assert rays.z[1, 1] == pytest.approx(0.5)
    assert np.isinf(rays.z[0, 0]), "0.1 m out at the top, 0.2 m at the floor: never inside"
    beside = RigidPose(down.rotation, np.array([0.2, 0.0, 1.0]))  # the ray leans back in
    assert BodyModel((box,), 0.0, 1).ray_depth(intr, beside).z[1, 0] == pytest.approx(0.75), (
        "x 0.2 - 0.2 t meets the side x = 0.05 at t 0.75, z 0.25: in through the side"
    )
    around = BodyBox("around", (-1.0, -1.0, 0.9), (1.0, 1.0, 1.1))
    both = BodyModel((box, around), margin_m=0.0, stride_px=1).ray_depth(intr, down)
    assert both.skipped == ("around",) and both.z[1, 1] == pytest.approx(0.5)
    coarse = BodyModel((box,), margin_m=0.0, stride_px=2).ray_depth(intr, down)
    assert coarse.z.shape == (2, 2), "one ray per 2 x 2 cell"


# ---- the volume --------------------------------------------------------------------------------
def test_a_frame_looking_down_at_the_own_shelf_paints_nothing_inside_the_body() -> None:
    """Reverse-gaze (pan 150, 45 deg down): the cart's top shelf fills part of the frame. Without
    the filter the volume grows a surface inside the cart; with it, not one voxel inside the body
    is written — and the floor behind the cart is still painted."""
    pose = camera(150.0, 45.0)
    depth = march(INTR, pose, SHIPPED.boxes)
    clip = BodyMask(MODEL).for_frame(INTR, pose)
    assert clip is not None and 0.05 < clip.share < 0.95, "the body is in this frame, not all of it"

    raw, cut = Tsdf(room()), Tsdf(room())
    raw.integrate(depth, None, INTR, pose)
    cut.integrate(depth, None, INTR, pose, clip=clip)

    true_body = BodyModel(SHIPPED.boxes, margin_m=0.0)
    within = inside(true_body, voxel_centres(room())).reshape(raw.weight.shape)
    assert np.count_nonzero(raw.weight[within]) > 20, "unfiltered, the frame paints the cart"
    assert np.count_nonzero(cut.weight[within]) == 0, "filtered, nothing inside the body"
    points, _colours = cut.surface(min_weight=0.1)
    assert len(points) and not inside(SHIPPED, points).any(), "no surface on the grown body"
    floor = points[points[:, 2] < 0.1]
    assert len(floor) > 20, "the floor the frame sees around the cart is still there"


@pytest.mark.slow  # the occlusion is marched voxel by voxel, 120 steps each
def test_a_ray_that_enters_the_body_carves_nothing_behind_it() -> None:
    """A dark frame (every pixel NaN, the stereo matched nothing) carves its rays to the reach.
    An obstacle the cart's body hides from the camera must survive it: unfiltered, the rays carve
    straight through the cart into it."""
    pose = camera(150.0, 45.0)
    dark = np.full((INTR.height, INTR.width), np.nan, dtype=np.float32)
    law = DepthLaw(no_depth_free=True, no_depth_weight=1.0, reach_m=3.0)
    clip = BodyMask(MODEL).for_frame(INTR, pose)
    assert clip is not None

    def painted() -> Tsdf:
        volume = Tsdf(room())
        volume.sdf[:] = -0.5  # everything occupied, firmly
        volume.weight[:] = 10.0
        return volume

    centres = voxel_centres(room())
    # Hidden: the segment from the lens to the voxel passes through the body as measured (marched
    # here in 120 steps), the voxel itself is outside the grown body. The margin is what absorbs
    # a voxel reading its NEAREST pixel's ray rather than its own at the silhouette's edge.
    true_body = BodyModel(SHIPPED.boxes, margin_m=0.0)
    hidden = np.zeros(len(centres), dtype=bool)
    for t in np.linspace(0.0, 1.0, 120)[1:-1]:
        hidden |= inside(true_body, pose.translation + t * (centres - pose.translation))
    hidden &= ~inside(SHIPPED, centres)
    hidden = hidden.reshape(painted().weight.shape)

    raw, cut = painted(), painted()
    raw.integrate(dark, None, INTR, pose, law)
    cut.integrate(dark, None, INTR, pose, law, clip=clip)
    carved_raw = raw.sdf != -0.5
    carved_cut = cut.sdf != -0.5
    assert np.count_nonzero(carved_raw & hidden) > 20, "unfiltered: carved through the cart"
    assert np.count_nonzero(carved_cut & hidden) == 0, "filtered: nothing behind the body"
    assert np.count_nonzero(carved_cut & ~hidden) > 100, "the open rays still carve"


def test_a_frame_that_never_sees_the_body_is_written_bit_for_bit() -> None:
    """The working pose (straight ahead, 23.8 deg down): no ray meets the body, the mask says so
    (None: the integrator pays nothing), and a clip of nothing but infinities changes nothing. The
    synthetic eye's bottom edge is the real eye's, 55 deg down (fy 50 over 30 rows): the grown
    top_load box's front face at x 0.13 m stands 3 cm short of where that edge crosses its top."""
    intr = Intrinsics(fx=50.0, fy=50.0, cx=39.5, cy=29.5, width=80, height=60)
    pose = camera(0.0, 23.8)
    depth = march(intr, pose, SHIPPED.boxes)
    mask = BodyMask(SHIPPED)
    assert mask.for_frame(intr, pose) is None
    assert mask.for_frame(intr, pose) is None and mask.rebuilds == 1, "cached while still"
    plain, clipped = Tsdf(room()), Tsdf(room())
    plain.integrate(depth, None, intr, pose, DepthLaw(no_depth_free=True, reach_m=3.0))
    nothing = MODEL.ray_depth(intr, pose)
    assert np.isinf(nothing.z).all()
    clipped.integrate(depth, None, intr, pose, DepthLaw(no_depth_free=True, reach_m=3.0), nothing)
    assert np.array_equal(plain.sdf, clipped.sdf) and np.array_equal(plain.weight, clipped.weight)


def test_the_mask_is_rebuilt_only_when_the_head_the_optics_or_the_model_move() -> None:
    mask = BodyMask(SHIPPED)
    pose = camera(150.0, 45.0)
    assert mask.for_frame(INTR, pose) is not None
    mask.for_frame(INTR, pose)
    assert mask.rebuilds == 1
    mask.for_frame(INTR, camera(151.0, 45.0))
    assert mask.rebuilds == 2, "a degree of pan is a new grid"
    mask.model = SHIPPED
    mask.for_frame(INTR, camera(151.0, 45.0))
    assert mask.rebuilds == 2, "the same model is no change"
    mask.model = dataclasses.replace(SHIPPED, margin_m=0.02)
    mask.for_frame(INTR, camera(151.0, 45.0))
    assert mask.rebuilds == 3 and mask.model.margin_m == 0.02


def test_at_the_working_tilt_a_forward_or_side_look_never_sees_the_body() -> None:
    """At 23.8 deg down, straight ahead and to the left the frame's bottom edge (55 deg down)
    passes over the grown body, so the mask answers None and the integrator runs exactly as
    without the filter. To the right the folded arm on the top basket (the top_load box, measured
    2026-10-02) is really in the bottom rows: under a tenth of the rays at pans -30..-95 deg.
    Looking further back the margin enters the bottom rows, the body as measured does not."""
    mask = BodyMask(SHIPPED)
    for pan in np.linspace(0.0, 95.0, 20):
        hit = mask.for_frame(EYE, camera(pan, 23.8))
        assert hit is None or hit.share < 0.005, pan
    for pan in np.linspace(-95.0, -30.0, 14):
        hit = mask.for_frame(EYE, camera(pan, 23.8))
        assert hit is not None and hit.share < 0.10, pan
    back = mask.for_frame(EYE, camera(150.0, 23.8))
    assert back is not None and back.share < 0.03
    measured = BodyModel(SHIPPED.boxes, margin_m=0.0, stride_px=4).ray_depth(EYE, camera(150, 23.8))
    assert np.isinf(measured.z).all()
    assert mask.for_frame(EYE, camera(150.0, 45.0)) is not None, "reverse-gaze does see it"


def test_the_mast_stands_on_the_neck_s_own_pan_axis() -> None:
    """The column is where the neck model that places every frame puts the pan axis."""
    (mast,) = (box for box in SHIPPED.boxes if box.name == "mast")
    x, y, _z = pan_pivot(NECK)
    assert (mast.lo[0] + mast.hi[0]) / 2 == pytest.approx(x, abs=0.005)
    assert (mast.lo[1] + mast.hi[1]) / 2 == pytest.approx(y, abs=0.005)
