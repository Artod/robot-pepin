"""A point in the picture as a point on the map (pepin.pixel_to_map): the geometry on synthetic
scenes whose answer is known, the depth of a box, and the locator over fake inputs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.mounts import OPTICAL_MOUNT
from pepin.neck import NeckConfig, NeckJoint, NeckPivot, NeckReference
from pepin.pixel_to_map import (
    Box,
    CameraChain,
    CameraGeometry,
    DepthImage,
    Eye,
    Head,
    Located,
    LocateError,
    Locator,
    Pair,
    Pose2D,
    RelativeError,
    StereoError,
    box_depth,
    disparity_to_depth,
    locate,
)

REPO = Path(__file__).resolve().parents[2]
W, H, F = 800, 600, 450.0
EYE = Eye.ideal(W, H, F, 0.06)
LEVEL = Head(0.0, 0.0)
ORIGIN = Pose2D(0.0, 0.0, 0.0)
STEREO = StereoError(F * 0.06, 0.5)


def neck_at(z_m: float) -> NeckConfig:
    """A neck with no lever arms whose camera sits ``z_m`` above base_link's origin."""
    pan = NeckJoint("neck", 9, 2048, 0, 4095)
    tilt = NeckJoint("head", 10, 2048, 0, 4095)
    reference = NeckReference(2048, 2048, 0.0, 0.0, z_m, 0.0, signs_verified=True)
    return NeckConfig(pan, tilt, reference, NeckPivot())


CHAIN = CameraChain(neck_at(1.2), OPTICAL_MOUNT)


def flat(z: float) -> DepthImage:
    """The same depth everywhere."""
    return DepthImage(np.full((H, W), z, dtype=np.float32), "stereo", STEREO)


def around(u: float, v: float, half: float = 20.0) -> Box:
    return Box(u - half, v - half, u + half, v + half)


def test_a_point_straight_ahead_at_two_metres() -> None:
    found = locate(EYE, CHAIN, around(400, 300), flat(2.0), LEVEL, ORIGIN)
    assert found is not None
    assert (found.forward, found.left, found.z) == pytest.approx((2.0, 0.0, 1.2), abs=1e-9)
    assert found.range_m == pytest.approx(2.0)
    assert found.bearing_deg == pytest.approx(0.0, abs=1e-9)
    assert found.sigma_m == pytest.approx(4.0 / 27.0 * 0.5)  # z^2 / (fx B) * sigma_d


def test_the_robot_s_pose_carries_the_point_onto_the_map() -> None:
    standing = Pose2D(1.0, 2.0, math.radians(90.0))  # facing the map's +y
    found = locate(EYE, CHAIN, around(400, 300), flat(2.0), LEVEL, standing)
    assert found is not None
    assert (found.x, found.y) == pytest.approx((1.0, 4.0))
    assert standing.to_base(found.x, found.y) == pytest.approx((2.0, 0.0))


def test_a_point_thirty_degrees_left_with_the_head_panned_there() -> None:
    panned = Head(math.radians(30.0), 0.0)
    found = locate(EYE, CHAIN, around(400, 300), flat(2.0), panned, ORIGIN)
    assert found is not None
    expected = (2.0 * math.cos(math.radians(30)), 2.0 * math.sin(math.radians(30)), 1.2)
    assert (found.forward, found.left, found.z) == pytest.approx(expected)
    assert found.bearing_deg == pytest.approx(30.0)


def test_a_point_thirty_degrees_left_of_a_straight_head() -> None:
    """Left in the robot is left in the picture: u = cx - f tan(30 deg); the depth image holds
    z along the optical axis, which is the forward distance for a level head."""
    u = 400.0 - F * math.tan(math.radians(30.0))
    forward = 2.0 * math.cos(math.radians(30.0))
    found = locate(EYE, CHAIN, around(u, 300, 4.0), flat(forward), LEVEL, ORIGIN)
    assert found is not None
    assert (found.forward, found.left) == pytest.approx((forward, 1.0))
    assert found.bearing_deg == pytest.approx(30.0)


def test_a_tilted_head_finds_the_floor_where_its_axis_meets_it() -> None:
    down = Head(0.0, math.radians(30.0))
    axis_to_floor = 1.2 / math.sin(math.radians(30))
    found = locate(EYE, CHAIN, around(400, 300), flat(axis_to_floor), down, ORIGIN)
    assert found is not None
    assert found.z == pytest.approx(0.0, abs=1e-6)  # the depth image is float32
    assert found.forward == pytest.approx(1.2 / math.tan(math.radians(30)))


def test_every_floor_pixel_lands_on_the_floor_whatever_the_head() -> None:
    """A depth image of a bare floor, made from the same chain, read back anywhere below the
    horizon: z is 0 within the box's own spread, panned and tilted."""
    head = Head(math.radians(-40.0), math.radians(35.0))
    rotation, origin = CHAIN.optical_in_base(head)
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = (us - EYE.cx) / EYE.fx, (vs - EYE.cy) / EYE.fy
    down = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2]  # each ray's climb in base
    z = np.where(down < -1e-6, -origin[2] / np.where(down < -1e-6, down, -1.0), np.nan)
    floor = DepthImage(z.astype(np.float32), "stereo", STEREO)
    for u, v in ((100, 450), (400, 300), (700, 580), (250, 200)):
        found = locate(EYE, CHAIN, around(u, v, 3.0), floor, head, ORIGIN)
        assert found is not None
        assert found.z == pytest.approx(0.0, abs=0.02), (u, v)


def test_the_box_depth_is_the_median_of_its_central_half() -> None:
    z = np.full((H, W), 5.0, dtype=np.float32)  # the background around the box
    z[280:320, 380:420] = 2.0  # the thing fills the box's inner half
    z[295:300, 395:400] = np.nan  # a hole in it
    depth = DepthImage(z, "stereo", STEREO)
    median, spread, pixels = box_depth(depth, Box(360, 260, 440, 340)) or (0.0, 0.0, 0)
    assert (median, spread) == (2.0, 0.0)
    assert pixels == 40 * 40 - 25
    blank = DepthImage(np.full((H, W), np.nan, dtype=np.float32), "stereo", STEREO)
    assert box_depth(blank, Box(360, 260, 440, 340)) is None


def test_a_disparity_is_depth_by_fx_times_baseline() -> None:
    z = disparity_to_depth(np.array([[27.5, 5.5, 0.0, -1.0, np.nan]]), 27.5)
    assert z[0, :2] == pytest.approx([1.0, 5.0])
    assert np.isnan(z[0, 2:]).all()
    assert StereoError(27.5, 0.5).sigma(3.0) == pytest.approx(0.1636, abs=1e-4)
    assert RelativeError(0.15).sigma(3.0) == pytest.approx(0.45)


def test_a_box_is_four_numbers_in_any_dress() -> None:
    assert Box.parse("412,300,470,352") == Box(412, 300, 470, 352)
    assert Box.parse("box [412.5, 300, 470, 352]") == Box(412.5, 300, 470, 352)
    for bad in ("1,2,3", "printer", "10,10,5,20"):
        with pytest.raises(ValueError):
            Box.parse(bad)
    assert Box(0, 0, 10, 10).inner(0.5) == Box(2.5, 2.5, 7.5, 7.5)
    assert not Box(900, 0, 950, 10).inside(W, H)


def test_the_rectified_eye_is_the_depth_host_s_pinhole() -> None:
    """The eye this module rectifies with is the one pepin.stereo.Rectifier hands the depth host:
    the same pinhole and baseline, and a raw pixel lands where the remap tables fetch it from."""
    from pepin.stereo import Rectifier, StereoCalibration

    calibration = StereoCalibration.load(REPO / "config" / "stereo_calibration.json")
    eye = Eye.from_calibration(calibration)
    rectifier = Rectifier.from_calibration(calibration)
    assert (eye.fx, eye.cx, eye.cy) == pytest.approx((rectifier.fx, rectifier.cx, rectifier.cy))
    assert eye.baseline_m == pytest.approx(rectifier.baseline_m)
    assert eye.fb_px_m == pytest.approx(27.5, abs=0.2)
    map_u, map_v = rectifier.left_maps
    for u, v in ((400.0, 300.0), (150.0, 120.0), (650.0, 480.0), (90.0, 520.0)):
        ru, rv = eye.rectified(u, v)
        col, row = round(ru), round(rv)
        assert (map_u[row, col], map_v[row, col]) == pytest.approx((u, v), abs=1.0), (u, v)


def test_the_real_chain_puts_the_eye_where_the_configs_say() -> None:
    """At the working pose the left eye sits 1.203 m up, 3 cm left of the neck's link, and looks
    down by the mount's 23.8 deg plus the eye block's own 1.1."""
    chain = CameraChain.load(REPO / "config")
    rest = Head(0.0, math.radians(chain.neck.reference.pitch_deg))
    rotation, origin = chain.optical_in_base(rest)
    assert origin == pytest.approx((0.0, 0.0305, 1.203), abs=2e-3)
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert math.degrees(math.asin(-axis[2])) == pytest.approx(24.9, abs=0.3)
    assert chain.camera_height_m(rest) == pytest.approx(1.203, abs=2e-3)


# -- the locator over fake inputs -----------------------------------------------------------------


class Pairs:
    def __init__(self) -> None:
        self.grabbed = 0

    def grab(self) -> Pair:
        self.grabbed += 1
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        return Pair(blank, blank, 0.0)


class Depth:
    def __init__(self, name: str, z: float | None, down: bool = False) -> None:
        self.name, self.z, self.down = name, z, down

    def depth(self, left: np.ndarray, right: np.ndarray) -> DepthImage:  # type: ignore[type-arg]
        if self.down:
            raise LocateError("refused")
        value = np.nan if self.z is None else self.z
        error = STEREO if self.name == "stereo" else RelativeError(0.15)
        return DepthImage(np.full((H, W), value, dtype=np.float32), self.name, error)


class Still:
    def __init__(self, head: Head = LEVEL, pose: Pose2D = ORIGIN) -> None:
        self._head, self._pose = head, pose

    def head(self) -> Head:
        return self._head

    def pose(self) -> Pose2D:
        return self._pose


def locator(*depths: Depth) -> Locator:
    still = Still()
    geometry = CameraGeometry(EYE, CHAIN, lambda left, right: (left, right))
    return Locator(Pairs(), list(depths), still, still, geometry)


def test_the_locator_takes_stereo_first_and_the_mono_network_where_stereo_is_blank() -> None:
    found: Located = locator(Depth("stereo", 2.0), Depth("mono", 3.0)).locate(around(400, 300))
    assert (found.source, found.forward) == ("stereo", pytest.approx(2.0))
    found = locator(Depth("stereo", None), Depth("mono", 3.0)).locate(around(400, 300))
    assert (found.source, found.forward) == ("mono", pytest.approx(3.0))
    assert found.sigma_m == pytest.approx(0.45)
    found = locator(Depth("stereo", 2.0, down=True), Depth("mono", 3.0)).locate(around(400, 300))
    assert found.source == "mono"


def test_the_locator_says_why_no_source_answered() -> None:
    with pytest.raises(LocateError) as refused:
        locator(Depth("stereo", None), Depth("mono", 0.0, down=True)).locate(around(400, 300))
    assert "stereo: no depth in that box" in str(refused.value)
    assert "mono: refused" in str(refused.value)
    with pytest.raises(LocateError, match="outside the 800x600 picture"):
        locator(Depth("stereo", 2.0)).locate(Box(900, 10, 950, 40))


# -- the live inputs, their remotes faked ---------------------------------------------------------

Grid = np.ndarray  # type: ignore[type-arg]


def test_the_depth_host_s_disparity_is_depth_and_its_silence_a_refusal() -> None:
    from pepin.pixel_to_map import HostStereo
    from pepin.stereo_host import StereoHostError

    sent: list[tuple[int, ...]] = []

    def answers(left: Grid, right: Grid) -> Grid:
        sent.append(left.shape)
        return np.full(left.shape, 13.75, dtype=np.float32)

    stereo = HostStereo("http://127.0.0.1:1", fb_px_m=27.5)
    stereo._remote = answers  # type: ignore[assignment]
    colour = np.zeros((6, 8, 3), dtype=np.uint8)
    depth = stereo.depth(colour, colour)
    assert sent == [(6, 8)], "the eyes go grey"
    assert float(depth.z[0, 0]) == pytest.approx(2.0)
    assert depth.error.sigma(2.0) == pytest.approx(4.0 / 27.5 * 0.5)

    def silent(left: Grid, right: Grid) -> Grid:
        raise StereoHostError("refused")

    stereo._remote = silent  # type: ignore[assignment]
    with pytest.raises(LocateError, match=r"depth_host\.sh stereo"):
        stereo.depth(colour, colour)


def test_the_mono_network_answers_metres_with_its_assumed_error() -> None:
    from pepin.depth_service import DepthServiceError
    from pepin.pixel_to_map import HostMono

    seen: list[Grid] = []

    def answers(rgb: Grid) -> Grid:
        seen.append(rgb)
        return np.full(rgb.shape[:2], 3.0, dtype=np.float32)

    mono = HostMono("http://127.0.0.1:1")
    mono._remote = answers  # type: ignore[assignment]
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue, in OpenCV's order
    depth = mono.depth(bgr, bgr)
    assert seen[0][0, 0].tolist() == [0, 0, 255], "the network is sent RGB"
    assert (float(depth.z[0, 0]), depth.source) == (3.0, "mono")
    assert depth.error.sigma(3.0) == pytest.approx(0.45)

    def silent(rgb: Grid) -> Grid:
        raise DepthServiceError("refused")

    mono._remote = silent  # type: ignore[assignment]
    with pytest.raises(LocateError, match="mono depth"):
        mono.depth(bgr, bgr)


def test_the_live_eyes_load_the_rig_and_aim_from_the_working_pose() -> None:
    """The geometry the live locator reads from config/ (stereo calibration, neck, mounts), its
    head and pose sources over the tools' clients, and the aim it gives the head: a face 1.6 m
    up at 0.8 m is above the 1.2 m camera."""
    from pepin.tools.eyes import aim_at, stereo_eyes
    from pepin.tools.fakes import FakeClock, FakeGoalServer, FakeNeck

    clock = FakeClock()
    goals = FakeGoalServer(clock)
    eyes = stereo_eyes("127.0.0.1", 1, "http://127.0.0.1:1", FakeNeck(clock), goals)
    geometry = CameraGeometry.load(REPO / "config")
    eyes._geometry = geometry
    assert geometry.eye.width == 800 and geometry.eye.fb_px_m == pytest.approx(27.5, abs=0.2)
    blank = np.zeros((600, 800, 3), dtype=np.uint8)
    assert geometry.rectify(blank, blank)[0].shape == (600, 800, 3)
    here = Pose2D(1.0, 1.0, math.radians(90.0))
    aim = eyes.aim(1.0, 1.8, 1.6, here)  # straight ahead, 0.8 m, 1.6 m up
    assert aim.pan_deg == pytest.approx(0.0, abs=1e-6)
    assert aim.tilt_deg == pytest.approx(-math.degrees(math.atan2(1.6 - 1.203, 0.8)), abs=0.2)
    side = aim_at(0.0, 1.0, 0.0, Pose2D(0, 0, 0), 1.0)
    assert (side.pan_deg, side.tilt_deg) == pytest.approx((90.0, 45.0))
    assert eyes.head_source.head() == Head(0.0, pytest.approx(math.radians(23.8)))
    assert eyes.pose_source.pose().x == pytest.approx(0.1)
    goals.pose = None
    with pytest.raises(LocateError, match="does not know where it is"):
        eyes.pose_source.pose()
