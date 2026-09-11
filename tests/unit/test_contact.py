"""The camera as a lidar at the floor: the inverse perspective map, the floor's run up a column,
the band's shadow, and the scan the three make — on scenes whose answer is known by geometry."""

import math

import numpy as np
import pytest

from pepin.camera import intrinsics
from pepin.contact import (
    CONTACT_MIN_RUN,
    DEPTH_NOISE,
    N_BINS,
    ColumnState,
    FloorPlane,
    bin_bearings,
    contact_scan,
    floor_scale,
    ipm,
)
from pepin.depth import SCAN_HALF_FOV, SCAN_STEP, CameraPose, Intrinsics, project

# the real optics at half the sensor's width: 78 degrees across 640 px, the camera 1.23 m above
# the wheels and tilted 26 degrees down. The horizon is above the picture, so every row is floor.
INTR = Intrinsics(*intrinsics(640, 360, 78.0), width=640, height=360)
CAM = CameraPose(x=0.0, y=0.0, z=1.23, pitch=math.radians(26.0))
PLANE = FloorPlane.of(INTR, CAM)
CENTRE = round(SCAN_HALF_FOV / SCAN_STEP)  # the bearing bin dead ahead
MID = 320  # the image column dead ahead


def _nose_down(degrees: float) -> np.ndarray:
    """The world's up in base_link when the cart leans nose down by ``degrees``."""
    a = math.radians(degrees)
    return np.array([-math.sin(a), 0.0, math.cos(a)])


def _rays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pixel's ray in base_link per unit optical depth (x, y, z), the mount's pitch in."""
    rows, cols = np.mgrid[0 : INTR.height, 0 : INTR.width]
    left = -(cols - INTR.cx) / INTR.fx
    lift = -(rows - INTR.cy) / INTR.fy
    c, s = math.cos(CAM.pitch), math.sin(CAM.pitch)
    return c + s * lift, left, -s + c * lift


def _scene(
    box_x: float | None = None,
    lean_deg: float = 0.0,
    box_height: float = 0.5,
    box_half_width: float = 0.35,
) -> np.ndarray:
    """The optical depth of every pixel of a room: the floor (the plane through the wheels, the
    cart nose down by ``lean_deg``) and an upright box whose front face stands ``box_x`` metres
    along that floor ahead, ``box_height`` tall and ``2 * box_half_width`` wide. NaN where the
    ray meets neither."""
    dx, dy, dz = _rays()
    n = _nose_down(lean_deg)  # the floor's normal
    face = np.array([math.cos(math.radians(lean_deg)), 0.0, math.sin(math.radians(lean_deg))])
    centre = np.array([CAM.x, CAM.y, CAM.z])
    n_dot_d = n[0] * dx + n[1] * dy + n[2] * dz
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(n_dot_d < -1e-9, -(n @ centre) / n_dot_d, np.inf)
    depth = np.where(t_floor > 0, t_floor, np.inf)
    if box_x is not None:
        f_dot_d = face[0] * dx + face[1] * dy + face[2] * dz
        with np.errstate(divide="ignore", invalid="ignore"):
            t_box = np.where(f_dot_d > 1e-9, (box_x - face @ centre) / f_dot_d, np.inf)
        px, py, pz = centre[0] + t_box * dx, centre[1] + t_box * dy, centre[2] + t_box * dz
        above = n[0] * px + n[1] * py + n[2] * pz
        on = (t_box > 0) & (above >= -1e-9) & (above <= box_height) & (np.abs(py) <= box_half_width)
        depth = np.where(on & (t_box < depth), t_box, depth)
    return np.where(np.isfinite(depth), depth, np.nan)


def _floor_depth_at(range_m: float) -> float:
    """The optical depth of the floor pixel whose base_link range is ``range_m``: the ray whose
    ``dx / -dz`` is ``range / height``, solved for its lift."""
    c, s = math.cos(CAM.pitch), math.sin(CAM.pitch)
    lift = (range_m * s - CAM.z * c) / (CAM.z * s + range_m * c)
    return CAM.z / (s - c * lift)


def test_the_inverse_perspective_map_undoes_the_projection_of_a_floor_point() -> None:
    """A point on the floor, projected to a pixel and lifted back through the same plane, is the
    point again — level and with the cart leaning, where the plane is the accelerometer's."""
    for lean in (0.0, 4.0):
        up = _nose_down(lean)
        wanted = np.array([[1.4, -0.3], [2.2, 0.0], [1.1, 0.45], [2.9, 0.8]])
        z = -(up[0] * wanted[:, 0] + up[1] * wanted[:, 1]) / up[2]  # on the leaning plane
        pixels = project(np.column_stack([wanted, z]), CAM, INTR)
        assert pixels.shape[0] == 4
        back_x, back_y = ipm(pixels[:, 1], pixels[:, 0], INTR, CAM, up)
        assert back_x == pytest.approx(wanted[:, 0], rel=1e-9)
        assert back_y == pytest.approx(wanted[:, 1], rel=1e-9)


def test_the_picture_s_lowest_row_already_looks_a_metre_ahead() -> None:
    """The mount's blind ring: 1.23 m up and 26 degrees down over a 49-degree vertical view, the
    bottom row's ray leaves the lens 50 degrees below the horizontal and meets the floor at
    1.23 / tan(50 deg). Nothing nearer is in the picture at all — that metre is the lidar's."""
    half_v = math.atan((INTR.height - 1 - INTR.cy) / INTR.fy)
    expected = CAM.z / math.tan(CAM.pitch + half_v)
    assert PLANE.range_m[-1, MID] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(1.02, abs=0.02)
    assert PLANE.range_m[0, MID] > 30.0  # the top row grazes the floor far away
    assert PLANE.height == pytest.approx(CAM.z)


def test_a_box_ends_the_floor_at_its_own_bearings_and_the_rest_is_clear() -> None:
    """A box 1.5 m ahead, 0.7 m wide, on an open floor: the bearings it covers carry its range
    (1.5 / cos of the bearing — the contact line is straight and the scan is polar), every other
    bearing of the fan is inf, clear out to the scan's range, and none is unknown."""
    _, _, ranges, verdict = contact_scan(_scene(box_x=1.5), PLANE)
    assert ranges[CENTRE] == pytest.approx(1.5, abs=0.02)
    for offset in (-20, 20):
        oblique = 1.5 / math.cos(offset * SCAN_STEP)
        assert ranges[CENTRE + offset] == pytest.approx(oblique, abs=0.03)
    edge = round(math.atan2(0.35, 1.5) / SCAN_STEP)
    assert np.isinf(ranges[CENTRE + edge + 4]) and np.isinf(ranges[CENTRE - edge - 4])
    assert np.isfinite(ranges[CENTRE - edge + 1 : CENTRE + edge]).all()
    assert verdict.contact > 150 and verdict.clear > 300 and verdict.unknown == 0
    assert verdict.median_range_m == pytest.approx(1.52, abs=0.03)  # the fan's own obliquity


def test_the_band_s_width_puts_the_contact_ten_per_cent_too_far_until_it_is_taken_back() -> None:
    """The mask calls a pixel floor while it stands within the band of the plane, so the last
    floor pixel on the box's face is a band up it and its ray reaches the floor past the foot:
    at 1.5 m the band is 12 cm, the camera 1.23 m, and the reported range grows by the factor
    1.23 / (1.23 - 0.12) — 16 cm of phantom clearance, the dangerous direction."""
    band = float(DEPTH_NOISE.height_band(np.array([_floor_depth_at(1.5)]), CAM.z)[0])
    assert band == pytest.approx(0.120, abs=0.005)
    raw = contact_scan(_scene(box_x=1.5), PLANE, shadow=False)[2]
    assert raw[CENTRE] == pytest.approx(1.5 * CAM.z / (CAM.z - band), abs=0.03)
    fixed = contact_scan(_scene(box_x=1.5), PLANE)[2]
    assert raw[CENTRE] - fixed[CENTRE] > 0.13


def test_a_hole_in_the_floor_mask_is_not_an_obstacle_but_a_tall_patch_is() -> None:
    """The network punches holes in the floor: a patch shorter than the minimum run is closed
    and the column stays clear out to the range, while the same patch three times as tall ends
    the floor and is reported at the range of the last floor row below it."""
    row = 270
    short = _scene()
    short[row : row + CONTACT_MIN_RUN - 2, MID - 20 : MID + 20] *= 0.5  # half a metre high
    assert np.isinf(contact_scan(short, PLANE)[2][CENTRE])
    assert np.isfinite(contact_scan(short, PLANE, min_run=1)[2][CENTRE])  # no cleaning: a hole
    tall = _scene()
    tall[row : row + 3 * CONTACT_MIN_RUN, MID - 20 : MID + 20] *= 0.5
    ranges = contact_scan(tall, PLANE, shadow=False)[2]
    assert ranges[CENTRE] == pytest.approx(PLANE.range_m[row + 3 * CONTACT_MIN_RUN, MID], abs=0.02)


def test_a_two_degree_lean_moves_every_ray_s_floor_point_by_the_geometry_s_amount() -> None:
    """A ray that leaves the lens at the pitch angle below the horizontal meets a level floor at
    ``h cos(p) / sin(p)`` — 2.52 m for the principal ray. Let the cart lean 2 degrees nose down
    on a slipper and the same ray meets the floor at ``h cos(L) cos(p) / sin(p + L)`` = 2.35 m,
    17 cm nearer, 7 % of the range: the accelerometer's up vector is an input, not a nicety."""
    p, lean = CAM.pitch, math.radians(2.0)
    level = CAM.z * math.cos(p) / math.sin(p)
    leaning = CAM.z * math.cos(lean) * math.cos(p) / math.sin(p + lean)
    assert PLANE.range_m[180, MID] == pytest.approx(level, rel=1e-12)
    tilted = FloorPlane.of(INTR, CAM, _nose_down(2.0))
    assert tilted.range_m[180, MID] == pytest.approx(leaning, rel=1e-12)
    assert level - leaning == pytest.approx(0.168, abs=0.005)
    assert tilted.height == pytest.approx(CAM.z * math.cos(lean))
    # and the box on the leaning floor comes back at its own distance along that floor
    depth = _scene(box_x=1.5, lean_deg=2.0)
    assert contact_scan(depth, tilted)[2][CENTRE] == pytest.approx(1.5 * math.cos(lean), abs=0.02)


def test_a_lean_the_frame_s_floor_scale_cannot_absorb_is_a_phantom_obstacle() -> None:
    """An unmodelled lean lifts the far floor out of the band and the scan calls open floor an
    obstacle. The frame's own floor scale hides small leans — it is fitted on the bottom rows,
    which rise with the lean — so 4 degrees still reads clear; at 7 degrees the far floor stands
    20 cm over a 15 cm band and a phantom lands at two and a half metres. This is the failure
    mode the costmap must be protected from until an open-floor drive has been measured."""
    assert np.isinf(contact_scan(_scene(lean_deg=4.0), PLANE)[2][CENTRE])
    phantom = contact_scan(_scene(lean_deg=7.0), PLANE)[2][CENTRE]
    assert 1.8 < phantom < 2.9
    told = FloorPlane.of(INTR, CAM, _nose_down(7.0))
    assert np.isinf(contact_scan(_scene(lean_deg=7.0), told)[2][CENTRE])


def test_the_range_does_not_depend_on_the_network_s_scale() -> None:
    """The claim the whole method rests on: the network's size may be a fifth wrong and the
    contact's range does not move, because the frame's own floor scale divides it out and only
    the mount and the optics place the ray. A scale the bottom of the picture cannot explain
    (0.55: a body against the bumper, not a floor) is refused and nothing is reported."""
    truth = contact_scan(_scene(box_x=1.8), PLANE)[2]
    for factor in (0.8, 1.25):
        scaled = contact_scan(_scene(box_x=1.8) * factor, PLANE)[2]
        assert np.array_equal(np.isnan(scaled), np.isnan(truth))
        both = np.isfinite(scaled) & np.isfinite(truth)
        assert both.sum() > 20 and np.abs(scaled[both] - truth[both]).max() < 1e-9
    refused = contact_scan(_scene(box_x=1.8) * 0.55, PLANE)
    assert refused[3].scale == 1.0 and np.isnan(refused[2]).all()


def test_the_floor_scale_is_the_bottom_rows_median_inside_its_bounds() -> None:
    expected = np.full((60, 10), 2.0)
    assert floor_scale(expected * 0.85, expected, rows=40) == pytest.approx(0.85)
    assert floor_scale(expected * 0.5, expected, rows=40) == 1.0  # outside the bounds
    assert floor_scale(expected, expected, rows=0) == 1.0
    blind = np.full((60, 10), np.nan)
    blind[:5] = 2.0
    assert floor_scale(blind, expected, rows=40) == 1.0  # under a tenth of the rows known


def test_a_frame_that_cannot_say_says_nothing() -> None:
    """The contract the costmap reads: NaN is unknown and neither marks nor clears. A depth
    image the law could not place is NaN everywhere and every column is blind; a floor that ends
    in unknown depth is not a clearance, so those bearings stay unknown too; and a body against
    the bumper (everything known, nothing floor) reads near, not clear."""
    blank = contact_scan(np.full((360, 640), np.nan), PLANE)
    assert np.isnan(blank[2]).all() and blank[3].blind == 640
    cut = _scene()
    cut[:240] = np.nan  # the floor ends in unplaceable depth well inside the range
    unknown = contact_scan(cut, PLANE)
    assert unknown[3].unknown == 640 and np.isnan(unknown[2]).all()
    wall = contact_scan(np.full((360, 640), 0.35), PLANE)  # a body 35 cm from the lens
    assert wall[3].near == 640 and np.isnan(wall[2]).all()


def test_the_bearing_grid_is_the_depth_scan_s_and_a_lone_column_marks_nothing() -> None:
    """inf where floor was verified, the k-th nearest contact where enough columns agree, NaN
    where nothing was seen; a mark always beats the clearance of the floor below it."""
    clear = np.array([10, 10, 11], dtype=np.int64)
    lone = bin_bearings(clear, np.array([12], dtype=np.int64), np.array([1.5]), kth=2)
    assert lone.size == N_BINS and np.isinf(lone[10]) and np.isinf(lone[11])
    assert np.isnan(lone[12]) and np.isnan(lone).sum() == N_BINS - 2
    marked = bin_bearings(
        clear, np.array([10, 10, 10], dtype=np.int64), np.array([2.0, 1.4, 1.7]), kth=2
    )
    assert marked[10] == pytest.approx(1.7) and np.isinf(marked[11])  # the 2nd nearest marks
    empty = np.array([], dtype=np.int64)
    assert np.isnan(bin_bearings(empty, empty, np.array([]))).all()


def test_the_verdict_counts_the_columns_and_the_bearings_for_the_report_line() -> None:
    verdict = contact_scan(_scene(box_x=2.0), PLANE)[3]
    assert verdict.columns == 640
    assert verdict.contact + verdict.clear + verdict.near + verdict.blind + verdict.unknown == 640
    assert verdict.marked + verdict.cleared + verdict.unseen == N_BINS
    assert 0.9 < verdict.floor_fraction <= 1.0 and verdict.scale == pytest.approx(1.0)
    assert verdict.median_range_m == pytest.approx(2.0, abs=0.05)
    assert "contact (median" in str(verdict) and "unseen" in str(verdict)
    assert ColumnState.CONTACT > ColumnState.CLEAR  # the states rank by how much the column knows
