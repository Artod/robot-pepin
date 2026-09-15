"""The depth correction as a pipeline: anchors say what is true, laws turn depth into metres.

:mod:`pepin.depth` holds the pieces — the lidar's beams projected into the image, the affine law
fitted on their (network, true) pairs, the edge filter, the floor snapped to its plane — and the
node ran them in one fixed order. Here the same pieces run as an ordered list of stages, each
switchable by name, so a node's flags toggle stages instead of branches, every stage's cost and
effect is counted per frame, and a new source of truth is one more stage in the list.

Two roles. An :class:`Anchor` is an external truth about some pixels: it contributes
:class:`Pairs` — (network depth, true depth, weight) — to the pool the laws fit on, and / or
corrects the pixels it knows directly. A :class:`Law` is the map from the network's depth to
metres, fitted on the pooled pairs and applied to the whole image. Today's chain is
:class:`EdgeFilter` -> :class:`LidarAnchor` -> :class:`AffineLaw` -> :class:`FloorAnchor`, bit
for bit what ``pepin.depth`` computed for the node (the tests hold the two against each other).
Two new anchors stand beside the lidar: :class:`FloorPairs` — the floor's pixels pair the
network's depth with the plane's geometric depth, a second hoop the law can be fitted through
with no lidar at all — and :class:`WallAnchor` — the lidar's returns extruded up the image
columns while the network's depth stays continuous (a wall goes on, a chair back ends at its
top), a third hoop above the lidar's row; and :class:`ParallaxAnchor` — the corners this frame
shares with the previous one, triangulated against the odometry's transform between the two
stamps (:mod:`pepin.parallax`), a fourth hoop that needs no second sensor and no assumed plane
and that measures at every elevation the picture has. Two more laws let the data say whether the
error above the lidar's row depends on the elevation: :class:`ElevationLaw` adds a term in the
ray's lift, :class:`RowLaw` fits a law per band of rows.

Measured on run 0171 (29 frames against the run's lidar cloud and its COLMAP reference,
scratch/pipeline_vs_truth.py, 2026-09-11), which set the defaults of :func:`standard_pipeline`:
the network's error is not one law — 1.1x too far on the floor, 1.6x at the lidar's row, 2.0x
from 0.3 m up — so every hoop but the lidar's pulls the law off the row the costmap lives on.
Floor pairs alone put the lidar's row 2.0x too far (a camera without lidar sees walls where the
raw network does); wall pairs at 0.2 of a beam put it 10 % too near while fixing the 0.5-0.8 m
slice (0.86 -> 1.00 of the truth); the elevation term is real (c +0.1) but linear in lift is the
wrong shape (the error steps within 60 rows of the lidar's row and is flat above), and the row
law overfits. The lidar-only anchor stays the default; the new anchors ship switched off.

The law that ships live is :class:`RangeLawStage`, not the affine one: fitted on the same pool,
it bins the pairs by the network's own depth and measures a ratio in each bin, because the
affine law's residual tilts 12 % per metre of range (:class:`pepin.depth.RangeLaw`, measured
2026-09-14 in scratch/depth_scale_by_range.py). The affine law keeps running before it — it is
what the law file carries, what seeds every law at start, and the fallback the range law
publishes through until two of its bins fill.

The law of the frame in hand (:class:`FrameLaw`) is not one pair of numbers either: it is a
coarse grid of nodes over the picture (:class:`ScaleField`), each fitted on the pairs that land
near it and held toward the frame's global fit and toward its own last value, because this
network's error is regime-wise — 1.1x on the floor, 1.6x at the lidar's row, 2.0x above it —
and one law fitted across all three is wrong in all three. A grid of 1x1 is that single law
again, bit for bit.

:class:`RayLaw` (:mod:`pepin.elevation`) is the third such law and the one parameterised the
way the error is: by the ray's angle off the optical axis, so the neck may tilt without
refitting. Measured held out on the same 29 frames (scratch/ray_law_eval.txt, 2026-09-12) it
tightens the scatter of the beams' residual on three drive halves of four (q3 0.46 -> 0.13 on
the widest) and moves the median 5-10 % near, and its scale reproduces between two tapes of the
same room at the bottom and the middle of the frame (12 % and 8 % apart) but not at the top
(30 %). It ships switched off, and on the lidar's beams alone it is refused outright: a
return's elevation is a curve of its range, so there the angle and the depth are one regressor
(:func:`pepin.elevation.separable`) and the affine law stands instead.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pepin.contact import DEPTH_NOISE, DepthNoise
from pepin.depth import (
    B_BOUNDS,
    EDGE_REL_STEP,
    FLOOR_HEIGHT_TOLERANCE,
    FRAME_HOLD_TAU_S,
    FRAME_MIN_PAIRS,
    FRAME_MIN_SPREAD,
    MIN_DEPTH_SPREAD,
    MIN_SAMPLES,
    NEAR_M,
    POOL_FRAMES,
    POOL_MIN_SAMPLES,
    UP_LEVEL,
    Array,
    CameraPose,
    Intrinsics,
    Mask,
    RangeLaw,
    a_bounds,
    apply_affine,
    at_bound,
    beam_hits,
    drop_edges,
    edge_mask,
    fit_affine,
    fit_frame,
    fit_node,
    floor_anchor,
    floor_depth,
    in_image,
    inverse_sigma,
    node_unit,
    pair_weight,
    project,
    project_all,
)
from pepin.elevation import RAY_AZIMUTH_DEGREE, RAY_DEGREE, RayGain, fit_ray, ray_angles

if TYPE_CHECKING:  # the tracker's own module stays a lazy import inside the parallax stage
    from pepin.parallax import Features, FrameView, Lens, Motion, ParallaxTruth, TrackStore

LEAN_STEP = 0.003  # the floor's expected depth is recomputed when the up vector moves this much
FLOOR_PAIR_STRIDE = 8  # every 8th row and column of the floor: 3600 candidates of a 640x360 frame
FLOOR_SIGMA_PITCH_DEG = 1.5  # how well the camera's pitch is known: config/neck.json's ticks_note
# reads "head level by eye the tilt servo reads 2068 ticks and the picture is 1.0 deg down
# (+-1.5)". It is what a floor pair's own noise is made of: the plane's depth under a ray is
# h / sin(angle below the horizon), so a pitch error tilts the whole ruler (:func:`floor_sigma`).
FLOOR_BOOTSTRAP_STARTS = (50.0, 65.0, 75.0, 85.0, 90.0)  # percentiles of the candidates' own
# network-over-plane ratio the floor's first scale is tried from, before any law exists; the one
# whose three turns end up admitting the most pixels wins (:meth:`FloorPairs._bootstrap`). The
# median alone — where this started — is biased low, because everything below the horizon that is
# not floor is NEARER than the floor and reads a lower ratio. Measured over the four tapes of
# 2026-09-15 (scratch/_floor_bootstrap_start.py): the median start settles at 0.85 / 0.83 / 1.13 /
# 1.76 against lidar ratios of 2.21 / 1.38 / 1.91 / 2.10, and "the start that admits the most"
# settles at 1.41 / 0.94 / 1.63 / 1.76 — the right basin on the two tapes that have one.
FLOOR_NORMAL_TOL_DEG = 5.0  # how far the floor pixels' own fitted plane may lean from the geometric
# up before the frame's floor pairs are refused outright: a plane fitted to a table top, a ramp or
# a wrong law is not the floor, and pairs taken off it move every node they touch.
FLOOR_BAND_MAX_M = 0.20  # metres: the widest the floor's height band may ever grow. The band that
# decides "is this pixel on the floor?" is 2 * h * (0.03 + 0.01 E) — the network's relative error
# turned into height — so it grows with the floor's OWN depth and never stops: 12 cm at 2 m, 20 cm
# at 5 m, 32 cm at 10 m, 2.2 m at 90 m. The rows within half a degree of the horizon all sit at
# such depths, and there the band has stopped being a test: it admits everything between the floor
# and the ceiling, which in a room means the WALL standing at that bearing.
# Measured on tape 0318 (a closed door 2 m ahead, scratch/floor_gate_probe.txt, 2026-09-15): 7 % of
# the candidates were the door itself at head height — 79 pixels whose floor depth reads 90 m and
# whose band is 2.24 m wide, standing 1.18 m over the floor at x = 1.94 m. They turned the fitted
# plane from 12 degrees of lean into 50, with a normal pointing nearly forward; and once the gate
# let a frame through they took the floor-only law with them — a hundredth of the truth on the next
# frame, after which the near floor no longer fitted its own band and only the far pixels were left.
# 20 cm is where the band stops separating the floor from what stands on it (pepin.depth's
# SCAN_MIN_Z_M, the height the cart's own scan calls an obstacle from, is 15 cm), and it is a cap,
# not a cut: the far floor keeps its pairs and its lever arm, only the far WALL loses them.
FLOOR_PLANE_BAND = True  # judge the fitted floor plane in METRES against the very band the pixels
# were selected inside, instead of in fixed degrees off the up vector (:meth:`FloorPairs._is_floor`)
PLANE_OFF_PERCENTILE = 95.0  # the band gate reads the plane's departure at this percentile of the
# pixels rather than at the worst one: a single lattice pixel at the footprint's corner must not
# throw a frame's whole floor away.
WALL_ROW_STRIDE = 4  # rows between two wall pairs of one column
WALL_SIGMA_RANGE_M = 0.015  # metres: one LD19 beam's range noise, the measurement the plane is
# built out of (:func:`wall_sigma` turns it into the pair's own sigma; it was a flat weight of
# 0.2 until 2026-09-15, which said a wall pixel a beam away and one at the top of the picture
# were worth the same fifth of a beam).
WALL_SIGMA_HEIGHT = 0.05  # metres of doubt per metre of HEIGHT above the lidar's line: the
# world assumption's own error bar. "The surface goes on upwards" is true of a door and a wall
# and false of a sofa back, a shelf or a table, and nothing in the picture measures it — so the
# ruler FADES with the distance from its evidence instead of switching off at a threshold: a
# wall pixel a metre above the beam is trusted to 5 cm whatever the geometry says.
WALL_WEIGHT_CAP = 1.0  # a wall pair is one beam's range carried up a column, so it may never
# outweigh the beam it came from — the parallax anchor's cap, for the same reason.
WALL_COLUMN_SHARE = True  # one beam, one vote: the pairs of a column all rest on that column's
# single return, so they SHARE its weight instead of each carrying it (50 rows of one beam
# counted 50 times took 67-85 % of a frame's whole fit weight, scratch/wall_field_row_eval.txt).
WALL_MAX_HEIGHT = 2.0  # metres above the floor a wall point may stand: higher is a ceiling
WALL_NEIGHBOUR_GAP = 0.30  # metres between a return and its scan neighbours for a wall direction
WALL_SLOPE_TOL = 0.004  # per row: how much faster than the plane the network's depth may climb
WALL_SLOPE_WINDOW = 6  # rows either side over which that climb is measured (the noise averaged)
WALL_DRIFT_TOL = 0.0  # how far the network's depth may drift from the plane's OVER THE WALK,
# relative; 0 is off, and off is the reasoned default. The gate exists because the per-row
# slope test is blind to a SLOW recession: a surface leaning back 0.3 m per metre of height
# moves 0.08 % a row against a tolerance of 0.4 % and reaches the top of the picture as if it
# were a wall. The integral would catch it — and would catch the ruler's whole reason for
# existing with it. The network's depth over this camera climbs 1.6x at the lidar's row and
# 2.0x by 0.3 m above it (scratch/pipeline_vs_truth.txt, 2026-09-11): 50-80 % of scale drift
# per metre of height, where the recession to be caught is 15 %. A gate tight enough to refuse
# the recession refuses every real wall, and one loose enough to pass a real wall never fires.
# The network cannot tell "the surface is leaning back" from "I am wrong above the row", which
# is what this ruler is for; what bounds the damage instead is WALL_SIGMA_HEIGHT, the error bar
# that grows with the height, and the gates below that need no network at all.
WALL_MIN_WALK_M = 0.5  # metres above the lidar's line a column must reach, undisturbed, before
# any of its pairs count. The ruler exists for the TOP of the picture; a stump that dies 20 cm
# up adds pairs where the beams already speak and carries the full risk of being a chair back.
MIN_LIFT_SPREAD = 0.15  # the pool's elevation span (5th-95th of lift) before an elevation term
ROW_BANDS = 6  # bands of elevation of the row law
PARALLAX_MIN_GAP_S = 0.08  # a partner frame nearer in time than this has no baseline to speak of
PARALLAX_MAX_GAP_S = 0.60  # farther back than this the view has changed more than the flow follows
PARALLAX_ORB_MAX_GAP_S = 1.5  # the describer's window: a keypoint is recognised, not followed
PARALLAX_MIN_BASELINE_M = 0.10  # the parallax a partner is chosen to reach: 0.4 s at 0.25 m/s
PARALLAX_MATCHER = "klt"  # who finds the correspondences: the flow or the describer
PARALLAX_MOTION = "tf"  # whose word on the baseline: the map pose TF gives on EVERY frame (the
# newest map -> odom composed with this moment's odom -> base_link), the tracker's own map pose
# where it covers the frame's stamp, or the odometry
PARALLAX_MAP_WAIT = False  # ask the map pose without waiting: a wait costs the whole frame rate
PARALLAX_MAP_MAX_AGE_S = 0.3  # a map pose older than this is not this frame's: odometry answers
PARALLAX_MOTIONS = ("tf", "tracker", "odom")
PARALLAX_CORRECTION_TOL_M = 0.05  # how far the tracker's map -> odom correction may jump before
# the window is dropped: a relocalisation moves every pose stored before it relative to every
# pose after it, and that displacement is not one the camera made. Read as the metres it puts on
# a point PARALLAX_CORRECTION_REACH_M ahead, so a turn of the map counts as well as a shift.
PARALLAX_CORRECTION_REACH_M = 2.0
PARALLAX_UNDISTORT = True  # straighten the tracked pixels with camera_info's own distortion
# before the epipolar test, the triangulation and the reprojection
PARALLAX_RING_FRAMES = 48  # frames kept to reach back through, the BACKWARD window's and the
# pair's alone (the forward store keeps the previous grey and one per view, and no ring at all).
# The ring is pruned by TIME (parallax_track_window_s); this is the memory bound under it, and at
# 24 a 3 s window on a 16 frames/s camera was silently cut to 1.5 s. 48 greys at 640x360 is 11 MB,
# and it is what caps a backward window at 48 / the frame rate however long the knob is set.
PARALLAX_WEIGHT = 1.0  # the multiplier on a parallax pair's own 1 / sigma^2 (the A/B's knob)
PARALLAX_TRACK_MIN_OBS = 3  # frames a corner must be seen in to be a track; 2 is the old pair
PARALLAX_TRACKING = "forward"  # how a corner becomes a track: followed FORWARD one hop a frame
# (two flow calls a frame whatever the window), the old backward "window" re-tracked from the
# current frame every frame (two calls per view), or "pair" — this frame against one partner.
PARALLAX_TRACKINGS = ("forward", "window", "pair")
PARALLAX_TRACK_WINDOW_S = 3.0  # how far back a track reaches, seconds. 1.5 until 2026-09-15,
# when the backward build's cost stopped setting it: every error term of a parallax depth
# divides by the baseline and the tracker's map pose is absolute, so a longer window is free.
PARALLAX_MAX_TRACKS = 200  # corners the forward store follows at once
PARALLAX_REDETECT_EVERY = 5  # frames between two hunts for new corners
PARALLAX_VERIFY_EVERY = 10  # frames between two rounds of the long-range drift bound; 0 is off
PARALLAX_DRIFT_TOL_PX = 1.0  # how far a hopped corner may sit from where its birth patch lands
PARALLAX_MIN_TOTAL_BASELINE_M = 0.10  # the effective parallax a track's views must add up to
PARALLAX_TRACK_MAX_VIEWS = 8  # views one track may rest on (pepin.parallax.TRACK_MAX_VIEWS)
PARALLAX_SIGMA_MODEL = "covariance"  # a track's sigma: the solve's own covariance. The
# closed form the stage shipped with is "baseline" (pepin.parallax.TRACK_SIGMA_MODELS); the
# string is spelled out here because this module imports pepin.parallax lazily, for cv2's sake.
PARALLAX_SPLIT_TOL_SIGMA = 0.0  # how far a track's two halves may disagree, in sigmas; 0 is off
# and off is the measured default: at 3 the gate removes 1.1 % of the real errands' tracks, leaves
# the parallax-only law exactly where it was and costs 3.9 ms a frame of the stage's 27.0.
# Every PARALLAX_TRACK_* / _SIGMA_ / _SPLIT_ value above is pepin.parallax's own TRACK_* default,
# restated here so the pipeline's constants read in one place (the tracker's own module stays a
# lazy import in the stage).
LIDAR_SIGMA_M = 0.0  # metres: one beam's range noise, 0 meaning every beam weighs a flat 1 —
# the reference pair itself (REF_SIGMA_INV, "a beam at 2 m"), which is what the parallax anchor
# has always weighed its corners against, so the two rulers still share one unit.
# Measured 2026-09-15 (scratch/parallax_ruler_recheck.txt, the four errands of 2026-09-14,
# 112 frames, odd beams fitting and even beams judging): weighing each beam 1 / sigma^2 at
# sigma_m = 1.5 cm makes the LIDAR-ONLY law worse, 7.4 % -> 11.4 % of median |residual| overall
# and 6.4 % -> 18.3 % over 1.0-1.5 m, because sigma_m / z^2 puts the weight as z^4 (a beam at
# 8 m counts 256 beams at 2 m) and the far beams then fit themselves: 3-12 m improves
# 10.5 % -> 4.4 % and everything the cart parks against loses. The reason is that the fit
# minimises the residual of the NETWORK's 1 / D, whose own noise (0.02-0.10 of inverse depth at
# a few per cent of range) dwarfs a beam's (0.0002-0.023) everywhere — so a beam's sigma is not
# the residual's sigma, and 1 / sigma_beam^2 is not that pair's weight in this fit.
# Live knob: the node's lidar_sigma_m, > 0 to weigh the beams by range again.

FIELD_GRIDS = ("1x1", "2x2", "3x3", "4x3", "4x4")  # how many nodes a scale field carries, written
# the way an image size is — COLUMNS x ROWS, so 4x3 is four across and three down.
FIELD_PRIOR = 0.3  # in PAIRS (a lidar beam is 1): how hard a node is pulled toward the frame's
# own global fit. It means what it says — a Tikhonov prior carrying as much information about
# the node's law as 0.3 pairs of weight 1 would at that node (:func:`pepin.depth.fit_node`) — and
# a node that saw nothing still comes back as the global fit, which is what makes a field safe:
# it degrades to today's single law wherever the anchors are sparse.
# Swept 0.03 / 0.1 / 0.3 / 1 / 3 against a carry of 0.1 / 1 / 3, 2026-09-15, on two judges at
# once. (1) The lidar's own row, held out on the CONTIGUOUS split — the first half of a frame's
# beams fits, the second judges, then the reverse (scratch/field_prior_row_sweep.py; the parity
# split grades a node on the immediate neighbours of the beams that fitted it and flatters the
# field). Mean median |residual| over run 0171's drive and the three neck pitches: 11.9 / 11.2 /
# 11.3 at prior 0.03, 12.0 / 11.5 / 11.4 at 0.1, 12.3 / 12.1 / 11.8 at 0.3, 13.4 / 13.1 / 13.2 at
# 1, 14.1 / 13.7 / 13.7 at 3 (carry 0.1 / 1 / 3), against the single law's 15.8 and the OLD
# pseudo-observation prior's 12.1 at its own 1.0. (2) The wall ABOVE that row, where no beam ever
# judges (scratch/wall_truth_eval.py, tapes 0313 and 0268): the lidar chain reads 17.1 / 15.6 % of
# median |residual| at prior 0.3 against 16.7 / 15.0 at 3 and 17.3 / 16.1 at 0.03 — a whole
# hundred-fold of prior moves it by under a point, because above the row the field has almost
# nothing of its own to fit (the parallax anchor lands 0.000-0.004 of pair weight a frame in the
# top row of nodes). So the row decides, and the row wants a light prior: 0.3 improves the
# held-out row by 0.3 points against today's default and costs 0.4 above it.
FIELD_CARRY = 3.0  # the same units: how hard a node is pulled toward what it was on the last
# frame, decayed by exp(-dt / FIELD_CARRY_TAU_S). Three beams' worth, and the one knob that is a
# win everywhere it was measured (2026-09-15, the sweep above): at the lidar's row it takes the
# drive's TOP third of the picture from 28.7 to 23.5 % at prior 0.3 and leaves 1 node fit of 846
# pinned at a bound against 8 at carry 1; above the row it is worth 0.2-0.5 points on both tapes
# and both wall-pixel selections. What it is there for is the frames with no beams at all:
# without a lidar it is the only thing carrying a node's scale from the last frame that saw
# something, and FIELD_CARRY_SPENT is what stops it carrying for ever.
FIELD_CARRY_TAU_S = 2.0  # seconds over which that pull decays: a node starved for a time constant
# keeps a third of the carry, and one starved for five seconds is the global fit again. The frame
# law's own hold constant (FRAME_HOLD_TAU_S), by design, unmeasured as a choice of its own.
FIELD_CARRY_SPENT = 0.01  # what is left of the carry when it is dropped outright rather than
# decayed further. A decay is a RELATIVE statement, and at field_prior 0 there is nothing for it
# to be relative to: a node with no pairs and a carry of 1e-13 is still held by that carry alone
# and comes back as its own last value exactly, forever. Measured at field_prior 0 before this
# floor existed (scratch/_field_hazards.py, 2026-09-15): a starved node read its 20-second-old
# value to four decimals, the decay having done nothing at all. A hundredth is 4.6 time
# constants, 9.2 s at the default tau — past the point where the pull moves the fit by a per
# cent, so the defaults are unchanged to well under their own noise (at FIELD_PRIOR 0.3 and
# FIELD_CARRY 3.0 a carry spent to 0.03 is a tenth of the prior beside it).


# ---- one table of defaults ----------------------------------------------------------------------
# Every switch of the chain, its default once. The node's FLAGS table reads its defaults from here
# (pepin_bringup.depth_stream) and so does :func:`standard_pipeline`, so a default lives in one
# place instead of three (a unit test holds the two tables against each other).
PIPELINE_DEFAULTS: dict[str, bool | float | str] = {
    # floor_pairs ON since 2026-09-16: with the band capped and the plane judged in metres the
    # floor pairs feed 55-125 of every 79-125 door frames at 10 % of the fit weight and cost the
    # lidar chain nothing (scratch/wall_truth_eval.py), lift the lidar row on the 0171 drive
    # 17.0 -> 11.3 % (block split, scratch/scale_field_eval.py) and alone read a door 2 m away
    # to 4-9 % with no lidar in the chain — the field keeps them off the lidar's own nodes.
    "floor_pairs": True,
    # wall_anchor ON since 2026-09-16: judged on the COLMAP truth of run 0171 OFF the lidar's
    # extrusion (furniture, clutter: n=5159) 19.1 -> 16.1 %, 0.3-0.6 m up 46 -> 30, nothing
    # worse in any band; the held-out lidar row never loses (scratch/wall_vs_colmap.py,
    # scratch/wall_field_row_eval.py); the world assumption is priced by wall_sigma_height.
    "wall_anchor": True,
    # parallax_anchor ON since 2026-09-16: forward tracks cost 6 ms a frame and the depth
    # stream held 7-8 fps with them on during the door drives; alone they read a door 2 m away
    # to 5-6 % above the lidar's row on straight legs (scratch/wall_truth_eval.py).
    "parallax_anchor": True,
    "ray_law": False,
    "range_law": True,
    "frame_law": True,
    "wall_correct": False,
    "field_grid": "3x3",
    "field_prior": FIELD_PRIOR,
    "field_carry": FIELD_CARRY,
    "field_carry_tau_s": FIELD_CARRY_TAU_S,
    "floor_sigma_pitch_deg": FLOOR_SIGMA_PITCH_DEG,
    "floor_normal_tol_deg": FLOOR_NORMAL_TOL_DEG,
    "floor_band_max_m": FLOOR_BAND_MAX_M,
    "floor_plane_band": FLOOR_PLANE_BAND,
    "wall_sigma_height": WALL_SIGMA_HEIGHT,
}


def default_switch(name: str, given: bool | None) -> bool:
    """``given`` when a caller said so, else :data:`PIPELINE_DEFAULTS`' own value for the
    switch called ``name``."""
    return bool(PIPELINE_DEFAULTS[name]) if given is None else bool(given)


def grid_of(choice: str) -> tuple[int, int]:
    """The (rows, columns) of nodes named by a ``field_grid`` choice — ``"4x3"`` is four
    columns and three rows, the way an image size is written. ``ValueError`` for anything
    else."""
    if choice not in FIELD_GRIDS:
        raise ValueError(f"{choice!r} is not one of {', '.join(FIELD_GRIDS)}")
    cols, rows = (int(part) for part in choice.split("x"))
    return rows, cols


def grid_name(grid: tuple[int, int]) -> str:
    """A (rows, columns) grid written the way :func:`grid_of` reads it."""
    return f"{grid[1]}x{grid[0]}"


# ---- what flows through the pipeline ----------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Pairs:
    """What an anchor knows about some pixels: the network's depth ``d`` there, the true depth
    ``z`` (metres along the optical axis), each pair's ``weight`` in a fit (a lidar beam is 1),
    and where the ray points — its ``lift``, the elevation above the optical axis per unit
    depth, ``-(row - cy) / fy``, and its ``left``, the same for the azimuth,
    ``-(column - cx) / fx`` (zero where an anchor does not say). The angle-aware laws read
    those two: the tangents of the ray's angles off the optical axis, the camera's own
    parameter, which no tilt of the neck moves."""

    d: Array
    z: Array
    weight: Array
    lift: Array
    left: Array

    @classmethod
    def of(
        cls,
        d: Array,
        z: Array,
        lift: Array,
        weight: float | Array = 1.0,
        left: Array | None = None,
    ) -> Pairs:
        """Pairs from arrays, ``weight`` one number for all or one per pair, ``left`` zero
        (the optical axis' own column) when the anchor does not know the azimuth."""
        dd = np.asarray(d, dtype=float)
        w = (
            np.full(dd.shape, float(weight))
            if np.ndim(weight) == 0
            else np.asarray(weight, dtype=float)
        )
        lifted = np.asarray(lift, dtype=float)
        sideways = np.zeros_like(dd) if left is None else np.asarray(left, dtype=float)
        return cls(dd, np.asarray(z, dtype=float), w, lifted, sideways)

    @property
    def size(self) -> int:
        """How many pairs."""
        return int(self.d.size)

    @staticmethod
    def join(parts: Sequence[Pairs]) -> Pairs | None:
        """All the parts as one, or ``None`` for no parts."""
        if not parts:
            return None
        return Pairs(
            np.concatenate([p.d for p in parts]),
            np.concatenate([p.z for p in parts]),
            np.concatenate([p.weight for p in parts]),
            np.concatenate([p.lift for p in parts]),
            np.concatenate([p.left for p in parts]),
        )


def lift_of(rows: npt.ArrayLike, intr: Intrinsics) -> Array:
    """The elevation of the ray through ``rows`` above the optical axis, per unit depth."""
    out: Array = -(np.asarray(rows, dtype=float) - intr.cy) / intr.fy
    return out


def left_of(columns: npt.ArrayLike, intr: Intrinsics) -> Array:
    """The azimuth of the ray through ``columns`` left of the optical axis, per unit depth."""
    out: Array = -(np.asarray(columns, dtype=float) - intr.cx) / intr.fx
    return out


class Rigid(Protocol):
    """A rigid transform: a 3x3 rotation and a translation (:class:`pepin.tsdf.RigidPose`)."""

    @property
    def rotation(self) -> Array:
        """The 3x3 rotation."""
        ...

    @property
    def translation(self) -> Array:
        """The translation."""
        ...


class MotionSource(Protocol):
    """Who can say how the cart moved between two moments —
    :meth:`pepin.frame_pose.FramePoser.motion` in a node, a tape offline, a fake in a test."""

    def motion(self, from_stamp: float, to_stamp: float) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp``, or ``None`` when the
        odometry does not cover both moments."""
        ...


@runtime_checkable
class MapMotionSource(Protocol):
    """A motion source that can also answer through the map, where the lidar tracker's
    corrections live (:meth:`pepin.frame_pose.FramePoser.map_motion`) — what the parallax
    anchor asks for a baseline over a whole second. A source without the method is simply an
    odometry-only one; the anchor falls back to it."""

    def map_motion(self, from_stamp: float, to_stamp: float) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp`` through the map, or
        ``None`` when the map pose does not cover both moments (no tracker, or a stale one)."""
        ...


@runtime_checkable
class RecentMapMotionSource(Protocol):
    """A map motion source that can answer WITHOUT waiting — the only kind the frame path may
    ask, because a TF lookup that cannot be answered costs its whole timeout inside the frame.

    Live on 2026-09-15: the blocking ask waited the node's 0.2 s on every frame (map -> base_link
    is not in TF at a frame's stamp yet), the stream fell from 8.7 to 1.5 frames/s and
    rgbd_odometry starved to 0 poses/s. The recent ask uses the newest map pose already held,
    when it is fresh enough to be this frame's, and says no at once when it is not."""

    def map_motion_recent(
        self, from_stamp: float, to_stamp: float, max_age_s: float
    ) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp`` through the map, using only
        poses already held and only when the newest is within ``max_age_s`` of ``to_stamp``;
        ``None`` at once otherwise. Never waits."""
        ...


@runtime_checkable
class RecentMapPoseSource(Protocol):
    """A motion source that can hand out the cart's MAP POSE rather than a motion between two
    stamps, built so that it exists on every frame: the newest ``map <- odom`` composed with
    ``odom <- base_link`` at the frame's own stamp
    (:meth:`pepin.frame_pose.FramePoser.map_pose_recent`). A window of such poses is one
    source by construction — there is no frame on which it falls back to something else and
    nothing to cut — and the motion between any two of them is arithmetic, with no lookup."""

    def map_pose_recent(self, stamp: float) -> Rigid | None:
        """``map <- base_link`` at ``stamp`` from the newest correction and this moment's
        odometry; ``None`` only without a map edge at all. Never waits."""
        ...

    def map_correction_recent(self) -> Rigid | None:
        """The newest ``map <- odom``: the correction itself, which a caller watches for the
        jump a relocalisation puts between the poses it has already stored."""
        ...


@dataclass(frozen=True, eq=False)
class FrameContext:
    """What a frame brings besides the network's depth: the optics, the camera's place on the
    cart, which way is up (base_link), the lidar's returns as base_link points carried to the
    frame's moment (``None`` without a scan), the frame's stamp in seconds — and, for the
    anchors that read the picture rather than a sensor, the frame's grey image, the source of
    the cart's motion between stamps, and TF's whole ``base_link <- camera_optical`` edge, the
    one place a panned neck is carried (all three ``None`` unless a node supplies them)."""

    intr: Intrinsics
    cam: CameraPose
    up: Array = field(default_factory=lambda: UP_LEVEL.copy())
    lidar: Array | None = None
    stamp: float = 0.0
    gray: npt.NDArray[np.uint8] | None = None
    motion: MotionSource | None = None
    cam_optical: Rigid | None = None
    dist: tuple[float, ...] = ()  # the lens the PICTURE still carries (camera_info's d, empty
    # once camera_stream publishes a rectified one). Every projection in this module is a
    # pinhole and stays one; it is the parallax anchor, whose epipolar test is 1.5 px wide,
    # that asks for the pixels straightened before it measures with them.

    @cached_property
    def beams(self) -> Array | None:
        """The lidar's returns as pixels of this frame with their true depth (column, row,
        metres), inside the image only; ``None`` without lidar."""
        if self.lidar is None:
            return None
        return project(self.lidar, self.cam, self.intr)


@dataclass(eq=False)
class Frame:
    """A frame on its way through the pipeline: the network's raw depth, the frame's context,
    the pairs the anchors have contributed so far, and the raw depth's edge mask, computed
    once (relative, so it is the raw image's and every law's alike)."""

    raw: Array
    ctx: FrameContext
    pairs: list[Pairs] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # which anchor each block of pairs is from

    def add(self, source: str, found: Pairs) -> None:
        """Take one anchor's pairs into the pool, remembering whose they are (the report line
        says how much of the frame's weight each ruler brought)."""
        self.pairs.append(found)
        self.sources.append(source)

    @property
    def rulers(self) -> dict[str, float]:
        """How much fit weight each anchor contributed to this frame, by the anchor's name —
        the sum of its pairs' weights, which is the only honest measure of who is fitting the
        law when one ruler brings 30 pairs at weight 1 and another 200 at weight 0.03."""
        out: dict[str, float] = {}
        for name, part in zip(self.sources, self.pairs, strict=False):
            out[name] = out.get(name, 0.0) + float(np.sum(part.weight))
        return out

    @cached_property
    def edge(self) -> Mask:
        """Pixels on a depth discontinuity of the raw depth (:func:`pepin.depth.edge_mask`)."""
        return edge_mask(self.raw)

    @property
    def pool(self) -> Pairs | None:
        """Every pair contributed so far, as one."""
        return Pairs.join(self.pairs)


@dataclass(frozen=True)
class Verdict:
    """What a stage did to one frame, for the report: whether it ran, the pairs it contributed,
    the pixels it changed, a note in its own words, and whether it withholds the frame (a law
    with nothing to apply yet: the raw network's depth must not go out)."""

    stage: str
    on: bool
    pairs: int = 0
    pixels: int = 0
    note: str = ""
    withhold: bool = False


# ---- the roles ----------------------------------------------------------------------------------
class Stage(Protocol):
    """One step of the pipeline: a depth image in, a depth image and a verdict out."""

    name: str

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Do the stage's work on ``depth`` (never in place) and say what was done."""
        ...

    def describe(self) -> str:
        """The stage's state in a few words, for the report line."""
        ...


class Law(Protocol):
    """The map from the network's depth to metres."""

    name: str

    @property
    def ready(self) -> bool:
        """Whether there is a law worth applying."""
        ...

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pooled pairs (``None`` when no anchor had any) and refit."""
        ...

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth image in metres (NaN where the law cannot place a pixel)."""
        ...

    def describe(self) -> str:
        """The law's parameters in a few words."""
        ...


class Anchor(Protocol):
    """An external truth about some pixels of a frame."""

    name: str

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, true) pairs for the laws' pool, from the raw depth; ``None`` for none."""
        ...

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """``depth`` with the pixels this anchor knows set right, and how many."""
        ...

    def describe(self) -> str:
        """The anchor's settings in a few words."""
        ...


class AnchorStage:
    """An anchor as a stage: its pairs join the frame's pool, its correction touches the
    depth. Subclasses fill in :meth:`pairs` and / or :meth:`correct`."""

    name: str = "anchor"

    def pairs(self, frame: Frame) -> Pairs | None:
        """No pairs unless a subclass says otherwise."""
        return None

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """No correction unless a subclass says otherwise."""
        return depth, 0

    def describe(self) -> str:
        """Nothing to say unless a subclass has settings."""
        return ""

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Contribute the pairs, apply the correction."""
        found = self.pairs(frame)
        if found is not None and found.size:
            frame.add(self.name, found)
        out, touched = self.correct(depth, frame)
        return out, Verdict(
            self.name, True, pairs=0 if found is None else found.size, pixels=touched
        )


class LawStage:
    """A law as a stage: fitted on the frame's pool, applied to the depth; the frame is
    withheld while no law exists. Subclasses fill in :meth:`fit`, :meth:`apply`, ``ready``."""

    name: str = "law"

    @property
    def ready(self) -> bool:
        """Whether there is a law worth applying."""
        return False

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pooled pairs and refit."""

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth in metres."""
        return depth

    def describe(self) -> str:
        """The law in a few words."""
        return ""

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool, apply — or withhold the frame while there is no law."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = self.apply(depth, frame.ctx)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())


# ---- the pipeline -----------------------------------------------------------------------------
@dataclass
class StageStats:
    """A stage's running totals: frames it ran on, pairs contributed, pixels changed, and the
    time it took (seconds in all, and the longest frame)."""

    frames: int = 0
    pairs: int = 0
    pixels: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0

    def add(self, verdict: Verdict, seconds: float) -> None:
        """Count one frame's verdict and its time."""
        self.frames += 1
        self.pairs += verdict.pairs
        self.pixels += verdict.pixels
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)

    @property
    def ms(self) -> tuple[float, float]:
        """(mean, max) milliseconds a frame took."""
        return (1e3 * self.seconds / self.frames if self.frames else 0.0, 1e3 * self.max_seconds)


@dataclass(eq=False)
class Result:
    """One frame's way through the pipeline: the final depth, each stage's verdict, the depth
    after every stage that ran (by name; references, the stages never write in place), the
    frame with its pool, and whether a law withheld it (nothing should be published)."""

    depth: Array
    verdicts: list[Verdict]
    after: dict[str, Array]
    frame: Frame
    withheld: bool

    def verdict(self, name: str) -> Verdict:
        """The verdict of the stage called ``name``."""
        for v in self.verdicts:
            if v.stage == name:
                return v
        raise KeyError(name)

    def before(self, name: str) -> Array:
        """The depth as it stood when the stage called ``name`` began: the raw depth when no
        stage ran before it, else the output of the last one that did (the node's scan is
        built from the depth before the floor anchor, so what stops the cart is what was
        measured). ``KeyError`` when the run never reached that stage."""
        out = self.frame.raw
        for v in self.verdicts:
            if v.stage == name:
                return out
            if v.on:
                out = self.after[v.stage]
        raise KeyError(name)


class DepthPipeline:
    """An ordered list of stages, each switchable by name (a node's flags), run on every
    frame with per-stage statistics; a stage switched off is skipped and says so in its
    verdict."""

    def __init__(self, stages: Sequence[Stage], off: Sequence[str] = ()) -> None:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        self._stages = list(stages)
        self._on = dict.fromkeys(names, True)
        self._stats = {n: StageStats() for n in names}
        for name in off:
            self.set(name, False)

    @property
    def names(self) -> list[str]:
        """The stages' names, in running order."""
        return [s.name for s in self._stages]

    def stage(self, name: str) -> Stage:
        """The stage called ``name``."""
        for s in self._stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def set(self, name: str, on: bool) -> None:
        """Switch a stage on or off by name."""
        if name not in self._on:
            raise KeyError(name)
        self._on[name] = bool(on)

    def on(self, name: str) -> bool:
        """Whether the stage called ``name`` runs."""
        return self._on[name]

    @property
    def switches(self) -> dict[str, bool]:
        """Every stage's switch, in running order."""
        return dict(self._on)

    @property
    def stats(self) -> dict[str, StageStats]:
        """Every stage's running totals, by name."""
        return self._stats

    def reset_stats(self) -> None:
        """Start the totals afresh (a report window)."""
        self._stats = {n: StageStats() for n in self._on}

    def run(self, depth: Array, ctx: FrameContext) -> Result:
        """One frame through every stage that is on, in order; stops at a law that withholds."""
        frame = Frame(np.asarray(depth, dtype=float), ctx)
        out = frame.raw
        verdicts: list[Verdict] = []
        after: dict[str, Array] = {}
        for stage in self._stages:
            if not self._on[stage.name]:
                verdicts.append(Verdict(stage.name, False))
                continue
            t0 = time.perf_counter()
            out, verdict = stage.run(out, frame)
            self._stats[stage.name].add(verdict, time.perf_counter() - t0)
            verdicts.append(verdict)
            after[stage.name] = out
            if verdict.withhold:
                return Result(out, verdicts, after, frame, withheld=True)
        return Result(out, verdicts, after, frame, withheld=False)

    def report(self) -> str:
        """One line: every stage, on or off, its own words and its totals."""
        parts = []
        for stage in self._stages:
            st = self._stats[stage.name]
            state = "on" if self._on[stage.name] else "off"
            words = stage.describe()
            mean_ms, max_ms = st.ms
            parts.append(
                f"{stage.name} {state}"
                + (f" [{words}]" if words else "")
                + f" {st.frames} frames, {st.pairs} pairs, {st.pixels} px,"
                f" {mean_ms:.1f}/{max_ms:.1f} ms"
            )
        return "; ".join(parts)


# ---- today's stages ---------------------------------------------------------------------------
class EdgeFilter(AnchorStage):
    """Flying pixels dropped: the pixels on a depth discontinuity of the raw depth
    (:func:`pepin.depth.edge_mask`) become NaN. Off, the mask still keeps the beams off the
    edges (the anchors read ``Frame.edge`` regardless), as the node always did."""

    name = "edge_filter"

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """The depth with its edge pixels NaN, and how many."""
        return drop_edges(depth, frame.edge)

    def describe(self) -> str:
        return f"step {EDGE_REL_STEP:.0%}"


class LidarAnchor(AnchorStage):
    """The lidar's beams: where a beam lands in the image, the network's raw depth pairs with
    the beam's true depth (:func:`pepin.depth.beam_hits`; edge pixels left out). Off, the laws
    get no pairs from the lidar and hold — the failure mode of a lidar that stops, and the
    measure of what the lidar buys.

    A beam weighs a flat 1 by default — the reference pair every other ruler is weighed against
    (:data:`pepin.depth.REF_SIGMA_INV`), so the parallax anchor's corners can share the pool
    with the beams in one unit. ``sigma_m`` above 0 weighs each beam ``1 / sigma^2`` in inverse
    depth instead (``sigma_m / z^2``, :func:`pepin.depth.pair_weight`), which reads as a weight
    proportional to ``z^4`` and measured WORSE on the beams alone (:data:`LIDAR_SIGMA_M`); it is
    kept as the live ``lidar_sigma_m`` knob, not as the default."""

    name = "lidar_anchor"

    def __init__(self, weight: float = 1.0, sigma_m: float = LIDAR_SIGMA_M) -> None:
        self.weight = weight
        self.sigma_m = sigma_m

    def pairs(self, frame: Frame) -> Pairs | None:
        """The beams' pairs, or ``None`` without a scan or under MIN_SAMPLES clean hits."""
        beams = frame.ctx.beams
        if beams is None:
            return None
        hits = beam_hits(frame.raw, beams, frame.edge)
        if hits is None:
            return None
        cols = beams[hits, 0].astype(int)
        rows = beams[hits, 1].astype(int)
        intr = frame.ctx.intr
        z = beams[hits, 2]
        weight: float | Array = self.weight
        if self.sigma_m > 0.0:
            weight = self.weight * pair_weight(inverse_sigma(self.sigma_m, z))
        return Pairs.of(
            frame.raw[rows, cols],
            z,
            lift_of(rows, intr),
            weight,
            left_of(cols, intr),
        )

    def describe(self) -> str:
        """The weight a beam carries: its own 1 / sigma^2 at ``sigma_m`` of range noise, or the
        flat number every beam shared before."""
        if self.sigma_m <= 0.0:
            return f"weight {self.weight:g} flat"
        return f"weight {self.weight:g} / sigma^2, sigma {self.sigma_m * 100:.1f} cm"


class AffineLaw(LawStage):
    """The affine law in inverse depth, 1 / z = a / D + b, fitted on the pairs of the last
    ``pool_frames`` frames (:func:`pepin.depth.fit_affine`, each pair by its weight): a turn's
    worth of pairs spans the room's depths, so the fit is conditioned where one frame's is not;
    a frame without pairs keeps the law. Until POOL_MIN_SAMPLES pairs are pooled there is no
    law worth applying (``ready`` is false: the raw network's depth is 1.5-2x too far and must
    not reach the costmap) — unless a saved law was ``seed``-ed, which holds until the live
    pool can replace it.

    ``slew_per_s`` caps how fast the law may move: the largest relative change of the published
    inverse depth over the pool's own depth range, per second (0 applies every fit whole, which
    is bit for bit :class:`pepin.depth.AffineScale` on unit weights). The pool is a queue of
    frames, not of seconds, so at 9.4 frames/s a 600-frame pool is 64 s deep and half a minute
    of driving replaces half of it; and the shift term switches on and off with the pool's
    depth spread (:data:`pepin.depth.MIN_DEPTH_SPREAD`), so the same pairs are described first
    as a scale alone and then as a scale and a shift. Both were seen on 2026-09-14: a 1.74 b 0
    standing, a 2.33 b -0.200 within 30 s of driving, and back — while the volume kept the
    paint of whichever law was in force."""

    name = "affine_law"

    def __init__(
        self,
        pool_frames: int = POOL_FRAMES,
        slew_per_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.a = 1.0
        self.b = 0.0
        self.frames = 0
        self.held = 0
        self.slew_per_s = slew_per_s
        self._pool: list[Pairs] = []
        self._pool_frames = pool_frames
        self._seeded = False
        self._clock = clock
        self._last_fit: float | None = None  # when the law last moved, for the slew's seconds
        self._asked: tuple[float, float] | None = None  # the fit the slew is still walking to

    def seed(self, a: float, b: float) -> None:
        """Start from a saved law (the map's): applied until POOL_MIN_SAMPLES live pairs exist."""
        self.a, self.b, self._seeded = a, b, True

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists: enough live pairs, or a seed."""
        return self._seeded or self.fitted

    @property
    def fitted(self) -> bool:
        """Whether the law rests on POOL_MIN_SAMPLES live pairs (worth saving)."""
        return self.pooled >= POOL_MIN_SAMPLES

    @property
    def pooled(self) -> int:
        """How many live pairs the pool holds."""
        return int(sum(p.size for p in self._pool))

    @property
    def pool(self) -> Pairs | None:
        """Everything in the pool, as one."""
        return Pairs.join(self._pool)

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pairs (or ``None``); the law is refitted on the pool."""
        self.frames += 1
        if pairs is None or pairs.size == 0:
            self.held += 1
            return
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        if self._seeded and not self.fitted:
            return  # the map's law outranks a fit on a handful of pairs
        pool = self.pool
        assert pool is not None
        self.a, self.b = self._toward(*fit_affine(pool.d, pool.z, pool.weight), pool)

    def _toward(self, a_fit: float, b_fit: float, pool: Pairs) -> tuple[float, float]:
        """The law to hold now, walking toward the fresh fit no faster than ``slew_per_s``.

        The step is measured where it hurts: the largest relative move of the published inverse
        depth over the pool's own depth range (its 5th and 95th percentiles of network depth),
        not in ``a``, because the fit's two parameters trade against each other and agree near
        the pool's middle while they differ at its ends. Over the allowance the law takes the
        blend of old and new that exactly spends it — a blend of two affine laws is affine.
        ``slew_per_s`` at or below zero, or the first live fit (nothing to walk from), applies
        the fit whole."""
        now = self._clock()
        last, self._last_fit = self._last_fit, now
        if self.slew_per_s <= 0.0 or last is None:
            self._asked = None
            return a_fit, b_fit
        step = self._reach(a_fit, b_fit, pool)
        allowed = self.slew_per_s * max(0.0, now - last)
        if step <= allowed:
            self._asked = None
            return a_fit, b_fit
        self._asked = (a_fit, b_fit)
        t = allowed / step
        return self.a + t * (a_fit - self.a), self.b + t * (b_fit - self.b)

    def _reach(self, a_fit: float, b_fit: float, pool: Pairs) -> float:
        """How far the fresh law is from the one in hand: the largest relative change of the
        published inverse depth at the pool's 5th and 95th percentiles of network depth."""
        d = pool.d[np.isfinite(pool.d) & (pool.d > 0.0)]
        if d.size == 0:
            return 0.0
        ends = np.percentile(d, (5, 95))
        old = self.a / ends + self.b
        new = a_fit / ends + b_fit
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(new - old) / np.abs(old)
        return float(np.max(rel[np.isfinite(rel)], initial=0.0))

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through 1 / z = a / D + b."""
        return apply_affine(depth, self.a, self.b)

    def describe(self) -> str:
        """The law for the report line: its two numbers, the pairs behind them, where they came
        from, and — when a parameter sits on its bound — that the pool asked for more than the
        bounds allow, so the number is a ceiling and not a fit."""
        source = "" if self.fitted else " (seed)" if self._seeded else " (none yet)"
        clipped = at_bound(self.a, self.b)
        edge = f" [{clipped} AT BOUND]" if clipped else ""
        asked = ""
        if self._asked is not None:
            asked = f", slewing to a {self._asked[0]:.2f} b {self._asked[1]:+.3f}"
        return f"a {self.a:.2f} b {self.b:+.3f} on {self.pooled} pairs{source}{edge}{asked}"


class FloorGeometry:
    """The floor's expected depth image for the current optics, mount and lean
    (:func:`pepin.depth.floor_depth`), recomputed only when the lean moves by more than
    LEAN_STEP or the optics change — the trigonometry over a whole image is milliseconds the
    frame rate would notice."""

    def __init__(self, lean_step: float = LEAN_STEP) -> None:
        self._lean_step = lean_step
        self._expected: Array | None = None
        self._up: Array | None = None
        self._key: tuple[Intrinsics, CameraPose] | None = None

    def expected(self, ctx: FrameContext) -> Array:
        """Per pixel, the depth its ray would have on the floor (NaN at and above the horizon)."""
        up = np.asarray(ctx.up, dtype=float)
        if (
            self._expected is None
            or self._up is None
            or self._key != (ctx.intr, ctx.cam)
            or float(np.linalg.norm(up - self._up)) > self._lean_step
        ):
            self._expected = floor_depth(ctx.intr, ctx.cam, up)
            self._up = up.copy()
            self._key = (ctx.intr, ctx.cam)
        return self._expected


class FloorAnchor(AnchorStage):
    """Pixels within ``tolerance`` of the floor plane snap to the plane's exact depth
    (:func:`pepin.depth.floor_anchor`), the plane leaning with the cart's up vector."""

    name = "floor_anchor"

    def __init__(
        self, geometry: FloorGeometry | None = None, tolerance: float = FLOOR_HEIGHT_TOLERANCE
    ) -> None:
        self.geometry = geometry if geometry is not None else FloorGeometry()
        self.tolerance = tolerance

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """The depth with its floor pixels on the plane, and how many."""
        ctx = frame.ctx
        return floor_anchor(depth, self.geometry.expected(ctx), ctx.cam.z, self.tolerance)

    def describe(self) -> str:
        return f"tolerance {self.tolerance * 100:.0f} cm"


# ---- the floor as a second hoop ---------------------------------------------------------------
def floor_sigma(z: Array, camera_height: float, sigma_pitch: float, band: DepthNoise) -> Array:
    """How well a floor pixel's depth is known, in metres, at true depth ``z``.

    The geometry is the ruler's: a ray leaving a lens ``camera_height`` above the plane at an
    angle ``t`` below the horizon ends at ``z = h / sin(t)``, so an error in the angle — which
    is what the mount's pitch is uncertain by — moves that depth by ``dz = z^2 / h * dt`` for a
    small ``dt`` (differentiate, and ``cos t -> 1`` at the shallow angles a floor is seen at).
    A ruler whose noise grows as the SQUARE of the range is a different animal from a lidar
    beam's constant centimetres: the floor is a good ruler at a metre and a poor one at three,
    and the weight must say so instead of the flat share every floor pixel used to carry.
    ``sigma_pitch`` is in radians (config/neck.json's ticks_note measures the mount's pitch to
    +-1.5 deg). The network's own band at that depth (:class:`pepin.contact.DepthNoise`) is
    added in quadrature: a floor pixel is called floor because the network's depth put it near
    the plane, so its pair is only as good as that judgement."""
    depth = np.asarray(z, dtype=float)
    tilt = depth**2 / max(camera_height, 1e-6) * sigma_pitch
    network = depth * (band.rel_at_zero + band.rel_per_m * depth)
    out: Array = np.hypot(tilt, network)
    return out


def _unproject(z: Array, rows: Array, cols: Array, ctx: FrameContext) -> Array:
    """The (n, 3) base_link points sitting ``z`` metres along the optical axis on the rays
    through pixels (``rows``, ``cols``): the mount's pitch applied, the lens' place added."""
    lift, left = lift_of(rows, ctx.intr), left_of(cols, ctx.intr)
    c, s = math.cos(ctx.cam.pitch), math.sin(ctx.cam.pitch)
    return np.stack(
        [ctx.cam.x + z * (c + s * lift), ctx.cam.y + z * left, ctx.cam.z + z * (-s + c * lift)],
        axis=1,
    )


def fit_plane(points: Array) -> tuple[Array, Array] | None:
    """The plane those (n, 3) points lie on, as (unit normal, a point on it — their centroid),
    by the smallest eigenvector of their covariance; ``None`` under three points.

    Total least squares: the plane that minimises the PERPENDICULAR distances, which is the
    right estimator for a cloud that is genuinely a surface and the wrong one for a cloud
    selected inside a height band — see :func:`level_plane`."""
    if points.shape[0] < 3:
        return None
    centre = points.mean(axis=0)
    centred = points - centre
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    normal: Array = vectors[:, 0]
    return normal, centre


def level_plane(points: Array) -> tuple[Array, Array] | None:
    """The floor those (n, 3) base_link points draw, as (the plane's own HEIGHT at each of them,
    its unit normal): ``z = a + b x + c y`` by least squares — the one coordinate a downward ray
    leaves open, regressed on the two the ray fixes. ``None`` under three points or where the
    ground positions are collinear.

    Why not :func:`fit_plane`'s total least squares, which this replaces in the floor gate: TLS
    calls "normal" whichever direction the cloud is THINNEST in, and a floor cloud chosen inside
    a height band is a slab — so as soon as the slab's thickness approaches the footprint's own
    depth, the thinnest direction stops being the vertical one. On tape 0318 (a closed door 2 m
    ahead, the floor visible from 1.16 to 2.01 m and the band 12 cm) the TLS plane of pixels that
    are 90 % within 11 cm of the floor leans 50 degrees with a normal pointing nearly FORWARD;
    the same pixels regressed this way lean 38, and with the far pixels cut
    (:data:`FLOOR_BAND_MAX_M`) 12.1 against TLS' 12.4 (scratch/floor_gate_probe.txt,
    2026-09-15). Regressing the height also gives the gate the number it actually wants — how far
    the plane sits from the floor, in metres, at each pixel that voted for it.

    The height is base_link's z, which is the cart's up to within its own lean of a few degrees
    — the same approximation :func:`pepin.depth.floor_anchor` makes when it calls
    ``camera_height * (1 - d / E)`` a height, and the number this one is held against."""
    if points.shape[0] < 3:
        return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ground = np.stack([np.ones(x.shape[0]), x, y], axis=1)
    coefficients, *_ = np.linalg.lstsq(ground, z, rcond=None)
    if not np.isfinite(coefficients).all():
        return None
    drawn: Array = coefficients[0] + coefficients[1] * x + coefficients[2] * y
    normal = np.array([-coefficients[1], -coefficients[2], 1.0])
    unit: Array = normal / float(np.linalg.norm(normal))
    return drawn, unit


class FloorPairs(AnchorStage):
    """The floor pairs the network's depth with the plane's geometric depth: every
    ``stride``-th pixel whose ray meets the floor and that stands within the network's height
    band of the plane (:class:`pepin.contact.DepthNoise`, wider than the anchor's snap: these
    pairs feed a robust fit, not a correction). Which pixels are floor is judged with the law
    as it stands; before any law exists, with a scale bootstrapped from the pixels themselves
    (:meth:`_bootstrap`). So the law can be fitted from the floor alone, with no lidar.

    What that costs, said plainly, because it decides what a lidar-less cart can expect: the
    band is about a tenth of the depth wide, so the pixels a law calls floor are the pixels that
    AGREE with it, and the scale re-taken over them comes back as the scale it was handed — over
    run 0171 the map from assumed scale to fitted scale returns its own input to within 0.01
    everywhere between 0.6 and 2.4 (scratch/_floor_fixed_point.py, 2026-09-15). The floor's
    pixels therefore identify the scale only where they DOMINATE the picture below the horizon
    (a neck pitched down: tape 0237 reads the same 1.76 from every start), and where they do not,
    the answer is the one the bootstrap started in. That is why the bootstrap's start is chosen
    by how many pixels it ends up holding and not by the median, and why a law once fitted is
    never re-bootstrapped from a floor it may have locked onto.

    Each pair carries its OWN weight, not a flat share: the plane's depth under a ray is
    ``h / sin(angle below the horizon)``, so the mount's pitch uncertainty
    (``sigma_pitch_deg``) makes a floor pair's sigma grow as the square of its range
    (:func:`floor_sigma`), and the weight is that sigma against a lidar beam's in inverse depth
    (:func:`pepin.depth.pair_weight`, the unit every other ruler is weighed in). A floor pixel
    at a metre is worth about a hundredth of a beam and one at three metres a fortieth of that
    — which is what a ruler made of geometry and a guessed angle is worth.

    That band is capped at ``band_max_m``, because it grows with the floor's own depth and
    never stops — 12 cm at 2 m, 2.2 m at 90 m — and the rows within half a degree of the
    horizon all sit at such depths. There it has stopped being a test of anything: what stands
    at that bearing in a room is a WALL, and it passes (:data:`FLOOR_BAND_MAX_M`). The cap is a
    cap and not a cut: the far FLOOR keeps its pairs and the depth span the law's shift is
    fitted on, and only what stands metres above it loses them.

    And the frame must prove its floor is a floor: the candidate pixels are back-projected
    through the law as it stands and a plane is fitted to them. With ``plane_band`` (the
    default) that plane is :func:`level_plane`'s and the test is in METRES — the plane must
    stay inside the very height band the pixels were chosen in, at all but the outermost
    twentieth of the pixels that voted for it (:data:`PLANE_OFF_PERCENTILE`). With
    ``plane_band`` off the old test stands: :func:`fit_plane`'s normal within
    ``normal_tol_deg`` of the cart's up vector and the camera's distance to that plane within
    the network's band of the camera's height. Either way, a frame that fails contributes
    nothing at all and says so in the report line: a table top, a ramp, or a law that is wrong
    by a fifth all draw a plane the geometry never meant, and pairs taken off it move every
    node they touch.

    Why the test moved out of degrees, measured on the door tapes of 2026-09-15
    (scratch/floor_gate_probe.txt, scratch/floor_gate_eval.txt): a fixed angle asks for
    something the geometry does not always carry. A door 2 m ahead leaves a floor strip 0.85 m
    deep in view, and the pixels are chosen inside a band 12 cm wide, so the SELECTION itself
    admits any lean up to atan(0.24 / 0.85) = 16 degrees — a 5-degree gate is then a test of
    the law's row bias (the network reads the near floor and the far floor at different scales,
    so the back-projection ramps) and not of the floor. It refused every frame of all four door
    tapes, 90 of 95 frames of tape 0313 and 8 of 12 of run 0171. The band test asks instead
    whether the plane departs from the floor by more than the pixels' own noise allows, so it
    tightens by itself wherever more floor is in view."""

    name = "floor_pairs"

    def __init__(
        self,
        law: Law,
        geometry: FloorGeometry | None = None,
        *,
        stride: int = FLOOR_PAIR_STRIDE,
        band: DepthNoise = DEPTH_NOISE,
        sigma_pitch_deg: float = FLOOR_SIGMA_PITCH_DEG,
        normal_tol_deg: float = FLOOR_NORMAL_TOL_DEG,
        band_max_m: float = FLOOR_BAND_MAX_M,
        plane_band: bool = FLOOR_PLANE_BAND,
    ) -> None:
        self.law = law
        self.geometry = geometry if geometry is not None else FloorGeometry()
        self.stride = stride
        self.band = band
        self.sigma_pitch_deg = sigma_pitch_deg
        self.normal_tol_deg = normal_tol_deg
        self.band_max_m = band_max_m
        self.plane_band = plane_band
        self.frames = 0  # frames whose floor was looked at
        self.gated = 0  # of those, the frames whose plane was not a floor
        self.tilt_deg = 0.0  # the last plane's lean from the up vector
        self.plane_off = 0.0  # the last plane's worst departure from the floor, in its own bands

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, floor) pairs of the frame's floor pixels, each weighed by its own sigma —
        or ``None`` under MIN_SAMPLES of them, and ``None`` when the plane gate refuses."""
        ctx = frame.ctx
        s = self.stride
        expected = self.geometry.expected(ctx)
        raw = frame.raw
        metric = self.law.apply(raw, ctx)[::s, ::s] if self.law.ready else None
        expected, raw, edge = expected[::s, ::s], raw[::s, ::s], frame.edge[::s, ::s]
        rows = np.arange(0, frame.raw.shape[0], s)[:, None] * np.ones_like(raw, dtype=int)
        cols = np.arange(0, frame.raw.shape[1], s)[None, :] * np.ones_like(raw, dtype=int)
        ok = np.isfinite(expected) & np.isfinite(raw) & (raw > NEAR_M) & ~edge
        if int(ok.sum()) < MIN_SAMPLES:
            return None
        band = self.band.height_band(expected, ctx.cam.z)
        if self.band_max_m > 0.0:
            band = np.minimum(band, self.band_max_m)
        if metric is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                started = self._bootstrap(raw / expected, ok, band, ctx.cam.z)
            if started is None:
                return None
            scale, floor = started
            metric = raw / scale
        else:
            with np.errstate(invalid="ignore"):
                height = ctx.cam.z * (1.0 - metric / expected)
            floor = ok & np.isfinite(height) & (np.abs(height) < band)
        if int(floor.sum()) < MIN_SAMPLES:
            return None
        self.frames += 1
        if not self._is_floor(metric[floor], rows[floor], cols[floor], band[floor], ctx):
            self.gated += 1
            return None
        z = expected[floor]
        sigma = floor_sigma(z, ctx.cam.z, math.radians(self.sigma_pitch_deg), self.band)
        return Pairs.of(
            raw[floor],
            z,
            lift_of(rows[floor], ctx.intr),
            pair_weight(inverse_sigma(sigma, z)),
            left_of(cols[floor], ctx.intr),
        )

    def _bootstrap(
        self, ratio: Array, ok: Mask, band: Array, height: float
    ) -> tuple[float, Mask] | None:
        """The floor's own scale before any law exists, and the pixels it calls floor: the same
        three turns of "call floor what this scale puts on the plane, re-take the scale over
        them" the stage has always run, started from each of :data:`FLOOR_BOOTSTRAP_STARTS` and
        kept by whichever start ends up admitting the most pixels. ``None`` when no start keeps
        MIN_SAMPLES of them.

        Why not the median of every candidate, which is where this started: a pixel that sits on
        an OBJECT is nearer than the floor would be on the same ray, so its network-over-plane
        ratio is LOWER than the floor's. The candidates below the horizon are therefore a mixture
        whose clutter lives entirely on the low side, and the median of the mixture is biased low
        by construction — while the turns that follow it cannot recover, because the height band
        is only about a tenth of the depth wide and admits whatever scale it is handed (measured
        on the tapes of 2026-09-15, scratch/_floor_fixed_point.py: over run 0171 the map from
        assumed scale to fitted scale returns its own input to within 0.01 everywhere from 0.6 to
        2.4). The floor is the upper population and the biggest one, which is what these starts
        and this choice between them say: on tape 0236 the median start settles at 1.13 where the
        floor's own basin is at 1.63 and the lidar reads 1.91
        (scratch/_floor_bootstrap_start.py)."""
        with np.errstate(invalid="ignore"):
            starts = np.percentile(ratio[ok], FLOOR_BOOTSTRAP_STARTS)
        best: tuple[float, Mask] | None = None
        for start in starts:
            scale, found = float(start), None
            for _ in range(3):
                with np.errstate(invalid="ignore"):
                    sel = ok & (np.abs(height * (1.0 - ratio / scale)) < band)
                if int(sel.sum()) < MIN_SAMPLES:
                    found = None
                    break
                scale = float(np.median(ratio[sel]))
                found = (scale, sel)
            if found is not None and (best is None or int(found[1].sum()) > int(best[1].sum())):
                best = found
        return best

    def _is_floor(
        self, metric: Array, rows: Array, cols: Array, band: Array, ctx: FrameContext
    ) -> bool:
        """Whether the candidate pixels really lie on the floor: their metric 3D points (the
        law's depth back-projected into base_link) fitted with a plane, and that plane held
        against the floor.

        With ``plane_band`` the plane is :func:`level_plane`'s and the test is the honest one —
        the plane's own height, read at each pixel that voted for it, must stay inside that
        pixel's height band, the same band the pixel was called a candidate by. The departure at
        :data:`PLANE_OFF_PERCENTILE` of the pixels (in bands, so 1.0 is exactly at the edge) is
        kept as ``plane_off``, and it is what the frame is judged on. With ``plane_band``
        off the old test runs instead: :func:`fit_plane`'s normal within ``normal_tol_deg`` of
        up, and the camera's distance to that plane within the band of its height."""
        up = np.asarray(ctx.up, dtype=float)
        up = up / float(np.linalg.norm(up))
        points = _unproject(metric, rows, cols, ctx)
        if self.plane_band:
            drawn = level_plane(points)
            if drawn is None:
                return False
            height, normal = drawn
            self.tilt_deg = math.degrees(math.acos(min(1.0, abs(float(normal @ up)))))
            with np.errstate(divide="ignore", invalid="ignore"):
                off = np.abs(height) / np.maximum(band, 1e-6)
            self.plane_off = float(np.percentile(off[np.isfinite(off)], PLANE_OFF_PERCENTILE))
            return self.plane_off <= 1.0
        plane = fit_plane(points)
        if plane is None:
            return False
        normal, centre = plane
        self.tilt_deg = math.degrees(math.acos(min(1.0, abs(float(normal @ up)))))
        if self.tilt_deg > self.normal_tol_deg:
            return False
        lens = np.array([ctx.cam.x, ctx.cam.y, ctx.cam.z])
        stands = abs(float(normal @ (lens - centre)))  # the camera's own height over that plane
        return abs(stands - float(up @ lens)) <= float(np.median(band))

    def describe(self) -> str:
        """The stage for the report line: its lattice, what the mount's pitch is trusted to,
        how wide its height band may grow, which plane gate is running and how many frames that
        gate refused (with the last plane's own lean, and its worst departure in bands when the
        band gate is the one judging)."""
        widest = (
            f"band <= {100 * self.band_max_m:.0f} cm" if self.band_max_m > 0.0 else "band uncapped"
        )
        gate = (
            f"plane gate band (last off {self.plane_off:.2f} band)"
            if self.plane_band
            else f"plane gate {self.normal_tol_deg:.0f} deg"
        )
        return (
            f"stride {self.stride}, pitch +-{self.sigma_pitch_deg:.1f} deg, {widest}, "
            f"{gate}: {self.gated}/{self.frames} frames out"
            f" (last tilt {self.tilt_deg:.1f} deg)"
        )


# ---- the walls as a third hoop ----------------------------------------------------------------
def _column_mean(values: Array, window: int) -> Array:
    """Each row's mean of ``values`` over ``window`` rows either side, down every column,
    ignoring the NaNs (a row with no finite neighbour comes back NaN). Cumulative sums, so the
    cost does not grow with the window."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    zero = np.zeros((1, values.shape[1]))
    total = np.cumsum(np.vstack([zero, filled]), axis=0)
    count = np.cumsum(np.vstack([zero, finite.astype(float)]), axis=0)
    rows = np.arange(values.shape[0])
    lo = np.clip(rows - window, 0, values.shape[0])
    hi = np.clip(rows + window + 1, 0, values.shape[0])
    n = count[hi] - count[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Array = np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)
    return out


def wall_sigma(
    n_dot_d: Array,
    n_dot_b: Array,
    arm: Array,
    chord: Array,
    height: Array,
    sigma_range: float = WALL_SIGMA_RANGE_M,
    sigma_height: float = WALL_SIGMA_HEIGHT,
) -> Array:
    """How well a walked wall pixel's depth is known, in metres:

        sigma = sqrt( sigma_lidar^2 + (sigma_height * h)^2 )

    where ``h`` is the pixel's height above the lidar's line, ``sigma_lidar`` is what the beams
    themselves are worth at that pixel (below) and ``sigma_height`` is the price of the WORLD
    ASSUMPTION — that the surface the beams hit goes on upwards. That assumption is true of a
    door and a wall and false of a sofa back, a bookshelf, a table and a chair, and no
    measurement in the picture settles it, so the ruler fades with the distance from its
    evidence rather than switching off at a threshold: at the default 0.05 a wall pixel a metre
    above the beam is trusted to 5 cm, which is about a fortieth of a beam's weight at 2 m.
    (The gates still refuse a surface the network says is receding — :meth:`WallAnchor.walk` —
    but they cannot see what the network does not show.)

    The lidar's own half, in one line. The plane stands on a return ``p`` with the unit
    horizontal normal ``n`` fitted from that return's scan neighbours, and the pixel's ray
    leaves the lens along ``d``, so the depth the pixel is paired with is
    ``t = n . (p - lens) / (n . d)``. Two things move it, both of them the same range noise
    ``sigma_range`` (1.5 cm for an LD19):

    * the return SLIDES along its own beam ``b``, which moves the plane by
      ``sigma_range * |n . b|`` — the whole error for a wall met head on, and next to nothing
      for one seen edge-on, where a beam's range error runs along the wall instead of into it;
    * the plane TURNS about that return, because the normal is fitted to a chord of length
      ``chord`` between neighbouring returns whose own ranges are noisy:
      ``sigma_phi = sqrt(2) * sigma_range * |n . b| / chord`` radians. A turn of ``sigma_phi``
      moves the plane by ``|arm| * sigma_phi`` at the pixel, where ``arm`` is the along-wall
      distance between the return and the point the ray meets the plane — zero at the beam's
      own point and growing up the column, which is what makes a wall a WORSE ruler the
      further the pixel stands from the beam that vouches for it.

    Both are displacements of the plane along its normal; the ray converts one into depth by
    dividing by ``n . d`` — the column's bearing, 1 for a ray hitting the wall square and small
    where it grazes it, which is the third thing the sigma must say. So

        sigma_lidar = sigma_range * |n . b| * sqrt(1 + 2 * (arm / chord)^2) / |n . d| .

    ``n_dot_d``, ``arm`` and ``height`` are per (row, return), ``n_dot_b`` and ``chord`` per
    return; the result has the shape they broadcast to. The network's own band does NOT enter,
    unlike the floor's (:func:`floor_sigma`): a floor pixel is called floor because the
    network's depth put it near the plane, so its pair is only as good as that judgement, while
    a wall pixel is called wall by CONTINUITY up the column and its truth is the plane's,
    whatever the network says there.

    What it does not say: the pairs of one column all rest on one beam, so they are correlated
    and the fit would count them as independent (:data:`WALL_COLUMN_SHARE` is the answer to
    that, and :data:`WALL_WEIGHT_CAP` stops any one of them outweighing the beam it came
    from)."""
    turn = np.sqrt(1.0 + 2.0 * (np.asarray(arm, dtype=float) / np.asarray(chord)) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        beams = sigma_range * np.abs(n_dot_b) * turn / np.abs(n_dot_d)
    out: Array = np.hypot(beams, sigma_height * np.abs(np.asarray(height, dtype=float)))
    return out


@dataclass(frozen=True, eq=False)
class WallWalk:
    """Where the lidar's returns were extruded up the picture: the column of every usable
    return, the (rows, returns) mask of the rows walked above it, the geometric depth of the
    vertical surface at every such row and how well that depth is known
    (:func:`wall_sigma`, metres)."""

    cols: npt.NDArray[np.intp]
    walked: Mask
    depth: Array
    sigma: Array

    @property
    def count(self) -> int:
        """How many pixels the walk covered."""
        return int(self.walked.sum())


class WallAnchor(AnchorStage):
    """The lidar's returns extruded up the image: a return and its scan neighbours give the
    local direction of the surface they lie on; the vertical plane through that line is the
    wall, and for the rows above the return in its column — as long as the network's depth
    stays continuous from row to row (a step over ``rel_step`` of itself is another object:
    a chair back ends at its top, a wall goes on) and climbs the column no faster than the
    plane's own depth does (a table top or a seat is continuous with its front but recedes
    a percent a row where a vertical surface moves a tenth of that: the log-depth slopes over
    ``slope_window`` rows may differ by ``slope_tol`` a row) — the pixel's ray meets that
    plane at a depth the geometry knows. With ``pairs`` those (network, wall) pairs are a
    third hoop above the lidar's row; with ``correct`` the walked pixels are set to the
    wall's depth outright.

    "The surface goes on upwards" is a statement about the WORLD, true of a door and a wall
    and false of a sofa, a shelf, a table and a chair, so two more gates stand between a
    return and a pair, and the sigma carries what neither of them can see:

    * the DRIFT gate, ``drift_tol``, which ships OFF (0) and says something worth knowing: the
      integral of that disagreement WOULD catch a slow recession the per-row test cannot, and
      cannot be set to catch it without refusing every real wall, because this network's own
      scale climbs faster up the picture than the recession does
      (:data:`WALL_DRIFT_TOL`). Nothing in the picture separates "the surface leans back" from
      "the network is wrong above the row", which is the error this ruler exists to correct;
    * the CLIMB gate, ``min_walk_m``: a column counts only if it reaches that far above the
      lidar's line undisturbed. The ruler exists for the top of the picture, and a stump that
      dies 20 cm up adds pairs where the beams already speak while carrying the full risk of
      being a chair back;
    * and above all of it :func:`wall_sigma`'s ``sigma_height``, the price of the assumption
      itself: the pair's error bar grows with its height above the line whether or not the
      network shows anything wrong, so the ruler FADES as it leaves its evidence.

    Each pair carries its OWN weight, not a flat share: :func:`wall_sigma` is the beam's
    1.5 cm pushing the plane sideways, the fitted direction turning about the return on a
    chord of two neighbours, the ray's own bearing onto the plane, and that height term. A
    pair just above the beam is worth about a beam (``weight_cap``: it IS that beam, and may
    not outweigh it), one a metre up a fortieth of it, and one on a wall seen edge-on almost
    nothing — and with ``column_share`` the pairs of one column SHARE that single beam's
    weight rather than each carrying it, because that is how many measurements they are. It
    was a flat 0.2 apiece until 2026-09-15."""

    name = "wall_anchor"

    def __init__(
        self,
        *,
        rel_step: float = EDGE_REL_STEP,
        row_stride: int = WALL_ROW_STRIDE,
        sigma_range_m: float = WALL_SIGMA_RANGE_M,
        sigma_height: float = WALL_SIGMA_HEIGHT,
        weight_cap: float = WALL_WEIGHT_CAP,
        column_share: bool = WALL_COLUMN_SHARE,
        max_height: float = WALL_MAX_HEIGHT,
        neighbour_gap: float = WALL_NEIGHBOUR_GAP,
        slope_tol: float = WALL_SLOPE_TOL,
        slope_window: int = WALL_SLOPE_WINDOW,
        drift_tol: float = WALL_DRIFT_TOL,
        min_walk_m: float = WALL_MIN_WALK_M,
        pairs: bool = True,
        correct: bool = False,
    ) -> None:
        self.rel_step = rel_step
        self.row_stride = row_stride
        self.sigma_range_m = sigma_range_m
        self.sigma_height = sigma_height
        self.weight_cap = weight_cap
        self.column_share = column_share
        self.max_height = max_height
        self.neighbour_gap = neighbour_gap
        self.slope_tol = slope_tol
        self.slope_window = slope_window
        self.drift_tol = drift_tol
        self.min_walk_m = min_walk_m
        self.contribute = pairs
        self.correct_pixels = correct

    def walk(self, frame: Frame) -> WallWalk | None:
        """Extrude every usable return up its column; ``None`` without lidar or returns."""
        ctx = frame.ctx
        points = ctx.lidar
        if points is None or points.shape[0] < 3:
            return None
        intr, cam = ctx.intr, ctx.cam
        u, v, forward = project_all(points, cam, intr)
        inside = in_image(u, v, forward, intr)
        # the surface's direction at each return: the chord between its scan neighbours
        xy = points[:, :2]
        prev = np.roll(xy, 1, axis=0)
        nxt = np.roll(xy, -1, axis=0)
        near_prev = np.hypot(*(xy - prev).T) < self.neighbour_gap
        near_next = np.hypot(*(nxt - xy).T) < self.neighbour_gap
        near_prev[0] = near_next[-1] = False  # the roll wraps the scan's ends onto each other
        chord = np.where(
            (near_prev & near_next)[:, None],
            nxt - prev,
            np.where(near_next[:, None], nxt - xy, xy - prev),
        )
        usable = inside & (near_prev | near_next)
        if int(usable.sum()) == 0:
            return None
        idx = np.flatnonzero(usable)
        span = np.linalg.norm(chord[idx], axis=1)  # the chord the direction is fitted on
        tangent = chord[idx] / span[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # horizontal, unit
        # the beam the return came in on (the lidar stands within a centimetre of base_link's
        # origin, config/lidar.json): how much of its range noise pushes the plane sideways
        bearing = xy[idx] / np.maximum(np.linalg.norm(xy[idx], axis=1), 1e-9)[:, None]
        n_dot_b = normal[:, 0] * bearing[:, 0] + normal[:, 1] * bearing[:, 1]
        cols = u[idx].astype(int)
        rows0 = v[idx].astype(int)
        h = frame.raw.shape[0]
        raw = frame.raw[:, cols]  # (h, n): each return's column
        lift = lift_of(np.arange(h), intr)[:, None]
        left = -(cols[None, :] - intr.cx) / intr.fx
        c, s = np.cos(cam.pitch), np.sin(cam.pitch)
        shape = (h, cols.size)
        dx = np.broadcast_to(c + s * lift, shape)
        dz = np.broadcast_to(-s + c * lift, shape)
        dy = np.broadcast_to(left, shape)
        n_dot_d = normal[None, :, 0] * dx + normal[None, :, 1] * dy
        offset = normal[:, 0] * (xy[idx, 0] - cam.x) + normal[:, 1] * (xy[idx, 1] - cam.y)
        # along the wall: where the return stands, and where the ray meets the plane — their
        # difference is the lever arm a turn of the fitted direction acts on (:func:`wall_sigma`)
        along = tangent[:, 0] * (xy[idx, 0] - cam.x) + tangent[:, 1] * (xy[idx, 1] - cam.y)
        tan_dot_d = tangent[None, :, 0] * dx + tangent[None, :, 1] * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = offset[None, :] / n_dot_d
            z_up = cam.z + t * dz
            climbed = z_up - points[idx, 2][None, :]  # height above the lidar's own line
            sigma = wall_sigma(
                n_dot_d,
                n_dot_b,
                along[None, :] - t * tan_dot_d,
                span,
                np.maximum(climbed, 0.0),
                self.sigma_range_m,
                self.sigma_height,
            )
            step = np.abs(raw[:-1] - raw[1:]) / raw[1:]
        bad = ~np.isfinite(raw) | ~np.isfinite(t) | (t <= NEAR_M) | (z_up > self.max_height)
        bad[:-1] |= ~np.isfinite(step) | (step > self.rel_step)
        w = self.slope_window
        if w > 0 and h > 2 * w:
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_d, ln_t = np.log(raw), np.log(t)
                climb_d = (ln_d[: -2 * w] - ln_d[2 * w :]) / (2 * w)  # up the column, per row
                climb_t = (ln_t[: -2 * w] - ln_t[2 * w :]) / (2 * w)
                apart = np.abs(climb_d - climb_t)
            bad[w:-w] |= ~np.isfinite(apart) | (apart > self.slope_tol)
        if self.drift_tol > 0.0:
            bad |= self._drifted(raw, t, rows0, w)
        # rows above the return with no bad row between: the count of bad rows at or above a
        # row, less the count at or above the return, must be zero
        above = np.vstack([np.cumsum(bad[::-1], axis=0)[::-1], np.zeros((1, bad.shape[1]))])
        at_return = above[rows0, np.arange(rows0.size)]
        row = np.arange(h)[:, None]
        walked: Mask = (row < rows0[None, :]) & (above[:h] - at_return[None, :] == 0)
        # and the column must have carried the ruler min_walk_m above the line to count at all
        reached = np.where(walked.any(axis=0), np.argmax(walked, axis=0), rows0)
        with np.errstate(invalid="ignore"):
            far_enough = climbed[reached, np.arange(reached.size)] >= self.min_walk_m
        walked &= walked.any(axis=0) & far_enough
        return WallWalk(cols, walked, t, sigma)

    def _drifted(self, raw: Array, t: Array, rows0: Array, window: int) -> Mask:
        """Where the network's depth has wandered from the plane's by more than ``drift_tol``
        since the return's own row — the integral the per-row slope test is blind to. The log
        ratio is averaged over ``window`` rows either side first, so the network's pixel noise
        does not trip it."""
        with np.errstate(divide="ignore", invalid="ignore"):
            smooth = _column_mean(np.log(raw) - np.log(t), window)
        reference = smooth[rows0, np.arange(rows0.size)]
        drift = np.abs(smooth - reference[None, :])
        out: Mask = ~np.isfinite(drift) | (drift > self.drift_tol)
        return out

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, wall) pairs every ``row_stride`` rows of the walk, each weighed by its own
        sigma against a beam's (:func:`wall_sigma`, :func:`pepin.depth.pair_weight`), capped at
        ``weight_cap`` and, with ``column_share``, divided among the pairs of its column — they
        are one beam, not fifty. ``None`` under MIN_SAMPLES or with ``pairs`` off."""
        if not self.contribute:
            return None
        walk = self.walk(frame)
        if walk is None:
            return None
        rows_all = np.arange(frame.raw.shape[0])[:, None]
        rows0 = walk.walked.shape[0] - 1 - np.argmax(walk.walked[::-1], axis=0)
        strided = walk.walked & ((rows0[None, :] - rows_all) % self.row_stride == 0)
        r, k = np.nonzero(strided)
        if r.size < MIN_SAMPLES:
            return None
        intr = frame.ctx.intr
        z = walk.depth[r, k]
        weight = pair_weight(inverse_sigma(walk.sigma[r, k], z), cap=self.weight_cap)
        if self.column_share:
            weight = weight / np.bincount(k, minlength=walk.cols.size)[k]
        return Pairs.of(
            frame.raw[r, walk.cols[k]],
            z,
            lift_of(r, intr),
            weight,
            left_of(walk.cols[k], intr),
        )

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """With ``correct`` on, the walked pixels at the wall's depth; else the depth as is."""
        if not self.correct_pixels:
            return depth, 0
        walk = self.walk(frame)
        if walk is None:
            return depth, 0
        out = np.asarray(depth, dtype=float).copy()
        r, k = np.nonzero(walk.walked)
        out[r, walk.cols[k]] = walk.depth[r, k]
        return out, int(r.size)

    def describe(self) -> str:
        roles = ("pairs " if self.contribute else "") + (
            "correcting" if self.correct_pixels else ""
        )
        return (
            f"step {self.rel_step:.0%}, slope {self.slope_tol:.1%}/row, drift"
            f" {self.drift_tol:.0%}, climb {self.min_walk_m:.2f} m, sigma"
            f" {self.sigma_range_m * 100:.1f} cm + {self.sigma_height * 100:.0f} cm/m up"
            f"{', one vote a column' if self.column_share else ''},"
            f" {roles.strip() or 'idle'}"
        )


class WallCorrection(WallAnchor):
    """The wall anchor's correcting role as its own stage, after the law: the walked pixels
    are set to the extruded plane's metric depth, which a law run afterwards would scale
    again as if it were the network's (the pairs role stays before the law, where the raw
    depth is)."""

    name = "wall_correct"

    def __init__(self, **settings: float | int | bool) -> None:
        super().__init__(**{**settings, "pairs": False, "correct": True})  # type: ignore[arg-type]


# ---- motion as a hoop, with no lidar at all ---------------------------------------------------
def _base_motion(before: Rigid, after: Rigid) -> tuple[Array, Array]:
    """base_link at ``before`` into base_link at ``after``, from two poses in one fixed frame:
    the rotation and translation :func:`pepin.parallax.camera_motion` asks for, the way
    :meth:`pepin.frame_pose.FramePoser.motion` composes them from a history. Two poses of the
    same fixed frame are all a motion needs — no lookup, and no moment at which the answer
    happens not to exist."""
    back = np.asarray(after.rotation, dtype=float).T
    moved: Array = np.asarray(before.translation, dtype=float) - np.asarray(
        after.translation, dtype=float
    )
    return back @ np.asarray(before.rotation, dtype=float), back @ moved


@dataclass(eq=False)
class PreviousFrame:
    """A frame the parallax anchor keeps in its ring: its grey image, its stamp and where the
    lens sat on the cart when it was taken — the whole ``base_link <- camera_optical``, pan
    included, since the neck may have turned since — plus the describer's reading of the
    picture once it has been asked for, so a frame a dozen tracks reach back through is
    described once and not once per later frame (the flow keeps nothing: it re-reads the
    pixels)."""

    gray: npt.NDArray[np.uint8]
    stamp: float
    place: Rigid
    features: Features | None = None  # filled in by the describer the first time it is asked


class ParallaxAnchor(AnchorStage):
    """The cart's own motion as a source of truth: a corner seen in this frame and in the
    previous one, with the odometry's transform between the two stamps, is two rays with a
    known baseline, and where they meet is a depth in metres (:mod:`pepin.parallax`). No lidar
    is involved and no floor plane is assumed — this is the hoop a camera keeps when every
    other sensor is off, and the only one that measures at every elevation the picture has, so
    it is the pool a law over the ray's angle can be fitted on.

    A corner is a TRACK, not a pair (``track_min_obs``, the node's ``parallax_track_min_obs``,
    3 by default; 2 restores the pair the stage measured until 2026-09-15), and since
    2026-09-15 it is followed FORWARD by default (``tracking``, the node's
    ``parallax_tracking``): detected once and hopped one frame at a time in a
    :class:`pepin.parallax.TrackStore`, which costs two flow calls a FRAME instead of two per
    view, so the window may be as long as the pose deserves. Measured over the four errands of
    2026-09-14 (scratch/parallax_forward_eval.txt): 5.9 ms a frame at a 1.5 s window and 6.4 at
    5 s against the backward build's 31.3 at 1.5 s, the corners' own depth 0.94-1.02 of the
    lidar against the backward build's 0.89 and the pair's 0.79, and a parallax-only law at
    12.7-13.8 % of residual over the frames it fits against 16.0 % and 30.2 %. What it pays is
    corners: 6-12 a frame against the backward build's 39.5, because it follows ``max_tracks``
    of them where the backward build asks the detector for 400 fresh ones every frame, so it
    fits a law at all on 5-9 frames of 69 against 19. ``window`` keeps the old behaviour.

    In the backward ``window`` mode the corners of this frame are followed back through every
    ring frame inside ``track_window_s``
    (:func:`pepin.parallax.build_tracks` — hop by hop for the flow, recognised frame by frame
    for the describer) and all of a track's rays are met in ONE least-squares solve with the
    known camera placements, with a robust pass that drops a track's single worst observation
    (:func:`pepin.parallax.triangulate_tracks`). The point of it is the sigma: the two-view
    formula ``z^2 * sigma_px / (f * b)`` keeps its shape with ``b`` the quadrature sum of the
    views' perpendicular baselines, so four views 5 cm apart are worth one pair at 10 cm, and
    ``min_total_baseline_m`` is asked of that sum rather than of one partner's step. Two
    observations reproduce the pair's own numbers arithmetically, which is what makes 2 the off
    position of the knob rather than a different code path.

    A track can be asked to agree with itself: its older half and its newer half are solved
    separately and it is dropped when they disagree by more than ``split_tol_sigma`` combined
    sigmas (:func:`pepin.parallax._split_gap`, counted as ``split`` in the report line). It
    ships OFF, because it was measured: at the 3 sigma a clean corner never reaches it removes
    1.1 % of the real errands' tracks, leaves the parallax-only law's residual exactly where it
    was, and costs 3.9 ms a frame of the stage's 27.0. It cannot see a point moving along the
    camera's own motion either, which is a degeneracy of monocular geometry and not of the gate.

    The anchor keeps the last second and a half of frames in a ring (grey image, stamp, the
    lens's whole placement on the cart, and the describer's reading of it once asked for) and
    asks :class:`FrameContext`'s motion source for the transform between two stamps. While
    pairing, the partner is not the frame before this one: walking back from the
    newest, it is the first frame inside the gap window whose baseline reaches
    ``min_baseline_m``, and the widest baseline in the window when none does
    (:meth:`_partner`). Pairing with the frame before meant pairing 0.1 s apart, which at a
    cart's 0.2-0.3 m/s is 2 cm of baseline and a 23 cm sigma with most corners rejected for
    too little parallax (live run 2026-09-14 14:12); 10 cm asked for is 0.3-0.5 s back. The
    placement is TF's ``base_link <- camera_optical`` when the node supplies it, because that
    is the only pose carrying the neck's pan and a baseline turned by a pan-free pose points
    the wrong way; without it the mount's pitch stands in. A partner closer in time than
    ``min_gap_s`` has no baseline worth triangulating and one farther back than ``max_gap_s``
    has changed more than the flow follows; a frame while the cart stands still, or turns on
    the spot, yields nothing at all and says which. Each pair carries its own weight: the
    ratio of its inverse-depth variance to a lidar beam's, capped at 1, so a short baseline or
    a badly tracked corner counts for little without being thrown away.

    Measured offline on both legs of the errand of 2026-09-14 at 0.2-0.3 m/s
    (scratch/parallax_baseline_sweep.py, 4700 points matched to the lidar's own ranges, the
    calibrated fx 724.1): the parallax a pair rests on decides its noise, sigma 16.3 cm at
    2 cm of it, 12.2 at 5, 6.9 at 9, 4.1 at 18, and it decides the far field, 0.75 and 0.49 of
    the lidar at 1.5-2 and 2-3 m on a 2 cm baseline against 1.02-1.13 from 5 cm on. What it
    does not decide is a +9 to +13 % offset at 1.0-1.5 m, the same at every baseline — a
    scale-like error still unexplained, and not the range-dependent one reported on 2026-09-12
    (that shape was the thin baseline's skew, not the near field). Under a metre nothing can be
    checked this way: the lidar's plane leaves the bottom of the picture below 1.0 m. The stage
    costs 3-5 ms a frame and ships switched off.

    Who matches the corners is a switch (``matcher``, the node's ``parallax_matcher``) and it
    moves the gap window with it: the flow may look 0.60 s back, the describer 1.5 s, because a
    keypoint is recognised rather than followed. Measured on all four errands of 2026-09-14,
    both matchers on the same frame pairs (scratch/parallax_matcher_sweep.txt): ORB keeps 11
    pairs a frame at 1.0 and 1.5 s where the flow keeps 2 and 0, but at 0.5 s — the gap a 10 cm
    ask lands on at this speed — the flow reads 0.993 of the lidar at 1.5-2 m against ORB's
    1.031, with half the per-pair sigma (13.5 cm against 29.6) at half the cost (3.9 ms a frame
    against 8.3). Past a second of gap both read 1.25-1.45: the odometry's own drift over that
    second, not the matcher's doing. So the flow is the default and the longer window is there
    for a pose that deserves it.

    Which pose that is, is the second switch (``motion_source``, the node's ``parallax_motion``).
    A baseline is a length, and over a second the wheels and gyro do not know one: the same four
    errands re-measured with the motion taken from the lidar tracker's map pose instead
    (scratch/parallax_pose_sweep.txt, 2026-09-14) show the odometry reading 15.8 cm of travel at
    a 1.0 s gap and 25.5 at 1.5 s where the tracker reads 13.9 and 18.6 — and every depth is
    proportional to that length. At 1-2 m the flow reads 1.263 and 1.342 of the lidar on the
    odometry's motion at those two gaps and 1.138 and 0.944 on the tracker's; the describer
    1.347 and 1.506 against 0.951 and 0.968. The over-reading past a second was the baseline all
    along, and ``tracker`` is the default. It is not free of its own: at the 0.5 s gap the anchor
    really pairs across, the tracker's motion reads 0.932 at 1-2 m where the odometry's reads
    1.067 — the two bracket the truth — and the per-pair sigma stays 7-10 cm either way, which
    is why these pairs are still measured against the lidar rather than trusted under it. A
    source with no map to ask (a tape's bare odometry, a cart whose tracker is silent, a stale
    map -> odom TF cannot interpolate at these stamps) falls back to the odometry per window,
    and the report line counts both."""

    name = "parallax_anchor"

    def __init__(
        self,
        *,
        weight: float = PARALLAX_WEIGHT,
        min_gap_s: float = PARALLAX_MIN_GAP_S,
        max_gap_s: float | None = None,
        min_baseline_m: float = PARALLAX_MIN_BASELINE_M,
        matcher: str = PARALLAX_MATCHER,
        motion_source: str = PARALLAX_MOTION,
        map_wait: bool = PARALLAX_MAP_WAIT,
        map_max_age_s: float = PARALLAX_MAP_MAX_AGE_S,
        track_min_obs: int = PARALLAX_TRACK_MIN_OBS,
        track_window_s: float = PARALLAX_TRACK_WINDOW_S,
        min_total_baseline_m: float = PARALLAX_MIN_TOTAL_BASELINE_M,
        sigma_model: str = PARALLAX_SIGMA_MODEL,
        track_max_views: int = PARALLAX_TRACK_MAX_VIEWS,
        split_tol_sigma: float = PARALLAX_SPLIT_TOL_SIGMA,
        tracking: str = PARALLAX_TRACKING,
        correction_tol_m: float = PARALLAX_CORRECTION_TOL_M,
        undistort: bool = PARALLAX_UNDISTORT,
        max_tracks: int = PARALLAX_MAX_TRACKS,
        redetect_every: int = PARALLAX_REDETECT_EVERY,
        verify_every: int = PARALLAX_VERIFY_EVERY,
        drift_tol_px: float = PARALLAX_DRIFT_TOL_PX,
    ) -> None:
        self.weight = weight
        self.min_gap_s = min_gap_s
        self._max_gap_s = max_gap_s  # None: whatever window the matcher in use can carry
        self.min_baseline_m = min_baseline_m
        self.matcher = matcher
        self.motion_source = motion_source  # live: the node's parallax_motion flag writes it
        self.map_wait = map_wait  # live: parallax_map_wait — the old blocking ask, for an A/B
        self.map_max_age_s = map_max_age_s
        self.track_min_obs = track_min_obs  # live: parallax_track_min_obs; 2 is the old pair
        self.track_window_s = track_window_s  # live: parallax_track_window_s
        self.min_total_baseline_m = min_total_baseline_m  # live: parallax_min_total_baseline_m
        self.track_max_views = track_max_views  # views one track may rest on: the cost per frame
        self.sigma_model = sigma_model  # what a track's sigma is: the solve's covariance, or
        # the closed form sqrt(sum b^2) the stage shipped with (pepin.parallax.TRACK_SIGMA_MODELS)
        self.split_tol_sigma = split_tol_sigma  # live: parallax_split_tol_sigma; 0 is off
        self.tracking_mode = tracking  # live: parallax_tracking — forward, window or pair
        self.correction_tol_m = correction_tol_m  # live: parallax_correction_tol_m; 0 is off
        self.undistort = undistort  # live: parallax_undistort
        self.max_tracks = max_tracks  # live: parallax_max_tracks (the forward store's cap)
        self.redetect_every = redetect_every  # live: parallax_redetect_every, in frames
        self.verify_every = verify_every  # live: parallax_verify_every, in frames; 0 is off
        self.drift_tol_px = drift_tol_px  # live: parallax_drift_tol_px
        self.stale = 0  # frames whose map pose was too old (or absent) and fell back to odometry
        self.used: dict[str, int] = dict.fromkeys(PARALLAX_MOTIONS, 0)  # who gave each baseline
        self.frames = 0  # frames that reached the triangulation
        self.contributed = 0  # of those, the ones that gave at least one pair
        self.rejected: dict[str, int] = {}
        self._ring: deque[PreviousFrame] = deque(maxlen=PARALLAX_RING_FRAMES)
        self._store: TrackStore | None = None  # the forward ruler's live corners
        self._last_stamp: float | None = None  # the newest frame the store has seen
        self._pose: Rigid | None = None  # where the cart stood at this frame, when tf can say
        self._correction: Rigid | None = None  # the newest map <- odom, to notice it jumping
        self._lens_for: tuple[Intrinsics, tuple[float, ...]] | None = None
        self._lens_is: Lens | None = None
        self.deaths: dict[str, int] = {}  # forward tracks closed, by cause (lk, fb, edge, drift)
        self._live: list[int] = []  # corners the store was following, per frame
        self._born: list[int] = []
        self._gone: list[int] = []
        self._hop_ms: list[float] = []
        self._detect_ms: list[float] = []
        self._verify_ms: list[float] = []
        self._rest_ms: list[float] = []
        self._solve_ms: list[float] = []
        self._baseline: list[float] = []
        self._sigma: list[float] = []
        self._sigma_two: list[float] = []  # the same corners read as the widest pair alone
        self._obs: list[float] = []  # observations a track rests on, median per frame
        self._gap: list[float] = []
        self._kept: list[int] = []

    @property
    def mode(self) -> str:
        """Which ruler measures this frame: ``forward`` (a corner detected once and followed
        one hop a frame, :class:`pepin.parallax.TrackStore`), ``window`` (the current frame's
        corners re-tracked backwards through the ring every frame,
        :func:`pepin.parallax.build_tracks`) or ``pair`` (this frame against one partner, what
        the stage did until 2026-09-15). ``track_min_obs`` under 3 is the pair whatever the
        switch says: a corner seen twice IS a pair, arithmetically."""
        return "pair" if self.track_min_obs < 3 else self.tracking_mode

    @property
    def tracking(self) -> bool:
        """Whether a corner is a TRACK (forward or through the window) or the PAIR of this
        frame and one chosen partner."""
        return self.mode != "pair"

    @property
    def window_s(self) -> float:
        """How far back this frame reaches for a partner or for a track's oldest view: the
        track window while tracking, the matcher's own gap window while pairing."""
        return self.track_window_s if self.tracking else self.max_gap_s

    @property
    def max_gap_s(self) -> float:
        """How far back a partner frame may sit, in seconds: the window the constructor was
        given, or the one the matcher in use can carry — 0.60 s for the flow, which loses three
        corners in four by 1.5 s, and 1.5 s for the describer, which recognises a keypoint
        instead of following it."""
        if self._max_gap_s is not None:
            return self._max_gap_s
        return PARALLAX_ORB_MAX_GAP_S if self.matcher == "orb" else PARALLAX_MAX_GAP_S

    @max_gap_s.setter
    def max_gap_s(self, seconds: float | None) -> None:
        """Pin the window to a number of seconds, or to ``None`` to let the matcher set it."""
        self._max_gap_s = seconds

    @property
    def sigma_m(self) -> float | None:
        """The per-pair triangulation noise of the last frames, metres (their median), or
        ``None`` before the first pair — the number every pair's weight is 1 / sigma^2 of, in
        inverse depth (:func:`pepin.depth.pair_weight`), and the one to read against the
        lidar's ``lidar_sigma_m`` when the two rulers disagree."""
        return float(np.median(self._sigma)) if self._sigma else None

    @property
    def gap_s(self) -> float | None:
        """How far back in time the last triangulated frame reached, in seconds — its partner
        while pairing, its oldest track view while tracking — or ``None`` before the first one.
        The number the report line's span is the median of."""
        return self._gap[-1] if self._gap else None

    def _count(self, reason: str, n: int = 1) -> None:
        """Tally one cause of a lost pair or a lost frame, for the report line."""
        if n:
            self.rejected[reason] = self.rejected.get(reason, 0) + n

    def _moved(
        self, ctx: FrameContext, then: float, now: float, ask_tracker: bool
    ) -> tuple[Rigid | None, str]:
        """How the cart moved between two frame stamps and whose word it is: the tracker's map
        pose while ``ask_tracker`` and the source can answer through the map
        (:class:`MapMotionSource`), the odometry otherwise — the fallback a node needs when the
        tracker is silent or its map pose too stale for TF to interpolate at these stamps.

        ``ask_tracker`` is the caller's memory of that fallback within one walk of the ring: a
        TF lookup that cannot be answered costs its whole timeout, and a dead tracker must cost
        it once a frame rather than once a candidate."""
        source = ctx.motion
        if source is None:
            return None, ""
        if ask_tracker:
            through_map = self._through_map(source, then, now)
            if through_map is not None:
                return through_map, "tracker"
            self.stale += 1
        return source.motion(then, now), "odom"

    def _through_map(self, source: MotionSource, then: float, now: float) -> Rigid | None:
        """The tracker's word on the motion, asked the way ``map_wait`` says: the non-blocking
        ask of the newest map pose within ``map_max_age_s`` of the frame (the default), or the
        old ask that waits for TF to cover the frame's stamp. ``None`` when the source cannot
        answer at all — a tape's bare odometry, a silent tracker, a map pose too old."""
        if not self.map_wait and isinstance(source, RecentMapMotionSource):
            return source.map_motion_recent(then, now, self.map_max_age_s)
        if self.map_wait and isinstance(source, MapMotionSource):
            return source.map_motion(then, now)
        return None

    def _partner(self, ctx: FrameContext, place: Rigid) -> tuple[PreviousFrame, Motion] | None:
        """The frame of the ring this one is paired with and the camera motion between them:
        walking back from the newest, the first partner inside the gap window whose baseline
        reaches ``min_baseline_m``, and the widest baseline in the window when none does.

        Newest-first is the point of it: the shortest gap that carries the asked-for parallax
        is the one whose view has changed least, so the flow still tracks. A cart at 0.25 m/s
        reaches 10 cm about 0.4 s back and pairs across four frames; a cart standing still
        never reaches it and pairs with the widest it has, as the stage always did."""
        from pepin.parallax import camera_motion

        best: tuple[PreviousFrame, Motion] | None = None
        spoke = ""
        candidates = 0
        ask_tracker = self.motion_source in ("tf", "tracker")
        for previous in reversed(self._ring):
            gap = ctx.stamp - previous.stamp
            if gap < self.min_gap_s:
                continue
            if gap > self.max_gap_s:
                break
            candidates += 1
            moved, source = self._moved(ctx, previous.stamp, ctx.stamp, ask_tracker)
            ask_tracker &= source != "odom"  # a silent tracker is discovered once, not per frame
            if moved is None:
                continue
            motion = camera_motion(moved.rotation, moved.translation, previous.place, place)
            if best is None or motion.baseline > best[1].baseline:
                best, spoke = (previous, motion), source
            if motion.baseline >= self.min_baseline_m:
                break
        if best is None:
            if self._ring:
                self._count("gap" if candidates == 0 else "no odometry")
            return None
        self.used[spoke] = self.used.get(spoke, 0) + 1
        return best

    def _window(self, ctx: FrameContext, place: Rigid) -> list[tuple[PreviousFrame, Motion]]:
        """Every frame of the ring a track may reach back through, OLDEST first, each with the
        camera motion from its optical frame into this frame's — what
        :func:`pepin.parallax.build_tracks` wants. The walk is newest-first (so a silent
        tracker is discovered once a frame, as :meth:`_partner`'s is) and stops at
        ``track_window_s``; a window a motion source cannot answer simply loses that view, and
        the frame is triangulated on the ones it can.

        Every view of ONE window comes from one motion source. A pair chose a single partner and
        could not mix; a track meets a dozen rays in one solve, and the two sources disagree by
        about a quarter over a second (scratch/parallax_pose_sweep.txt: 25.5 cm of odometry
        against the tracker's 18.6 over 1.5 s), so a bundle half measured by each is not a
        geometry at all. The walk is newest-first and the tracker can only fall silent as it
        goes back, so the window ends where the source changes — a tracker silent from the start
        still gives the whole window on the odometry, which is the fallback that matters."""
        from pepin.parallax import camera_motion

        out: list[tuple[PreviousFrame, Motion]] = []
        spoke = ""
        candidates = 0
        ask_tracker = self.motion_source in ("tf", "tracker")
        for previous in reversed(self._ring):
            gap = ctx.stamp - previous.stamp
            if gap < self.min_gap_s:
                continue
            if gap > self.track_window_s:
                break
            candidates += 1
            moved, source = self._moved(ctx, previous.stamp, ctx.stamp, ask_tracker)
            ask_tracker &= source != "odom"  # a silent tracker is discovered once, not per frame
            if moved is None:
                continue
            if spoke and source != spoke:
                self._count("mixed motion")  # windows cut short where the source changed
                break
            spoke = spoke or source
            out.append(
                (previous, camera_motion(moved.rotation, moved.translation, previous.place, place))
            )
        if not out:
            if self._ring:
                self._count("gap" if candidates == 0 else "no odometry")
            return out
        self.used[spoke] = self.used.get(spoke, 0) + 1
        out.reverse()
        return out

    def _forward_store(self, ctx: FrameContext) -> TrackStore:
        """The forward ruler's store, with every live knob pushed into it. The knobs are pushed
        on EVERY frame and nothing is reset by pushing them, which is what makes a window
        lengthened or shortened live take effect on the next frame with no restart: the store
        simply uses more or fewer of the observations it already holds. Only a change of
        matcher builds a new store, the flow and the describer keeping different things about a
        corner."""
        from pepin.parallax import TrackStore

        store = self._store
        if store is None or store.matcher != self.matcher:
            store = TrackStore(matcher=self.matcher)
            self._store = store
        store.window_s = self.track_window_s
        store.max_views = self.track_max_views
        store.max_tracks = self.max_tracks
        store.redetect_every = self.redetect_every
        store.verify_every = self.verify_every
        store.drift_tol_px = self.drift_tol_px
        store.lens = self._lens(ctx)
        return store

    def _lens(self, ctx: FrameContext) -> Lens | None:
        """The lens the tracked pixels are straightened with, or ``None`` when they are not:
        ``parallax_undistort`` off, or a picture that carries no distortion (camera_stream
        publishing a rectified one, or an uncalibrated camera). Built once per optics rather
        than per frame — it is a dataclass over the same two numbers every frame."""
        from pepin.parallax import PlumbBob

        if not self.undistort or not ctx.dist or not any(ctx.dist):
            return None
        key = (ctx.intr, ctx.dist)
        if self._lens_for != key:
            self._lens_for, self._lens_is = key, PlumbBob(ctx.intr, tuple(ctx.dist))
        return self._lens_is

    def _source(self, ctx: FrameContext) -> tuple[str, Rigid | None]:
        """Whose word this frame's motion is, asked ONCE a frame, and — where that word is a
        POSE rather than a motion — the pose itself.

        ``tf`` (the default) asks :class:`RecentMapPoseSource` for the map pose TF can build on
        every frame: the newest ``map <- odom`` composed with this moment's ``odom <-
        base_link``. It is the answer to the thing that broke a long window live on 2026-09-15:
        the tracker's own ``map <- base_link`` covers a frame's stamp only sometimes (821
        fallbacks over a 30 s drive), a bundle may not span two sources, and so every fallback
        cut the window — a 5 s window never reached past 2.1 s and the store counted 16718 cut
        bundles. Split in two, the slow half asked for its newest value and the fast half for
        this exact moment, there is one source on every frame and nothing to cut.

        ``tracker`` is the older ask (:meth:`_through_map`) and ``odom`` the wheels; both
        answer with a motion between two stamps, so they keep the cut. A ``tf`` frame the map
        cannot answer for at all — no ``map -> odom`` in TF, a tape with bare odometry — falls
        back to the odometry and is counted, and the cut then applies to it as it always did."""
        if self.motion_source == "tf":
            source = ctx.motion
            pose = (
                source.map_pose_recent(ctx.stamp)
                if isinstance(source, RecentMapPoseSource)
                else None
            )
            if pose is not None:
                return "tf", pose
            self.stale += 1
            return "odom", None
        if self.motion_source != "tracker":
            return "odom", None
        reference = self._last_stamp if self._last_stamp is not None else ctx.stamp
        moved, spoke = self._moved(ctx, reference, ctx.stamp, True)
        return (spoke if moved is not None else "odom"), None

    def _corrected(self, ctx: FrameContext) -> bool:
        """Whether the tracker has just moved the map under the poses already stored: the
        newest ``map <- odom`` against the one seen last frame, read as the metres it puts on a
        point :data:`PARALLAX_CORRECTION_REACH_M` ahead so a turn of the map counts as well as
        a shift. A jump over ``correction_tol_m`` invents a displacement the camera never made,
        and every pose stored before it is no longer comparable with every pose after it."""
        source = ctx.motion
        if not isinstance(source, RecentMapPoseSource):
            return False
        now = source.map_correction_recent()
        if now is None:
            return False
        before, self._correction = self._correction, now
        if before is None or self.correction_tol_m <= 0:
            return False
        shift = float(np.linalg.norm(np.asarray(now.translation) - np.asarray(before.translation)))
        spun = np.asarray(before.rotation, dtype=float).T @ np.asarray(now.rotation, dtype=float)
        turn = math.acos(float(np.clip(0.5 * (float(np.trace(spun)) - 1.0), -1.0, 1.0)))
        return shift + PARALLAX_CORRECTION_REACH_M * turn > self.correction_tol_m

    def _motions(
        self, ctx: FrameContext, place: Rigid, views: Sequence[FrameView], source: str
    ) -> dict[float, Motion]:
        """The transform from each view's optical frame into this frame's camera, keyed by the
        view's stamp — what :meth:`pepin.parallax.TrackStore.tracks` turns into a bundle.

        With ``tf`` every view carries the pose the cart stood at when it was taken, so this is
        arithmetic: one transform per view, no lookup, nothing that can fail on some views and
        not others. The price is written into the pose itself — a view's pose is the tracker's
        estimate AS OF THAT FRAME, so a correction landing inside the window moves the older
        views by the correction; :meth:`_corrected` watches for one big enough to matter.

        With ``tracker`` or ``odom`` there is no pose, and each view's motion is asked of the
        source between the two stamps. Asked newest first, as :meth:`_window`'s walk is, so a
        silent tracker costs its lookup once a frame and not once a view; the walk stops where
        the source changes, because the two disagree by about a quarter over a second and a
        bundle half measured by each is not a geometry. A view the source cannot answer for at
        all is left out of the map and the tracks that wanted it lose that one observation."""
        from pepin.parallax import camera_motion

        out: dict[float, Motion] = {}
        here = self._pose
        if source == "tf" and here is not None:
            for view in views:
                if view.pose is None:
                    continue
                rotation, translation = _base_motion(view.pose, here)
                out[view.stamp] = camera_motion(rotation, translation, view.place, place)
            return out
        ask_tracker = source == "tracker"
        for view in reversed(views):
            moved, spoke = self._moved(ctx, view.stamp, ctx.stamp, ask_tracker)
            ask_tracker &= spoke != "odom"
            if moved is None:
                continue
            if spoke != source:
                self._count("mixed motion")
                break
            out[view.stamp] = camera_motion(moved.rotation, moved.translation, view.place, place)
        return out

    def _forward_truth(
        self, ctx: FrameContext, gray: npt.NDArray[np.uint8], place: Rigid
    ) -> tuple[ParallaxTruth, float] | None:
        """This frame through the forward ruler: hop every live corner into it, ask the motion
        of the views the store wants, meet each track's rays and gate the result. Returns the
        truth and the span in seconds its oldest view reaches back over, or ``None`` when there
        is no view with a motion behind it yet."""
        from pepin.parallax import gate_tracks

        store = self._forward_store(ctx)
        source, self._pose = self._source(ctx)
        if self._corrected(ctx):
            self._count("correction", store.forget_views())
        report = store.follow(gray, ctx.stamp, place, source, self._pose)
        self._last_stamp = ctx.stamp
        self._live.append(report.live)
        self._born.append(report.born)
        # 'source' is a bundle cut short and not a corner closed: it is counted apart
        self._gone.append(sum(n for cause, n in report.died.items() if cause != "source"))
        for cause, count in report.died.items():
            if count:
                self.deaths[cause] = self.deaths.get(cause, 0) + count
        self._hop_ms.append(report.hop_ms)
        self._detect_ms.append(report.detect_ms)
        self._verify_ms.append(report.verify_ms)
        self._rest_ms.append(report.rest_ms)
        views = store.views()
        if not views:
            self._count("gap")
            return None
        motions = self._motions(ctx, place, views, source)
        if not motions:
            self._count("no odometry")
            return None
        self.used[source] = self.used.get(source, 0) + 1
        started = time.perf_counter()
        truth = gate_tracks(
            store.tracks(motions),
            ctx.intr,
            matcher=self.matcher,
            min_obs=self.track_min_obs,
            min_total_baseline_m=self.min_total_baseline_m,
            sigma_model=self.sigma_model,
            split_tol_sigma=self.split_tol_sigma,
        )
        self._solve_ms.append(1000.0 * (time.perf_counter() - started))
        return truth, ctx.stamp - min(motions)

    def _remember(self, ctx: FrameContext, gray: npt.NDArray[np.uint8], place: Rigid) -> None:
        """Put this frame in the ring and drop the frames no later frame can reach: the track
        window while tracking, the matcher's gap window while pairing. The forward ruler keeps
        no ring at all — it needs the previous grey and one grey per view, not the window."""
        while self._ring and ctx.stamp - self._ring[0].stamp > self.window_s:
            self._ring.popleft()
        self._ring.append(PreviousFrame(gray, ctx.stamp, place))

    def _track_truth(
        self,
        ctx: FrameContext,
        gray: npt.NDArray[np.uint8],
        views: list[tuple[PreviousFrame, Motion]],
    ) -> tuple[ParallaxTruth, float]:
        """This frame's corners as tracks through ``views``, and the span in seconds the oldest
        of them reaches back over. The describer's reading of each ring frame is kept on the
        frame, so a window of a dozen views costs one description each and not one per frame."""
        from pepin.parallax import track_truth

        grays = [previous.gray for previous, _ in views] + [gray]
        motions = [motion for _, motion in views]
        features: list[Features | None] = [previous.features for previous, _ in views] + [None]
        truth = track_truth(
            grays,
            motions,
            ctx.intr,
            matcher=self.matcher,
            features=features if self.matcher == "orb" else None,
            min_obs=self.track_min_obs,
            min_total_baseline_m=self.min_total_baseline_m,
            max_views=self.track_max_views,
            sigma_model=self.sigma_model,
            split_tol_sigma=self.split_tol_sigma,
        )
        if self.matcher == "orb":
            for (previous, _), found in zip(views, features, strict=False):
                previous.features = found
        return truth, ctx.stamp - views[0][0].stamp

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, triangulated) pairs at the corners of this frame: each corner followed
        back through the window of ring frames and triangulated from every view it was seen in
        (``track_min_obs`` 3 or more), or shared with the one partner frame chosen out of the
        ring (2, what the stage did until 2026-09-15). ``None`` when there is no usable motion
        behind this frame, or when nothing survived the gates."""
        from pepin.parallax import CameraPlacement, parallax_truth

        ctx = frame.ctx
        gray = ctx.gray
        if gray is None:
            self._count("no image")
            return None
        # TF's edge when the node read one: it is the only pose that carries the neck's pan,
        # and a baseline built from a pan-free pose points the wrong way (pepin.parallax).
        place: Rigid = (
            ctx.cam_optical if ctx.cam_optical is not None else CameraPlacement.of(ctx.cam)
        )
        if self.mode == "forward":
            measured = self._forward_truth(ctx, gray, place)
            if measured is None:
                return None
            truth, span = measured
        elif self.tracking:
            views = self._window(ctx, place)
            self._remember(ctx, gray, place)
            if not views:
                return None
            truth, span = self._track_truth(ctx, gray, views)
        else:
            chosen = self._partner(ctx, place)
            self._remember(ctx, gray, place)
            if chosen is None:
                return None
            previous, motion = chosen
            truth = parallax_truth(previous.gray, gray, ctx.intr, motion, matcher=self.matcher)
            span = ctx.stamp - previous.stamp
        self.frames += 1
        for reason, n in truth.rejected.items():
            self._count(reason, n)
        if truth.kept == 0:
            self._count(truth.verdict or "none")
            return None
        cols = np.rint(truth.points[:, 0]).astype(int)
        rows = np.rint(truth.points[:, 1]).astype(int)
        d = frame.raw[rows, cols]
        ok = np.isfinite(d) & (d > NEAR_M) & ~frame.edge[rows, cols]
        self._count("edge", int((~ok).sum()))
        if not bool(ok.any()):
            return None
        self.contributed += 1
        self._baseline.append(float(np.median(truth.baseline[ok])))
        self._sigma.append(float(np.median(truth.sigma[ok])))
        self._gap.append(span)
        self._kept.append(int(ok.sum()))
        if truth.observations is not None:
            self._obs.append(float(np.median(truth.observations[ok])))
        if truth.sigma_two is not None:
            self._sigma_two.append(float(np.median(truth.sigma_two[ok])))
        del self._baseline[:-POOL_FRAMES], self._sigma[:-POOL_FRAMES], self._gap[:-POOL_FRAMES]
        del self._kept[:-POOL_FRAMES], self._obs[:-POOL_FRAMES], self._sigma_two[:-POOL_FRAMES]
        del self._live[:-POOL_FRAMES], self._born[:-POOL_FRAMES], self._gone[:-POOL_FRAMES]
        del self._hop_ms[:-POOL_FRAMES], self._detect_ms[:-POOL_FRAMES]
        del self._verify_ms[:-POOL_FRAMES], self._solve_ms[:-POOL_FRAMES]
        del self._rest_ms[:-POOL_FRAMES]
        return Pairs.of(
            d[ok],
            truth.z[ok],
            lift_of(rows[ok], ctx.intr),
            self.weight * truth.weight[ok],
            left_of(cols[ok], ctx.intr),
        )

    def describe(self) -> str:
        """The verdict for the report line: who matched the corners and how far back it may
        look, whether a corner is a track or a pair and how many views one must be seen in,
        whose motion the baseline came from and how many windows each source actually answered,
        the parallax asked for, the frames that triangulated, the span and the effective
        baseline they rest on, the corners a frame yields and their sigma — and, while
        tracking, the observations a track carries and what the same corners' sigma would have
        been read as the widest single pair, which is the whole point of the change."""
        spoke = ", ".join(f"{k} {v}" for k, v in self.used.items() if v)
        shape = (
            f" {self.mode} >= {self.track_min_obs} obs over <= {self.track_max_views} views,"
            f" asks {self.min_total_baseline_m * 100:.0f} cm total"
            f", sigma from the {self.sigma_model}"
            + (
                f", halves within {self.split_tol_sigma:g} sigma"
                if self.split_tol_sigma > 0
                else ", halves unchecked"
            )
            + (
                f", <= {self.max_tracks} corners, detect every {self.redetect_every}"
                + (", lens undone" if self.undistort else ", raw pixels")
                + (
                    f", drift <= {self.drift_tol_px:.1f} px every {self.verify_every}"
                    if self.verify_every > 0 and self.drift_tol_px > 0
                    else ", drift unchecked"
                )
                if self.mode == "forward"
                else ""
            )
            if self.tracking
            else f", asks {self.min_baseline_m * 100:.0f} cm"
        )
        carried = (
            f" (correction gate {self.correction_tol_m * 100:.0f} cm)"
            if self.motion_source == "tf" and self.correction_tol_m > 0
            else " (correction ungated)"
            if self.motion_source == "tf"
            else ""
        )
        whose = {
            "tf": "TF's continuous map pose",
            "tracker": "the tracker's motion",
            "odom": "the odometry's motion",
        }.get(self.motion_source, f"the {self.motion_source}'s motion")
        asked = (
            f"{self.matcher} <= {self.window_s:.2f} s{shape},"
            f" on {whose}{carried} ({spoke or 'none yet'}),"
            f" weight {self.weight:g} / sigma^2"
            + (f", map pose stale -> odom {self.stale}" if self.stale else "")
            + (" [map_wait: the frame path WAITS for TF]" if self.map_wait else "")
        )
        dropped = ", ".join(f"{k} {v}" for k, v in self.rejected.items())
        if not self._baseline:
            return f"{asked}{self._store_line()}, no pairs yet" + (
                f" ({dropped})" if dropped else ""
            )
        what = "tracks" if self.tracking else "pairs"
        gained = (
            f" ({np.median(self._obs):.1f} obs a track,"
            f" 2-view sigma {np.median(self._sigma_two) * 100:.1f} cm)"
            if self._obs and self._sigma_two
            else ""
        )
        return (
            f"{asked}{self._store_line()}, {self.contributed}/{self.frames} frames,"
            f" span {np.median(self._gap):.2f} s,"
            f" baseline {np.median(self._baseline) * 100:.1f} cm,"
            f" {np.median(self._kept):.0f} {what} a frame,"
            f" sigma {np.median(self._sigma) * 100:.1f} cm{gained}"
            + (f", rejected: {dropped}" if dropped else "")
        )

    def _store_line(self) -> str:
        """The forward ruler's own half of the report line: the corners alive right now, how
        many are born and closed in a frame and of what, and where the milliseconds went — the
        hop (one flow call for every corner and one back), the detector, the drift bound and
        the solve. Empty while the stage is not following corners forward."""
        if self.mode != "forward" or not self._live:
            return ""
        killed = ", ".join(f"{k} {v}" for k, v in self.deaths.items() if v and k != "source")
        cut = self.deaths.get("source", 0)
        return (
            f", {self._live[-1]} live corners,"
            f" +{np.median(self._born):.0f}/-{np.median(self._gone):.0f} a frame"
            + (f" ({killed})" if killed else "")
            + (f", source cut {cut}" if cut else "")
            # the MEAN, not the median: the detector runs one frame in redetect_every and the
            # drift bound one cycle in verify_every, and a median of those is 0.0 on a stage
            # that really costs them (scratch/parallax_store_profile.txt)
            + f", hop {np.mean(self._hop_ms):.1f}"
            f" / detect {np.mean(self._detect_ms):.1f}"
            f" / verify {np.mean(self._verify_ms):.1f}"
            f" / rest {np.mean(self._rest_ms):.1f}"
            f" / solve {np.mean(self._solve_ms) if self._solve_ms else 0.0:.1f} ms a frame"
        )


# ---- laws that read the elevation -------------------------------------------------------------
class ElevationLaw(AffineLaw):
    """The affine law with a term in the ray's elevation, 1 / z = a / D + b + c * lift, for an
    error that grows up the picture. Fitted the way the affine law is (the noisy 1 / D
    regressed on the exact 1 / z and lift, then inverted). Each term has its own gate: ``c``
    needs the pool to span elevations (MIN_LIFT_SPREAD between its 5th and 95th lift
    percentiles), ``b`` needs it to span depths (MIN_DEPTH_SPREAD, as the affine law) and is
    0 otherwise; the fit falls back to the affine law when its (a, b) leave the bounds."""

    name = "elevation_law"

    def __init__(self, pool_frames: int = POOL_FRAMES) -> None:
        super().__init__(pool_frames)
        self.c = 0.0

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then the elevation term when the pool can carry it."""
        super().fit(pairs)
        pool = self.pool
        self.c = 0.0
        if pool is None or not self.fitted:
            return
        lo, hi = np.percentile(pool.lift, (5, 95))
        z_lo, z_hi = np.percentile(pool.z, (5, 95))
        if hi - lo < MIN_LIFT_SPREAD:
            return
        with_shift = z_hi / z_lo >= MIN_DEPTH_SPREAD
        x, y = 1.0 / pool.d, 1.0 / pool.z
        w = np.sqrt(pool.weight)
        columns = [y, pool.lift] + ([np.ones_like(y)] if with_shift else [])
        design = np.stack(columns, axis=1) * w[:, None]
        coef, *_ = np.linalg.lstsq(design, x * w, rcond=None)
        res = np.abs(x - (coef[0] * y + coef[1] * pool.lift + (coef[2] if with_shift else 0.0)))
        keep = res <= np.percentile(res, 75)
        if int(keep.sum()) >= 4:
            coef, *_ = np.linalg.lstsq(design[keep], (x * w)[keep], rcond=None)
        alpha, gamma = float(coef[0]), float(coef[1])
        beta = float(coef[2]) if with_shift else 0.0
        if alpha == 0.0:
            return
        a, b, c = 1.0 / alpha, -beta / alpha, -gamma / alpha
        a_lo, a_hi = a_bounds()
        if a_lo <= a <= a_hi and B_BOUNDS[0] <= b <= B_BOUNDS[1]:
            self.a, self.b, self.c = a, b, c

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the law, the elevation of each row from the optics."""
        d = np.asarray(depth, dtype=float)
        lift = lift_of(np.arange(d.shape[0]), ctx.intr)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = self.a / d + self.b + self.c * lift
            z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        out: Array = z
        return out

    def describe(self) -> str:
        return super().describe() + f" c {self.c:+.3f}"


class RowLaw(AffineLaw):
    """A law per band of elevation: the pool's lift range cut into ``bands`` equal bands, an
    affine law fitted in each band that holds POOL_MIN_SAMPLES pairs (the whole pool's law
    elsewhere), the bands' (a, b) interpolated along the rows at apply time — the shape a
    row-dependent error would take, held against the elevation term's straight line."""

    name = "row_law"

    def __init__(self, pool_frames: int = POOL_FRAMES, bands: int = ROW_BANDS) -> None:
        super().__init__(pool_frames)
        self.bands = bands
        self.centres: Array = np.zeros(0)
        self.a_of: Array = np.zeros(0)
        self.b_of: Array = np.zeros(0)

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then one per band of elevation."""
        super().fit(pairs)
        pool = self.pool
        if pool is None or not self.fitted:
            self.centres = np.zeros(0)
            return
        edges = np.linspace(pool.lift.min(), pool.lift.max() + 1e-9, self.bands + 1)
        centres, a_of, b_of = [], [], []
        for lo, hi in pairwise(edges):
            sel = (pool.lift >= lo) & (pool.lift < hi)
            if int(sel.sum()) >= POOL_MIN_SAMPLES:
                a, b = fit_affine(pool.d[sel], pool.z[sel], pool.weight[sel])
            else:
                a, b = self.a, self.b
            centres.append(0.5 * (lo + hi))
            a_of.append(a)
            b_of.append(b)
        self.centres, self.a_of, self.b_of = np.array(centres), np.array(a_of), np.array(b_of)

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the band laws interpolated along the rows."""
        d = np.asarray(depth, dtype=float)
        if self.centres.size == 0:
            return apply_affine(d, self.a, self.b)
        lift = lift_of(np.arange(d.shape[0]), ctx.intr)
        a = np.interp(lift, self.centres, self.a_of)[:, None]
        b = np.interp(lift, self.centres, self.b_of)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = a / d + b
            z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        out: Array = z
        return out

    def describe(self) -> str:
        if self.centres.size == 0:
            return super().describe()
        bands = " ".join(f"{a:.2f}/{b:+.3f}" for a, b in zip(self.a_of, self.b_of, strict=True))
        return f"{self.pooled} pairs, bands a/b {bands}"


# ---- the law of the ray -----------------------------------------------------------------------
class RayLaw(AffineLaw):
    """The affine law with its scale a smooth function of the ray's angle off the optical axis
    (:mod:`pepin.elevation`): ``1 / z = a(eps, az) / D + b(eps, az)``, fitted on the same pool,
    with the plain affine law as its fallback whenever the angular fit does not stand.

    The error this corrects belongs to the camera and the network, not to the room: it is
    parameterised by the ray's own elevation (and azimuth) in radians, so the neck may tilt and
    pan without moving the law — the head's pose enters where it always did, through TF into
    :class:`FrameContext`'s :class:`pepin.depth.CameraPose`, which decides which world point a
    ray meets, not how far the network thinks it is. As a stage the law reads the frame's raw
    depth and keeps the holes of the depth handed to it, so it can stand either as the chain's
    only law or behind the affine law as the flag ``ray_law`` switching between the two live
    (rule 19): with the flag off, the affine law's image goes on unchanged; with it on, the
    same pixels come from the ray's law. Off it also falls back, without a word, whenever
    :func:`pepin.elevation.fit_ray` returns nothing — too few pairs, too narrow a cone, a slope
    that turns non-positive — and the frame is withheld only where the affine law would withhold
    it (no law at all).

    Cost: the stage refits on every frame as the affine law does, and the angular fit is the
    wider design — 8.5 ms a frame (10.5 max) against the affine law's 2.9 on a 47 000-pair pool
    of run 0171's frames, measured on the Mac. The stage is off by default; a pool that large
    only exists with the wall anchor on."""

    name = "ray_law"

    def __init__(
        self,
        pool_frames: int = POOL_FRAMES,
        *,
        degree: int = RAY_DEGREE,
        azimuth_degree: int = RAY_AZIMUTH_DEGREE,
    ) -> None:
        super().__init__(pool_frames)
        self.degree = degree
        self.azimuth_degree = azimuth_degree
        self.gain: RayGain | None = None
        self._seeded_gain: RayGain | None = None
        self._live_gain = False

    def seed_gain(self, gain: RayGain) -> None:
        """Start from a saved gain (the map's), applied until the live pool can fit its own."""
        self._seeded_gain = gain
        self.gain = gain
        self._live_gain = False

    @property
    def ray_ready(self) -> bool:
        """Whether an angular law exists; false means this stage is the affine law."""
        return self.gain is not None

    @property
    def ray_fitted(self) -> bool:
        """Whether the angular law rests on the live pool rather than on a seed."""
        return self._live_gain

    def saved_state(self) -> dict[str, Any] | None:
        """The angular law that stands as plain JSON values for :func:`pepin.depth.save_law`,
        ``None`` while there is none. A seed the live pool has not replaced is written back
        rather than dropped: the file holds one record for both laws, so anything left out of a
        save is erased, and this law belongs to the camera and the network, not to the room the
        last window happened to show (:mod:`pepin.elevation`) — the whole file is only ever
        written while the affine law rests on live pairs."""
        return self.gain.state() if self.gain is not None else None

    @property
    def clipped(self) -> bool:
        """Whether the angular law meets a bound inside its own span (a law at its limit)."""
        return self.gain is not None and self.gain.clipped

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then the angular one on the whole pool; a pool that cannot carry an
        angular law leaves the seeded gain, or none, and the affine law stands."""
        super().fit(pairs)
        pool = self.pool
        if pool is None or not self.fitted:
            return
        elevation, azimuth = ray_angles(pool.lift, pool.left)
        gain = fit_ray(
            pool.d,
            pool.z,
            elevation,
            pool.weight,
            azimuth=azimuth,
            degree=self.degree,
            azimuth_degree=self.azimuth_degree,
        )
        if gain is not None:
            self.gain, self._live_gain = gain, True
        elif self._seeded_gain is None:
            self.gain, self._live_gain = None, False

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the angular law — one (a, b) per pixel, from that pixel's ray —
        or through the plain affine law while no angular law stands."""
        d = np.asarray(depth, dtype=float)
        if self.gain is None:
            return apply_affine(d, self.a, self.b)
        rows = np.arange(d.shape[0])[:, None]
        columns = np.arange(d.shape[1])[None, :]
        elevation, azimuth = ray_angles(lift_of(rows, ctx.intr), left_of(columns, ctx.intr))
        return self.gain.apply(d, elevation, azimuth)

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool and correct the frame's raw depth, keeping the holes of the depth
        handed in (the edge filter's, and the affine law's where it ran before this stage);
        the frame is withheld only while no law of any kind exists."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    def describe(self) -> str:
        """The affine law's words, then the angular law's — marked ``(seed)`` while it is the
        one the last run saved — or why there is none."""
        if self.gain is None:
            return super().describe() + "; affine fallback"
        source = "" if self._live_gain else " (seed)"
        return super().describe() + "; " + self.gain.describe() + source


class RangeLawStage(LawStage):
    """The law whose scale follows the range: the pooled pairs binned by the network's own
    depth and a robust ratio measured in each bin (:class:`pepin.depth.RangeLaw`), applied to
    the same raw depth the affine law reads. On, its image replaces the affine law's.

    The affine law it is built on stays its fallback and its readiness: while the pool has not
    filled :data:`pepin.depth.RANGE_MIN_BINS` bins — the warm-up, or a cart facing one wall at
    one range — the frame goes out through the affine law rather than being withheld, which is
    also what a seed of the saved affine numbers means here. The pool is this stage's own queue
    of the last ``pool_frames`` frames' pairs (the same objects the affine law pools: the
    references cost nothing), so switching the affine law off for an A/B does not blind it."""

    name = "range_law"

    def __init__(self, affine: AffineLaw, pool_frames: int = POOL_FRAMES) -> None:
        self.affine = affine
        self.law: RangeLaw | None = None
        self._live = False  # whether the law rests on the live pool rather than on a seed
        self._pool: list[Pairs] = []
        self._pool_frames = pool_frames

    def seed(self, law: RangeLaw) -> None:
        """Start from a saved range law (the map's): applied until the live pool can fit one."""
        self.law, self._live = law, False

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists — this one's, or the affine law behind it."""
        return self.law is not None or self.affine.ready

    @property
    def fitted(self) -> bool:
        """Whether the range law rests on the live pool (worth saving)."""
        return self._live

    @property
    def pooled(self) -> int:
        """How many live pairs the pool holds."""
        return int(sum(p.size for p in self._pool))

    @property
    def pool(self) -> Pairs | None:
        """Everything in the pool, as one."""
        return Pairs.join(self._pool)

    def saved_state(self) -> dict[str, Any] | None:
        """The range law that stands as plain JSON values for :func:`pepin.depth.save_law`,
        ``None`` while there is none. A seed the live pool has not replaced is written back
        rather than dropped: the file holds one record for every law, so anything left out of
        a save is erased."""
        return self.law.state() if self.law is not None else None

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pairs (or ``None``); the law is refitted on the pool. A pool that
        cannot fill two bins leaves the seeded law, or none, and the affine law stands."""
        if pairs is None or pairs.size == 0:
            return
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        if self.pooled < POOL_MIN_SAMPLES:
            return
        pool = self.pool
        assert pool is not None
        law = RangeLaw.fit(pool.d, pool.z, pool.weight)
        if law is not None:
            self.law, self._live = law, True

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the range law — each pixel scaled by the ratio measured at its own
        range — or through the plain affine law while no range law stands."""
        if self.law is None:
            return apply_affine(depth, self.affine.a, self.affine.b)
        return self.law.apply(depth)

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool and correct the frame's raw depth, keeping the holes of the depth
        handed in (the edge filter's, and the affine law's where it ran before this stage);
        the frame is withheld only while no law of any kind exists."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    def describe(self) -> str:
        """The law for the report line: its bins and their ratios — marked ``(seed)`` while it
        is the one the last run saved — or that the affine law is standing in for it."""
        if self.law is None:
            return f"affine fallback on {self.pooled} pairs"
        source = "" if self._live else " (seed)"
        return f"{self.law.describe()} on {self.pooled} pairs{source}"


def _axis_weights(coord: npt.ArrayLike, size: int, nodes: int) -> Array:
    """Each coordinate's bilinear membership in ``nodes`` node positions spread evenly over
    ``0 .. size - 1``: a (len(coord), nodes) matrix whose rows sum to 1 and hold at most two
    non-zeros — the two nodes a coordinate falls between, by how near it is to each. One node
    means one column of ones: the whole axis belongs to it."""
    out = np.zeros((np.size(coord), nodes))
    if nodes <= 1:
        out[:, 0] = 1.0
        return out
    place = np.clip(np.asarray(coord, dtype=float) / max(size - 1, 1) * (nodes - 1), 0.0, nodes - 1)
    low = np.clip(np.floor(place).astype(int), 0, nodes - 2)
    part = place - low
    who = np.arange(np.size(coord))
    out[who, low] = 1.0 - part
    out[who, low + 1] += part
    return out


class ScaleField:
    """The frame's law as a coarse grid of nodes over the picture instead of one pair of
    numbers for all of it: each node holds its own (a, b) — scale and shift in inverse depth,
    exactly what :func:`pepin.depth.fit_frame` fits — and a pixel's law is the bilinear blend
    of the nodes around it, so there are no seams and no bands.

    Why a field at all: this network's error is regime-wise, not global. Measured against
    COLMAP on run 0171 (scratch/pipeline_vs_truth.txt, 2026-09-11) the raw network reads about
    1.1x on the floor, 1.6x at the lidar's row and 2.0x from 0.3 m up — so any single law
    fitted on a pool holding both the floor and the beams sits between the two and is wrong in
    both places, which is why the floor pairs pulled the lidar's row 5-8 % near and had to be
    switched off. A field lets the floor's pairs move the bottom nodes and leave the row the
    costmap drives on to the beams.

    Every node is fitted by the same weighted, robust regression as the whole frame
    (:func:`pepin.depth.fit_node`) on the pairs that belong to it — each pair's own weight
    times its bilinear membership in that node, so a lidar row falling between two node rows
    feeds both — plus two priors: one pulling the node toward the frame's GLOBAL fit with
    weight ``prior``, and one pulling it toward its own last value with weight ``carry``
    decayed by ``exp(-dt / carry_tau_s)``. The first is what makes the field safe: a node that
    saw nothing is the global fit, so the field degrades to today's single law wherever the
    anchors are sparse. The second is what makes it steady: with a lidar row in the picture it
    barely matters, and on a frame whose only ruler is the floor it is what carries the scale
    of the nodes that saw no pair this time.

    Both are read in PAIRS and mean it: ``prior = N`` carries as much information about a
    node's law as N pairs of weight 1 would AT THAT NODE (:func:`pepin.depth.fit_node`, the
    Tikhonov rows and the ``unit`` that scales them). Until 2026-09-15 they were pseudo-
    observations on the prior's line at the two ends of the frame's depth span, whose pull ran
    from 0.4 to 36 pairs depending on where in that span the node's own pairs sat — so a top
    node holding a handful of far, weak corners could not leave the global fit whatever the
    number said. The frame's own mean inverse depth squared is handed down as that exchange
    rate for the nodes with no pairs of their own.

    A grid of (1, 1) is the old behaviour reachable: one node, no membership to compute and no
    pull to apply — the node IS the frame's global fit, bit for bit what :class:`FrameLaw`
    published before the field existed.

    Measured held out on the four tapes (scratch/scale_field_eval.py, 2026-09-15: every frame's
    lidar pairs split odd / even, the odd fitting and the even judging, over the raw network so
    the numbers are the whole correction). Median |corrected / true - 1| falls from the single
    law's 11.4 / 15.6 / 16.3 / 4.5 % to 8.3 / 15.4 / 14.1 / 4.3 % at 3x3 on run 0171's drive and
    the three neck pitches, and on the drive the residual across the top, middle and bottom third
    of the picture goes 25.8 / 13.6 / 9.2 % -> 11.8 / 9.9 / 6.8 %. 4x3 is a wash against 3x3.
    Where it matters most is the floor's pairs: the arrangement that pulled the lidar's row 5-8 %
    near under one law (11.4 -> 17.6 % on the drive, 4.5 -> 16.3 % at the 40.9 deg pitch) costs
    8.3 -> 10.6 % and 4.3 -> 5.0 % under the field, and IMPROVES the 25.8 deg pitch,
    14.1 -> 13.6 %. The field costs 0.94 ms a frame on a 640x360 image against the single law's
    0.46 (2x2 0.81, 4x3 1.14, 4x4 1.19)."""

    def __init__(
        self,
        grid: tuple[int, int] = (3, 3),
        prior: float = FIELD_PRIOR,
        carry: float = FIELD_CARRY,
        carry_tau_s: float = FIELD_CARRY_TAU_S,
    ) -> None:
        self.prior = prior
        self.carry = carry
        self.carry_tau_s = carry_tau_s
        self._grid = (1, 1)
        self.grid = grid

    @property
    def grid(self) -> tuple[int, int]:
        """How many nodes the field carries, as (rows, columns)."""
        return self._grid

    @grid.setter
    def grid(self, grid: tuple[int, int]) -> None:
        """Re-cut the grid (a live flag): every node starts again from the next frame's fit."""
        rows, cols = int(grid[0]), int(grid[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"a scale field needs at least one node, not {rows}x{cols}")
        self._grid = (rows, cols)
        self._a = np.ones((rows, cols))
        self._b = np.zeros((rows, cols))
        self._seen = np.zeros((rows, cols))
        self._bound = 0
        self._fitted = False
        # Keyed on the GRID as well as the image: the grid is re-cut from the node's parameter
        # thread while a frame is being corrected on the worker's, and axes left over from the
        # grid before would meet the new nodes in a matrix product of the wrong shape.
        self._axes: tuple[tuple[int, int], tuple[int, int], Array, Array] | None = None

    @property
    def fitted(self) -> bool:
        """Whether any frame has fitted the field yet."""
        return self._fitted

    @property
    def nodes(self) -> tuple[Array, Array]:
        """Every node's (scale, shift), as two (rows, columns) images of the grid."""
        return self._a.copy(), self._b.copy()

    @property
    def seen(self) -> Array:
        """How much pair weight each node saw on the last frame fitted (a lidar beam is 1) —
        the only honest measure of which part of the picture spoke for itself."""
        return self._seen.copy()

    def membership(self, rows: Array, cols: Array, shape: tuple[int, int]) -> Array:
        """Each pair's share of each node, as a (grid rows, grid columns, pairs) array: the
        bilinear weights of the (row, column) it sits at, the nodes being the vertices of a
        regular grid spanning the image of ``shape``."""
        down = _axis_weights(rows, shape[0], self._grid[0])
        across = _axis_weights(cols, shape[1], self._grid[1])
        share: Array = np.einsum("ni,nj->ijn", down, across)
        return share

    def fit(
        self,
        d: Array,
        z: Array,
        weight: Array,
        rows: Array,
        cols: Array,
        shape: tuple[int, int],
        law: tuple[float, float],
        dt: float = 0.0,
        shift: bool = True,
    ) -> None:
        """Fit every node on this frame's pairs — ``d`` the depth to correct FROM (whatever the
        prior law published), ``z`` the true depth, ``weight`` each pair's own, ``rows`` and
        ``cols`` where it sits in an image of ``shape`` — around the frame's global fit
        ``law``, with the carry decayed over ``dt`` seconds since the last fit and ``shift``
        saying whether a node may fit a shift at all (the frame's gate, not the node's)."""
        a0, b0 = law
        grid_rows, grid_cols = self._grid
        if self._grid == (1, 1):  # one node: the field IS the frame's law, to the bit
            self._a[:] = a0
            self._b[:] = b0
            self._seen[:] = float(np.sum(weight))
            self._fitted = True
            return
        if not (math.isfinite(a0) and math.isfinite(b0)):
            return  # a NaN global fit would fill every node that saw nothing, and the picture
        share = self.membership(rows, cols, shape) * np.asarray(weight, dtype=float)
        self._seen = share.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / np.asarray(z, dtype=float)
        # What one pair of weight 1 is worth about the SLOPE over the frame as a whole — the
        # exchange rate a node with no pairs of its own has to read its prior in
        # (:func:`pepin.depth.node_unit`).
        finite = np.isfinite(inverse)
        unit = node_unit(inverse[finite], np.asarray(weight, dtype=float)[finite], 1.0)
        carried = 0.0
        if self._fitted and self.carry > 0.0:
            tau = self.carry_tau_s
            decay = math.exp(-max(0.0, dt) / tau) if tau > 0.0 else 0.0
            carried = self.carry * (decay if decay >= FIELD_CARRY_SPENT else 0.0)
        a_new, b_new = np.full_like(self._a, a0), np.full_like(self._b, b0)
        bound = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                mine = share[i, j] > 0.0
                priors = [(a0, b0, self.prior)]
                if carried > 0.0:
                    priors.append((float(self._a[i, j]), float(self._b[i, j]), carried))
                fitted = fit_node(d[mine], z[mine], share[i, j][mine], priors, unit, shift)
                if fitted is not None:
                    a_new[i, j], b_new[i, j] = fitted
                    bound += 1 if at_bound(*fitted) else 0
        self._a, self._b = a_new, b_new
        self._bound = bound
        self._fitted = True

    def law_image(self, shape: tuple[int, int]) -> tuple[float | Array, float | Array]:
        """The (a, b) of every pixel of an image of ``shape``: the nodes blended bilinearly —
        two small matrix products, not a loop over the picture — or the node's two numbers
        themselves on a one-node grid."""
        grid, a_now, b_now = self._grid, self._a, self._b  # one read each: see the grid setter
        if grid == (1, 1):
            return float(a_now[0, 0]), float(b_now[0, 0])
        axes = self._axes
        if axes is None or axes[0] != shape or axes[1] != grid:
            down = _axis_weights(np.arange(shape[0]), shape[0], grid[0])
            across = _axis_weights(np.arange(shape[1]), shape[1], grid[1])
            axes = (shape, grid, down, across)
            self._axes = axes
        _shape, _grid, down, across = axes
        # errstate because this laptop's BLAS raises "divide by zero encountered in matmul" on
        # any float matmul, finite operands and all (scratch/_matmul_warn.py): a spurious flag,
        # and a report line is not the place to print it every frame.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return down @ a_now @ across.T, down @ b_now @ across.T

    def apply(self, depth: Array) -> Array:
        """``depth`` through the field: per pixel 1 / z = a / D + b, the two interpolated."""
        rows, cols = depth.shape
        a, b = self.law_image((int(rows), int(cols)))
        return apply_affine(depth, a, b)

    def describe(self) -> str:
        """The field for the report line: its grid and, node by node, the pair weight each one
        saw on the last frame fitted — ``3x3 [120 0 0 | 85 0 0 | 0 0 0]`` reads as a lidar row
        down the left of the picture and nothing anywhere else, which is the whole question a
        field asks — then how many nodes saw NOTHING (those are the frame's global fit, not a
        fit of their own) and how many came back pinned at the law's bounds.

        The last is not decoration: over run 0171's drive 17 of 72 fitted nodes land on the
        shift's bound at a 3x3 grid (scratch/_field_hazards.py, 2026-09-15), where the bound is
        doing the fitting and the node is not a measurement. The global law has said
        ``AT BOUND`` since it existed; a field of nine laws had no way of saying it."""
        rows = " | ".join(
            " ".join(f"{value:.0f}" for value in row) for row in np.atleast_2d(self._seen)
        )
        empty = int(np.sum(self._seen <= 0.0)) if self._fitted else self._a.size
        tail = f", {empty} empty" if empty else ""
        tail += f", {self._bound} AT BOUND" if self._bound else ""
        return f"{grid_name(self._grid)} [{rows}]{tail}"


class FrameLaw(LawStage):
    """The law of THIS frame: scale and shift fitted on the beams of the frame in hand
    (:func:`pepin.depth.fit_frame`), not on the pool — the alignment the field performs.

    Depth Anything V2's metric heads are evaluated after a per-image scale-and-shift alignment
    against sparse ground truth, and a robot carrying a depth sensor aligns its monocular depth
    against that sensor's points frame by frame. The pool's laws (:class:`AffineLaw`,
    :class:`RangeLawStage`) describe the camera over the last ``POOL_FRAMES`` frames, which is
    the right thing when the network's error is a property of the lens and the wrong thing when
    it is a property of the scene: a new room, a new light, a neck that has tilted. This stage
    asks each frame's own beams what the network is doing right now.

    It corrects what ``prior`` published, not the raw network: the law behind it in the chain
    (the range law, or the affine one) keeps the shape it measured over the pool, and this
    stage removes only what that law left on this frame. Measured on 2026-09-14
    (scratch/frame_law_eval.py, every frame's pairs split odd / even, odd fitting, even
    judging): over the raw network the per-frame law reads a median |residual| of 10.2 % on run
    0171's drive against the range law's 23.2 %, and over the range law 7.5 %; the far band
    (2.0-2.5 m), where the network saturates and no scale can help, is the one place the pool's
    shape still earns its keep. ``prior`` must therefore be a law that maps depth to depth pixel
    by pixel (both pool laws are); an angular law is not one.

    A frame carrying at least ``min_pairs`` pairs gets its own law. A frame carrying fewer
    holds the last one, decaying back to the prior with a time constant of ``tau_s``: the blend
    is of the two published inverse depths, weight ``exp(-held / tau_s)`` on the frame's own, so
    a cart that turns away from every surface the lidar and the camera share returns to the
    pool's law in a few seconds instead of carrying one frame's numbers forever. With no law of
    its own yet the prior's image goes out untouched, so switching this stage on never withholds
    a frame the chain would otherwise have published.

    The law is a FIELD, not a pair of numbers: the frame's fit above is the global one, and a
    grid of nodes over the picture (:class:`ScaleField`, ``grid``, where the numbers are) is
    fitted around it, each node on the pairs that land near it and pulled toward the global fit
    and toward its own last value. ``grid`` (1, 1) is the old behaviour, bit for bit. The reason
    is that this network's error is a property of where in the picture a pixel is — 1.1x on the
    floor, 1.6x at the lidar's row, 2.0x above it — which one law cannot hold and a field can."""

    name = "frame_law"

    def __init__(
        self,
        prior: Law,
        min_pairs: int = FRAME_MIN_PAIRS,
        tau_s: float = FRAME_HOLD_TAU_S,
        clock: Callable[[], float] = time.monotonic,
        shift_needs_beams: bool = True,
        grid: tuple[int, int] = (3, 3),
        field_prior: float = FIELD_PRIOR,
        field_carry: float = FIELD_CARRY,
        field_carry_tau_s: float = FIELD_CARRY_TAU_S,
    ) -> None:
        self.prior = prior
        self.a = 1.0
        self.b = 0.0
        self.min_pairs = min_pairs
        self.tau_s = tau_s
        self.shift_needs_beams = shift_needs_beams
        self.field = ScaleField(grid, field_prior, field_carry, field_carry_tau_s)
        self.frames = 0
        self.fits = 0
        self.held = 0
        self.pairs = 0  # pairs behind the law in hand
        self._fitted = False
        self._clock = clock
        self._last_fit: float | None = None  # when a frame last spoke for itself
        self._rulers: dict[str, float] = {}  # weight per anchor behind the law in hand

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists — this frame's, or the prior behind it."""
        return self._fitted or self.prior.ready

    @property
    def weight(self) -> float:
        """How much of the frame's own law stands right now: 1.0 the moment it was fitted,
        decaying as ``exp(-seconds held / tau_s)`` toward the prior's law, 0.0 with no law of
        its own. A ``tau_s`` at or below zero drops to the prior as soon as a frame is held."""
        if not self._fitted or self._last_fit is None:
            return 0.0
        if self.tau_s <= 0.0:
            return 1.0 if self.held == 0 else 0.0
        return float(math.exp(-max(0.0, self._clock() - self._last_fit) / self.tau_s))

    def fit(self, pairs: Pairs | None, ctx: FrameContext | None = None, beams: bool = True) -> None:
        """Feed this frame's pairs and its context (the prior is asked what it makes of those
        pairs' depths, and the fit is that law's residual): they fit this frame's law outright,
        or the last law is held and starts decaying toward the prior's. Without a context there
        is no prior to correct and the frame is held.

        ``beams`` says whether the lidar is one of the rulers in this pool. On a pool with no
        beams at all — parallax corners alone, the lidar-off case — the shift is shut off and
        the frame gets a scale only while ``shift_needs_beams`` stands: the corners span the
        whole picture's depths, so the spread gate opens, and a two-parameter fit on a ruler of
        7-10 cm per pair runs to the law's bounds. Measured over the four errands of 2026-09-14
        (scratch/parallax_ruler_eval.txt, 101 frames, the corners fitting and the beams judging):
        with the shift the parallax-only law reads 44.5 % median |residual| and its scale jumps
        1.69 between consecutive frames, with the scale alone 30.9 % and 1.15."""
        self.frames += 1
        law, corrected = None, None
        if pairs is not None and pairs.size and ctx is not None:
            corrected = self.prior.apply(pairs.d, ctx)
            spread = math.inf if self.shift_needs_beams and not beams else FRAME_MIN_SPREAD
            law = fit_frame(
                corrected,
                pairs.z,
                pairs.weight,
                self.min_pairs,
                min_spread=spread,
            )
        if law is None or pairs is None or ctx is None or corrected is None:
            self.held += 1
            return
        self.a, self.b = law
        now = self._clock()
        since = 0.0 if self._last_fit is None else max(0.0, now - self._last_fit)
        intr = ctx.intr
        # Where each pair sits in the picture: the anchors carry the ray's angles, and the
        # column and row are those angles back through the optics, whoever measured the pair.
        self.field.fit(
            corrected,
            pairs.z,
            pairs.weight,
            intr.cy - pairs.lift * intr.fy,
            intr.cx - pairs.left * intr.fx,
            (intr.height, intr.width),
            law,
            since,
            shift=law[1] != 0.0,  # the frame's own gate: a node never opens a term it refused
        )
        self.pairs, self._fitted = pairs.size, True
        self.fits += 1
        self._last_fit = now

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The prior's depth through this frame's own law, the prior's alone where this one has
        decayed (the blend is of the published inverse depths) or while no frame has spoken.

        The field is asked for the law only while it HAS one: re-cutting the grid live
        (``field_grid``) starts every node again from the next frame's fit, and between the flag
        and that fit the field is a grid of ones — the identity. This stage's own global two
        numbers stand in over that gap, so a flag flip changes the law's shape and not whether
        there is a law at all; on a cart whose lidar has gone quiet that gap is however many
        seconds the frames are held for, not one frame."""
        prior = self.prior.apply(depth, ctx)
        w = self.weight
        if w <= 0.0:
            return prior
        own = self.field.apply(prior) if self.field.fitted else apply_affine(prior, self.a, self.b)
        if w >= 1.0:
            return own
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = w / own + (1.0 - w) / prior
            out: Array = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        return out

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on this frame's pairs and correct the frame's raw depth, keeping the holes of
        the depth handed in (the edge filter's, and the laws' that ran before); the frame is
        withheld only while no law of any kind exists."""
        pool = frame.pool
        rulers = frame.rulers
        self.fit(pool, frame.ctx, beams=rulers.get("lidar_anchor", 0.0) > 0.0)
        if rulers and self.pairs:
            self._rulers = rulers
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    @property
    def rulers(self) -> str:
        """Which ruler fitted the law in hand, by share of the fit's total weight — e.g.
        ``rulers: lidar 98%, parallax 2%, 312 pts``. Pairs are not the measure: a frame can
        carry 200 parallax corners at 0.03 of a beam's weight each and 30 beams, and the beams
        still write the law. Empty before the first fit."""
        total = sum(self._rulers.values())
        if not self._rulers or total <= 0.0:
            return ""
        shares = ", ".join(
            f"{name.removesuffix('_anchor').removesuffix('_pairs')} {w / total:.0%}"
            for name, w in sorted(self._rulers.items(), key=lambda kv: -kv[1])
        )
        return f"rulers: {shares}, {self.pairs} pts"

    def describe(self) -> str:
        """The law for the report line: this frame's GLOBAL two numbers over the prior's depth,
        the pairs behind them, the field's grid with the pair weight every node saw, which
        rulers' weight fitted them, how many frames have been held against how many seen, and
        how much of the frame's own law still stands against the prior's."""
        if not self._fitted:
            return f"prior stands, {self.held}/{self.frames} frames held"
        clipped = at_bound(self.a, self.b)
        edge = f" [{clipped} AT BOUND]" if clipped else ""
        rulers = self.rulers
        return (
            f"a {self.a:.2f} b {self.b:+.3f} on {self.pairs} pairs{edge}, "
            + f"field {self.field.describe()}, "
            + (f"{rulers}, " if rulers else "")
            + f"{self.held}/{self.frames} frames held (own {self.weight:.2f})"
        )


# ---- the chain ----------------------------------------------------------------------------------
def standard_pipeline(
    law: AffineLaw | None = None,
    *,
    ray: RayLaw | None = None,
    range_stage: RangeLawStage | None = None,
    frame_stage: FrameLaw | None = None,
    floor_pairs: bool | None = None,
    wall_anchor: bool | None = None,
    parallax_anchor: bool | None = None,
    ray_law: bool | None = None,
    range_law: bool | None = None,
    frame_law: bool | None = None,
    wall_correct: bool | None = None,
) -> DepthPipeline:
    """The node's chain: edges -> lidar -> (floor pairs) -> (wall pairs) -> (parallax) -> law
    -> (ray law) -> range law -> frame law -> (wall correction) -> floor anchor; the seven
    switchable stages are in the list and switched by the flags of the same name
    (``wall_anchor`` is the pairs role, ``wall_correct`` the pixels). The ray law, the range law
    and the frame law all sit behind the affine one and correct the same raw depth by their own
    rule instead — by the ray's angle, by the range, by this frame's own beams — and on, each
    replaces the image of the law before it; off, that law's stands. ``range_law`` and
    ``range_law`` and ``frame_law`` are the two of the seven on by
    default: one affine law leaves a residual that tilts 12 % per metre
    (:class:`pepin.depth.RangeLaw`), a law fitted on a minute of pool describes the last
    minute's scene rather than this frame's (:class:`FrameLaw`), and the lidar's one row is not
    the whole picture — the parallax anchor is the second ruler of the scale, weighed against
    the beams by its own noise and measuring where they cannot reach (:class:`ParallaxAnchor`).

    Every switch's default, and every knob of the field and of the floor's sigma, is
    :data:`PIPELINE_DEFAULTS` — the one table the node's FLAGS read theirs from as well, so a
    default is written once. A switch handed in as ``None`` takes the table's value.

    Every law may be handed in so the caller keeps them: the affine and the ray law pool and
    fit on their own, so a saved law must be seeded into **both** (:meth:`AffineLaw.seed`), or
    the ray law withholds every frame of the warm-up while the affine law publishes from the
    seed; the range law falls back to the affine law it is built on and needs no seed of its
    own to publish."""
    defaults = PIPELINE_DEFAULTS
    the_law = law if law is not None else AffineLaw()
    the_ray = ray if ray is not None else RayLaw()
    the_range = range_stage if range_stage is not None else RangeLawStage(the_law)
    the_frame = (
        frame_stage
        if frame_stage is not None
        else FrameLaw(
            the_range,
            grid=grid_of(str(defaults["field_grid"])),
            field_prior=float(defaults["field_prior"]),
            field_carry=float(defaults["field_carry"]),
            field_carry_tau_s=float(defaults["field_carry_tau_s"]),
        )
    )
    geometry = FloorGeometry()
    stages: list[Stage] = [
        EdgeFilter(),
        LidarAnchor(),
        FloorPairs(
            the_law,
            geometry,
            sigma_pitch_deg=float(defaults["floor_sigma_pitch_deg"]),
            normal_tol_deg=float(defaults["floor_normal_tol_deg"]),
            band_max_m=float(defaults["floor_band_max_m"]),
            plane_band=bool(defaults["floor_plane_band"]),
        ),
        WallAnchor(sigma_height=float(defaults["wall_sigma_height"])),
        ParallaxAnchor(),
        the_law,
        the_ray,
        the_range,
        the_frame,
        WallCorrection(),
        FloorAnchor(geometry),
    ]
    given = (
        ("floor_pairs", floor_pairs),
        ("wall_anchor", wall_anchor),
        ("parallax_anchor", parallax_anchor),
        ("ray_law", ray_law),
        ("range_law", range_law),
        ("frame_law", frame_law),
        ("wall_correct", wall_correct),
    )
    flags = [(name, default_switch(name, value)) for name, value in given]
    return DepthPipeline(stages, off=[name for name, on in flags if not on])


__all__ = [
    "FIELD_GRIDS",
    "PIPELINE_DEFAULTS",
    "AffineLaw",
    "Anchor",
    "AnchorStage",
    "DepthPipeline",
    "EdgeFilter",
    "ElevationLaw",
    "FloorAnchor",
    "FloorGeometry",
    "FloorPairs",
    "Frame",
    "FrameContext",
    "FrameLaw",
    "Law",
    "LawStage",
    "LidarAnchor",
    "MapMotionSource",
    "MotionSource",
    "Pairs",
    "ParallaxAnchor",
    "PreviousFrame",
    "RangeLawStage",
    "RayLaw",
    "RecentMapMotionSource",
    "Result",
    "Rigid",
    "RowLaw",
    "ScaleField",
    "Stage",
    "StageStats",
    "Verdict",
    "WallAnchor",
    "WallCorrection",
    "WallWalk",
    "default_switch",
    "fit_plane",
    "floor_sigma",
    "grid_name",
    "grid_of",
    "left_of",
    "level_plane",
    "lift_of",
    "standard_pipeline",
    "wall_sigma",
]
