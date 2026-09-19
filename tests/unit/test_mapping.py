import math

import numpy as np
import pytest

from pepin.mapping import GridSpec, MapChoice, OccupancyGrid, map_shift, transform_to_world
from pepin.odometry import Pose2D


def test_transform_rotates_then_translates() -> None:
    pts = np.array([[1.0, 0.0]])
    out = transform_to_world(pts, Pose2D(x=2.0, y=3.0, theta=math.pi / 2))
    assert out[0] == pytest.approx([2.0, 4.0])


def test_hit_cell_becomes_occupied_and_beam_cells_free() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.1, x_min_m=-1, y_min_m=-1, width_m=4, height_m=2))
    for _ in range(3):
        grid.integrate(Pose2D(), np.array([[1.0, 0.0]]))
    p = grid.probability()
    hit = grid.world_to_cell(np.array([[1.0, 0.0]]))[0]
    mid = grid.world_to_cell(np.array([[0.5, 0.0]]))[0]
    untouched = grid.world_to_cell(np.array([[0.0, 0.5]]))[0]
    assert p[hit[0], hit[1]] > 0.9
    assert p[mid[0], mid[1]] < 0.3
    assert p[untouched[0], untouched[1]] == pytest.approx(0.5)


def test_points_outside_the_grid_are_ignored() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.5, x_min_m=0, y_min_m=0, width_m=1, height_m=1))
    grid.integrate(Pose2D(), np.array([[5.0, 5.0], [-3.0, 0.0]]))
    assert grid.log_odds.shape == (2, 2)


def test_log_odds_are_clamped() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.1, x_min_m=-1, y_min_m=-1, width_m=2, height_m=2))
    for _ in range(50):
        grid.integrate(Pose2D(), np.array([[0.5, 0.0]]))
    assert grid.log_odds.max() <= 5.0 and grid.log_odds.min() >= -5.0


# ---- MapChoice: when a tracker takes the next picture of the one map -------------------------


class Adopted:
    """A fake adopter: counts how often the choice said "take this one"."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def offer(choice: MapChoice, source: str, digest: str, now: float, taken: Adopted) -> bool:
    return choice.offer(source, lambda: digest, now, taken)


def test_the_first_grid_on_the_one_topic_is_adopted() -> None:
    """One map, one topic (World R): there is nothing to choose between, only when to take the
    next picture of it."""
    choice, taken = MapChoice(), Adopted()
    assert offer(choice, "/map", "b", 0.0, taken)
    assert (taken.count, choice.source, choice.digest) == (1, "/map", "b")
    assert choice.adoptions == 1


def test_a_republished_map_is_ignored_while_the_refresh_is_zero() -> None:
    """A served file's behaviour: the first map and no other. RTAB-Map's grid arrives once a
    second, and every adoption rebuilds the matcher on four A53 cores."""
    choice, taken = MapChoice(), Adopted()
    offer(choice, "/map", "a", 0.0, taken)
    for t in (1.0, 2.0, 600.0):
        assert not offer(choice, "/map", "changed", t, taken)
    assert taken.count == 1 and choice.take_ignored() == 3 and choice.take_ignored() == 0


def test_with_a_refresh_a_changed_map_is_taken_and_an_unchanged_one_is_not() -> None:
    choice, taken = MapChoice(refresh_s=30.0), Adopted()
    offer(choice, "/map", "a", 0.0, taken)
    assert not offer(choice, "/map", "b", 29.0, taken), "too soon"
    assert not offer(choice, "/map", "a", 31.0, taken), "old enough, but the same cells"
    assert offer(choice, "/map", "b", 31.0, taken)
    assert taken.count == 2 and choice.digest == "b" and choice.adoptions == 2


def test_the_refresh_is_a_live_switch_of_its_own() -> None:
    choice, taken = MapChoice(), Adopted()
    offer(choice, "/map", "a", 0.0, taken)
    choice.switch("map_refresh_s", 10.0)
    assert offer(choice, "/map", "b", 11.0, taken) and taken.count == 2


def test_the_digest_is_read_only_when_the_answer_hangs_on_the_cells() -> None:
    """Reading a whole grid costs something on the board; the gate usually answers without it."""
    reads = Adopted()

    def digest() -> str:
        reads()
        return "a"

    choice = MapChoice()
    choice.offer("/map", digest, 0.0, Adopted())  # the first: the cells are remembered
    choice.offer("/map", digest, 2.0, Adopted())  # a republication under refresh 0
    assert reads.count == 1


def test_a_republication_is_refused_before_anything_reads_its_cells() -> None:
    """The board's own budget: RTAB-Map offers a grid a second and nearly all of them are refused,
    so neither the digest nor the emptiness question may be asked of one that is too soon."""
    reads = Adopted()
    choice = MapChoice()
    choice.offer("/map", lambda: "a", 0.0, Adopted(), empty=lambda: bool(reads()))
    assert reads.count == 1, "the first grid is asked once"
    choice.offer("/map", lambda: "b", 0.5, Adopted(), empty=lambda: bool(reads()))
    assert reads.count == 1 and choice.take_ignored() == 1


def test_a_map_with_nothing_in_it_is_not_adopted() -> None:
    """A newborn database can publish a grid with no known cell in it. Adopting that one spends the
    single adoption a refresh_s of 0 allows, and the tracker then refuses every real map for the
    rest of the session — so it is turned away, and counted, until a node fills it."""
    choice = MapChoice()
    taken: list[str] = []
    blank = True

    def offer() -> bool:
        return choice.offer(
            "/map", lambda: "d1", 0.0, lambda: taken.append("/map"), empty=lambda: blank
        )

    assert not offer() and not taken and choice.source == "" and choice.adoptions == 0
    assert choice.take_ignored() == 1, "the wait is visible"
    blank = False
    assert offer() and taken == ["/map"] and choice.source == "/map"


def test_a_caller_that_asks_nothing_about_the_cells_behaves_as_before() -> None:
    """The question is optional: every existing caller passes four arguments and is unchanged."""
    choice = MapChoice()
    taken: list[str] = []
    assert choice.offer("/map", lambda: "d1", 0.0, lambda: taken.append("/map"))
    assert taken == ["/map"]


# ---- map_shift: what a re-rendered grid did to the one before it ------------------------------


def _grid(x_min: float = 0.0, cells: int = 4) -> OccupancyGrid:
    return OccupancyGrid(
        GridSpec(
            resolution_m=0.05,
            x_min_m=x_min,
            y_min_m=0.0,
            width_m=cells * 0.05,
            height_m=cells * 0.05,
        )
    )


def test_the_first_map_is_compared_with_nothing() -> None:
    assert map_shift(None, _grid()) is None


def test_a_grid_of_the_same_geometry_reports_the_cells_that_changed() -> None:
    """The ordinary re-render: RTAB-Map assembled the same canvas again and a few cells moved."""
    before, after = _grid(), _grid()
    after.log_odds[1, 1] = 4.0
    after.log_odds[2, 2] = -4.0
    shift = map_shift(before, after)
    assert shift is not None and not shift.resized and shift.changed_cells == 2
    assert shift.origin_m == 0.0 and "2 cells changed" in shift.phrase()


def test_a_bent_map_reports_its_moved_origin_and_counts_no_cells() -> None:
    """A loop closure clears RTAB-Map's global map and re-assembles it, so the canvas itself moves
    (rtabmap/core/GlobalMap.cpp). Across two lattices a cell is not the same cell, so nothing is
    counted and the phrase says the map was resized."""
    shift = map_shift(_grid(), _grid(x_min=-0.30))
    assert shift is not None and shift.resized and shift.changed_cells == 0
    assert shift.origin_m == pytest.approx(0.30) and "resized" in shift.phrase()


def test_a_grown_map_is_a_resize_even_at_the_same_origin() -> None:
    shift = map_shift(_grid(), _grid(cells=8))
    assert shift is not None and shift.resized and shift.origin_m == 0.0
