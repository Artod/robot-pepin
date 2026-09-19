import math

import numpy as np
import pytest

from pepin.mapping import GridSpec, MapChoice, OccupancyGrid, transform_to_world
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


# ---- MapChoice: which map a tracker matches on -----------------------------------------------


class Adopted:
    """A fake adopter: counts how often the choice said "take this one"."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def offer(choice: MapChoice, source: str, digest: str, now: float, taken: Adopted) -> bool:
    return choice.offer(source, lambda: digest, now, taken)


def test_the_first_map_on_the_wanted_topic_is_adopted_and_another_topic_is_not() -> None:
    choice, taken = MapChoice("map"), Adopted()
    assert not offer(choice, "map_lidar", "a", 0.0, taken), "nobody asked for that topic"
    assert offer(choice, "map", "b", 0.0, taken)
    assert (taken.count, choice.source, choice.digest) == (1, "map", "b")


def test_a_republished_map_is_ignored_while_the_refresh_is_zero() -> None:
    """The served file's behaviour, and the default: the first map and no other. /map_lidar is
    republished every second, and every adoption rebuilds the matcher on four A53 cores."""
    choice, taken = MapChoice("map_lidar"), Adopted()
    offer(choice, "map_lidar", "a", 0.0, taken)
    for t in (1.0, 2.0, 600.0):
        assert not offer(choice, "map_lidar", "changed", t, taken)
    assert taken.count == 1 and choice.take_ignored() == 3 and choice.take_ignored() == 0


def test_with_a_refresh_a_changed_map_is_taken_and_an_unchanged_one_is_not() -> None:
    choice, taken = MapChoice("map_lidar", refresh_s=30.0), Adopted()
    offer(choice, "map_lidar", "a", 0.0, taken)
    assert not offer(choice, "map_lidar", "b", 29.0, taken), "too soon"
    assert not offer(choice, "map_lidar", "a", 31.0, taken), "old enough, but the same cells"
    assert offer(choice, "map_lidar", "b", 31.0, taken)
    assert taken.count == 2 and choice.digest == "b"


def test_moving_the_flag_adopts_the_other_map_whatever_the_refresh_says() -> None:
    """A map on the OTHER topic is the operator asking for it: no gate applies, and the caller
    is told to offer what every topic last published (a served map speaks once, latched)."""
    choice, taken = MapChoice("map"), Adopted()
    offer(choice, "map", "file", 0.0, taken)
    moved = Adopted()
    choice.on_choice(moved)
    choice.switch("map_topic", "map_lidar")
    assert moved.count == 1 and choice.wanted == "map_lidar"
    assert offer(choice, "map_lidar", "volume", 0.1, taken)
    assert taken.count == 2 and choice.source == "map_lidar"
    choice.switch("map_topic", "map_lidar")
    assert moved.count == 1, "the same topic again is not a move"


def test_the_refresh_is_a_live_switch_of_its_own() -> None:
    choice, taken = MapChoice("map_lidar"), Adopted()
    offer(choice, "map_lidar", "a", 0.0, taken)
    choice.switch("map_refresh_s", 10.0)
    assert offer(choice, "map_lidar", "b", 11.0, taken) and taken.count == 2


def test_the_digest_is_read_only_when_the_answer_hangs_on_the_cells() -> None:
    """Reading a whole grid costs something on the board; the gate usually answers without it."""
    reads = Adopted()

    def digest() -> str:
        reads()
        return "a"

    choice = MapChoice("map_lidar")
    choice.offer("map_lidar", digest, 0.0, Adopted())  # the first: the cells are remembered
    choice.offer("map", digest, 1.0, Adopted())  # another topic
    choice.offer("map_lidar", digest, 2.0, Adopted())  # a republication under refresh 0
    assert reads.count == 1


def test_the_wanted_map_never_speaking_falls_back_to_the_one_that_is_there() -> None:
    """The board serves /map itself and reads /map_lidar from the laptop: a tracker asking for
    the laptop's map with the wifi down must not wait for ever (CLAUDE.md rule 20)."""
    choice, taken = MapChoice("map_lidar", fallback="map", fallback_after_s=10.0), Adopted()
    assert choice.lapsed(0.0) is None, "the clock starts at the first ask"
    assert choice.lapsed(9.0) is None
    assert choice.lapsed(10.0) == "map"
    assert offer(choice, "map", "file", 10.0, taken)
    assert (taken.count, choice.source, choice.fell_back) == (1, "map", True)

    assert choice.lapsed(60.0) is None, "a map is in use: the fallback is over"
    assert offer(choice, "map_lidar", "volume", 61.0, taken), "the wanted map still replaces it"
    assert (taken.count, choice.source, choice.fell_back) == (2, "map_lidar", False)


def test_a_map_in_use_is_never_dropped_because_its_publisher_went_quiet() -> None:
    """A grid in memory does not stop working when the laptop does; a rebuild on a lesser map
    would cost the board 15 s of lattice and every candidate it holds, for nothing."""
    choice, taken = MapChoice("map_lidar", fallback="map", fallback_after_s=10.0), Adopted()
    offer(choice, "map_lidar", "volume", 0.0, taken)
    assert choice.lapsed(1000.0) is None and not choice.fell_back


def test_the_fallback_is_a_live_switch_and_zero_waits_for_ever() -> None:
    choice = MapChoice("map_lidar", fallback="map", fallback_after_s=10.0)
    choice.switch("map_fallback_s", 0.0)
    choice.lapsed(0.0)
    assert choice.lapsed(1000.0) is None, "0: the behaviour before the fallback existed"
    choice.switch("map_fallback_s", 5.0)
    choice.lapsed(1000.0)
    assert choice.lapsed(1006.0) == "map"


def test_moving_the_flag_restarts_the_wait_for_the_new_map() -> None:
    """The tracker asked for another map at this moment; the seconds it spent waiting for the
    previous one are not seconds it waited for this one."""
    choice = MapChoice("map", fallback="map", fallback_after_s=10.0)
    choice.lapsed(0.0)
    choice.switch("map_topic", "map_lidar")
    assert choice.lapsed(11.0) is None, "the wait for map_lidar starts here"
    assert choice.lapsed(21.0) == "map"


def test_a_map_with_nothing_in_it_is_not_adopted() -> None:
    """An unknown room's first publication is an all-unknown grid. Adopting it spends the single
    adoption a refresh_s of 0 allows, and the tracker then refuses every real map for the rest of
    the session — so it is turned away, and counted, until the first sweep fills it."""
    choice = MapChoice()
    taken: list[str] = []
    blank = True

    def offer() -> bool:
        return choice.offer(
            "map", lambda: "d1", 0.0, lambda: taken.append("map"), empty=lambda: blank
        )

    assert not offer() and not taken and choice.source == ""
    assert choice.take_ignored() == 1, "the wait is visible"
    blank = False
    assert offer() and taken == ["map"] and choice.source == "map"


def test_a_caller_that_asks_nothing_about_the_cells_behaves_as_before() -> None:
    """The question is optional: every existing caller passes four arguments and is unchanged."""
    choice = MapChoice()
    taken: list[str] = []
    assert choice.offer("map", lambda: "d1", 0.0, lambda: taken.append("map"))
    assert taken == ["map"]
