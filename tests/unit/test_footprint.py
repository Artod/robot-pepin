"""The cart's hull: one source, and the scan filter's box grown from it."""


def test_the_hull_has_one_source_and_the_scan_filter_box_grows_from_it() -> None:
    """config/base.json is what the base server reads; pepin.footprint.HULL is what the ROS side
    derives from. They must be the same cart. The scan filter's box is the hull plus the contact
    band on every side."""
    import json
    from pathlib import Path

    from pepin.footprint import CONTACT_BAND_M, HULL, Footprint, hull_box

    cfg = json.loads((Path(__file__).resolve().parents[2] / "config/base.json").read_text())
    assert Footprint.from_config(cfg["footprint"]) == HULL
    assert HULL.inscribed_radius_m == HULL.front_m
    assert abs(HULL.circumscribed_radius_m - 0.4070) < 1e-3
    assert HULL.polygon() == [(0.0625, 0.275), (0.0625, -0.275), (-0.30, -0.275), (-0.30, 0.275)]
    box = hull_box(band_m=0.05)
    assert box == {
        "min_x": -(HULL.rear_m + 0.05),
        "max_x": HULL.front_m + 0.05,
        "min_y": -(HULL.half_width_m + 0.05),
        "max_y": HULL.half_width_m + 0.05,
    }
    assert hull_box()["max_x"] == HULL.front_m + CONTACT_BAND_M


def test_the_hull_s_cells_on_a_grid_follow_its_pose() -> None:
    """The cart at (0.01, 0.01) facing +x covers x -0.29..0.0725 and y -0.265..0.285: on a 0.1 m
    grid from (-1, -1), columns 7..10 and rows 7..12; turned to face +y the same outline lies
    across, columns 7..12 and rows 7..10; moved half a metre, it moves five columns."""
    import math

    from pepin.footprint import hull_cells

    def spans(cells: set[tuple[int, int]]) -> tuple[set[int], set[int]]:
        return {c for c, _ in cells}, {r for _, r in cells}

    cells = hull_cells(0.01, 0.01, 0.0, (-1.0, -1.0), 0.1)
    assert spans(cells) == (set(range(7, 11)), set(range(7, 13)))
    assert len(cells) == 4 * 6, "the whole rectangle, not its outline"
    across = hull_cells(0.01, 0.01, math.pi / 2, (-1.0, -1.0), 0.1)
    assert spans(across) == (set(range(7, 13)), set(range(7, 11)))
    moved = hull_cells(0.51, 0.01, 0.0, (-1.0, -1.0), 0.1)
    assert moved == {(c + 5, r) for c, r in cells}
