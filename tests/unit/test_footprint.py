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
