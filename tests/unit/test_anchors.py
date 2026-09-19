"""The one-seating anchor file kept beside its map: what it holds, and what a seating must be
worth for the tie between the two frames to be measured from it at all."""

import json
import math
from pathlib import Path

import pytest

from pepin.anchors import (
    Anchor,
    anchor_path,
    describe_sigma,
    load_anchor,
    map_slug,
    save_anchor,
    seating_refusal,
)
from pepin.odometry import Pose2D

MAP = "239x215@-18.53,-4.38"


def test_a_map_s_identity_becomes_a_file_name_and_stays_recognisable() -> None:
    """The file is named by the map it belongs to, with only the characters a shell would
    stumble over replaced: an owner listing ros/maps must see which map an anchor is for."""
    assert map_slug(MAP) == "239x215_-18.53_-4.38"
    assert anchor_path("/maps", MAP).name == "239x215_-18.53_-4.38.graph_anchor.json"


def test_the_anchor_survives_the_round_trip_and_comes_back_marked_as_read(tmp_path: Path) -> None:
    """A session learns the anchor and writes it; the next one reads the same transform back and
    knows it did not learn it itself — which is what lets it speak before the tracker does."""
    written = Anchor(Pose2D(0.8, 2.0, 0.5), MAP, learned_at=11.0, origin="learned")
    path = save_anchor(tmp_path, written)
    assert path == anchor_path(tmp_path, MAP)
    assert json.loads(path.read_text())["map"] == MAP, "the identity, not only the file name"

    read = load_anchor(tmp_path, MAP)
    assert read is not None
    assert (read.pose.x, read.pose.y, read.pose.theta) == (0.8, 2.0, 0.5)
    assert read.origin == "file" and read.learned_at == 11.0
    assert "from file" in read.described()


def test_no_file_is_silence_and_a_broken_one_is_said_out_loud(tmp_path: Path) -> None:
    """A missing anchor is the normal first session. A file that is there but cannot be believed
    is not: a wrong anchor is a cart confidently in the wrong room, so it is refused loudly."""
    assert load_anchor(tmp_path, MAP) is None

    anchor_path(tmp_path, MAP).write_text("{ not json")
    with pytest.raises(ValueError, match="not an anchor"):
        load_anchor(tmp_path, MAP)

    # ...and a file of ANOTHER map under this one's name, which is only checkable while the file
    # name and the identity inside are one string — the legacy size@origin naming. Under a room name
    # they
    # are two different things and the room is the identity (see load_anchor).
    save_anchor(tmp_path, Anchor(Pose2D(1.0, 0.0), "341x341@-8.50,-8.50"))
    foreign = anchor_path(tmp_path, "341x341@-8.50,-8.50")
    foreign.rename(anchor_path(tmp_path, MAP))
    with pytest.raises(ValueError, match="anchor of map"):
        load_anchor(tmp_path, MAP, MAP)
    read = load_anchor(tmp_path, MAP)
    assert read is not None and read.map_id == "341x341@-8.50,-8.50", (
        "asked nothing, told the truth"
    )


def test_a_seating_the_scan_pins_in_one_axis_only_is_no_place_to_learn_a_frame() -> None:
    """The anchor is a constant of the pair: whatever seating it is learned off is baked into
    every word the graph says until the file is rewritten. A scan sliding along a sofa reports
    an honest fit and half a metre of freedom in y, so the gate reads the error bar the tracker
    publishes, not its score — and refuses with a reason a person can read in a log."""
    sharp = (0.008, 0.012, math.radians(0.4))
    assert seating_refusal(sharp) is None

    along_a_sofa = seating_refusal((0.008, 0.40, math.radians(0.4)))
    assert along_a_sofa is not None and "soft" in along_a_sofa and "40.0 cm" in along_a_sofa

    free_to_turn = seating_refusal((0.008, 0.012, math.radians(4.0)))
    assert free_to_turn is not None and "heading" in free_to_turn

    assert seating_refusal(None) is not None, "no covariance is not a sharp seating"
    assert seating_refusal((0.008, 0.40, math.radians(4.0)), 1.0, 180.0) is None, "wide open"


def test_a_seating_reads_as_centimetres_and_degrees_in_a_report_line() -> None:
    """An operator reads the report line, not the covariance: metres and radians are neither."""
    assert describe_sigma((0.008, 0.012, math.radians(0.23))) == "0.8/1.2 cm, 0.23 deg"
    assert describe_sigma(None) == "unknown"
