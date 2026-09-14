"""The anchor kept beside its map: what the file holds, what it refuses, when it is re-learned."""

import json
from pathlib import Path

import pytest

from pepin.anchors import (
    RELEARN_HOLD_S,
    Anchor,
    AnchorWatch,
    anchor_path,
    load_anchor,
    map_slug,
    save_anchor,
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

    save_anchor(tmp_path, Anchor(Pose2D(1.0, 0.0), "341x341@-8.50,-8.50"))
    foreign = anchor_path(tmp_path, "341x341@-8.50,-8.50")
    foreign.rename(anchor_path(tmp_path, MAP))
    with pytest.raises(ValueError, match="anchor of map"):
        load_anchor(tmp_path, MAP)


def test_a_closure_landing_is_not_a_stale_anchor() -> None:
    """A loop closure moves the graph's word by centimetres and the word comes back; the watch
    must not rewrite the file for that. Only a disagreement that HOLDS is evidence."""
    watch = AnchorWatch()
    assert not watch.update(0.0, True, 0.05, 1.0), "a closure-sized gap is not a gap at all"
    assert watch.since is None

    assert not watch.update(1.0, True, 0.9, 0.0), "the clock starts here"
    assert watch.since == 1.0
    assert not watch.update(1.0 + RELEARN_HOLD_S - 0.1, True, 0.9, 0.0)
    assert not watch.update(3.0, True, 0.05, 0.0), "the word came back: nothing to re-learn"
    assert watch.since is None


def test_a_gap_that_holds_while_the_lidar_drives_fires_exactly_once() -> None:
    """Five seconds of disagreement with a trusted tracker is the evidence; one stale anchor
    must produce one re-learn, not one per graph message afterwards."""
    watch = AnchorWatch()
    watch.update(0.0, True, 0.9, 0.0)
    assert watch.update(RELEARN_HOLD_S + 0.1, True, 0.9, 0.0)
    assert watch.since is None, "the clock restarts: the next re-learn needs its own five seconds"
    assert not watch.update(RELEARN_HOLD_S + 0.2, True, 0.9, 0.0)


def test_a_turn_alone_is_enough_and_an_untrusted_tracker_is_never_evidence() -> None:
    """Twenty degrees is past anything a closure accounts for, so a heading gap counts on its
    own — but with the lidar silent there is nothing to re-learn FROM, and the clock never even
    starts: a graph corrected against a dead-reckoned pose would write that drift into the file."""
    watch = AnchorWatch()
    assert watch.disagrees(0.0, 25.0) and watch.disagrees(0.0, -25.0)
    assert not watch.disagrees(0.4, 19.0)

    watch.update(0.0, False, 9.0, 90.0)
    assert watch.since is None
    assert not watch.update(100.0, False, 9.0, 90.0)
