"""rtabmap_frame's two new switches under the ROS stubs: where RTAB-Map's adapters compute (a file
it writes for them, their counters read back into its report line), and how RTAB-Map finds which
node a picture is — the descriptor sent only when it cannot abort RTAB-Map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup import rtabmap_frame  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from pepin.flags import load_knobs, with_knobs  # noqa: E402
from pepin.global_descriptor import (  # noqa: E402
    NULL_TAG,
    PLACE_TOPIC,
    TFIDF,
    Census,
    SnapshotPlace,
)
from pepin.live_settings import StatusBoard, registration_file  # noqa: E402

DIM = 12288
TAG = "boq_dinov2@d72ee0ce"
UNIFORM = Census(169, 0, 0, {(TAG, DIM): 147, (NULL_TAG, DIM): 22})


def _node(census: Census | None = None, monkeypatch: pytest.MonkeyPatch | None = None) -> Any:
    if monkeypatch is not None:
        if census is None:
            monkeypatch.delenv(rtabmap_frame.CENSUS_ENV, raising=False)
        else:
            monkeypatch.setenv(rtabmap_frame.CENSUS_ENV, census.to_json())
    node = rtabmap_frame.RtabmapFrame()
    node._rtabmap_keeps = True  # the patched image's marker; its absence is its own test
    for client in node.service_clients.values():
        client.ready = True
    return node


def _tick(node: Any) -> None:
    node.timers[0][1]()


def _likelihoods(node: Any) -> list[str]:
    """The Kp/TfIdfLikelihoodUsed values this node has sent RTAB-Map, in order."""
    tuner = node.service_clients[f"{rtabmap_frame.RTABMAP_NODE}/set_parameters_atomically"]
    return [
        p.value.string_value for call in tuner.calls for p in call.parameters if p.name == TFIDF
    ]


def _say(node: Any, place: SnapshotPlace) -> None:
    node.subs[PLACE_TOPIC][1](String(data=place.to_json()))


# ---- where the adapters compute ---------------------------------------------------------------
def test_the_registration_flags_are_written_for_rtabmap_s_adapters_and_move_live() -> None:
    node = _node()
    _tick(node)
    written = json.loads(Path(registration_file()).read_text())
    assert written == {"backend": "auto", "timeout_s": 1.0, "top_k": 2048}
    node._switches.set("registration_backend", "local")
    node._switches.set("xfeat_top_k", 1024)
    _tick(node)
    assert json.loads(Path(registration_file()).read_text())["backend"] == "local"
    assert json.loads(Path(registration_file()).read_text())["top_k"] == 1024
    with pytest.raises(ValueError):
        node._switches.set("registration_backend", "gpu")


def test_the_adapters_counters_are_in_the_report_line() -> None:
    node = _node()
    StatusBoard().publish(
        "xfeat",
        {
            "backend": "auto",
            "service": 118,
            "local": 0,
            "fallback": 2,
            "failed": 0,
            "round_trip_ms": 41.0,
        },
    )
    node._report()
    line = node.logger.texts("info")[-1]
    assert "adapters (auto) xfeat auto: service 118, local 0, fallback 2, failed 0, 41.0 ms" in line
    assert "registration_backend=auto" in line and "place_recognition=words" in line
    assert "descriptor_null_share=0.5" in line


# ---- how a place is found ---------------------------------------------------------------------
def test_the_first_decision_is_always_sent_then_only_a_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This node may be a respawn beside an RTAB-Map still holding the descriptor likelihood an
    earlier incarnation sent: the words are SENT once at start, not assumed, then only changes."""
    node = _node(UNIFORM, monkeypatch)
    _say(node, SnapshotPlace(True, "service", DIM, TAG))
    _tick(node)
    assert _likelihoods(node) == ["true"] and node._recognition_why == "words"
    _tick(node)
    assert _likelihoods(node) == ["true"], "sent once, not every tick"


def test_descriptor_goes_out_only_when_every_node_carries_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(UNIFORM, monkeypatch)
    node._switches.set("place_recognition", "descriptor")
    node._switches.set("visual_confirm", "rtabmap")
    _tick(node)
    assert _likelihoods(node) == ["true"], "sensor_pack has not said what its snapshots carry"
    assert "sensor_pack has not said" in node._recognition_why
    _say(node, SnapshotPlace(True, "service", DIM, TAG))
    _tick(node)
    assert _likelihoods(node) == ["true", "false"]
    assert node._recognition_why.startswith("descriptor (169 nodes: 147 boq_dinov2@d72ee0ce/12288")
    _tick(node)
    assert _likelihoods(node) == ["true", "false"], "sent once, not every tick"
    _say(node, SnapshotPlace(False, "service", 0, ""))  # the snapshots stopped carrying one
    _tick(node)
    assert _likelihoods(node) == ["true", "false", "true"], "the words back at once"
    node._report()
    assert (
        "place recognition words (descriptor asked; the snapshots carry no descriptor)"
        in (node.logger.texts("info")[-1])
    )


def test_a_service_that_stops_describing_hands_the_places_back_to_the_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null query scores every node alike and no hypothesis forms: descriptor mode would go
    blind while the service is down, where the words still recognise the place."""
    node = _node(UNIFORM, monkeypatch)
    node._switches.set("place_recognition", "descriptor")
    node._switches.set("visual_confirm", "rtabmap")
    _say(node, SnapshotPlace(True, "service", DIM, TAG, recent=10, recent_null=1))
    _tick(node)
    assert _likelihoods(node)[-1] == "false"
    _say(node, SnapshotPlace(True, "service", DIM, TAG, recent=10, recent_null=7))
    _tick(node)
    assert _likelihoods(node)[-1] == "true"
    assert "7 of the last 10 camera snapshots carried the null descriptor" in (
        node._recognition_why
    )
    node._switches.set("descriptor_null_share", 1.0)  # live: the descriptor whatever happens
    _tick(node)
    assert _likelihoods(node)[-1] == "false"
    _say(node, SnapshotPlace(True, "null", DIM, TAG))  # place_descriptor off on sensor_pack
    _tick(node)
    assert _likelihoods(node)[-1] == "true" and "place_descriptor is off" in node._recognition_why


def test_the_descriptor_under_aggressive_confirm_says_what_it_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node(UNIFORM, monkeypatch)
    node._switches.set("place_recognition", "descriptor")
    _say(node, SnapshotPlace(True, "service", DIM, TAG))
    _tick(node)
    assert _likelihoods(node) == ["false"], "sent: the confirm is a separate choice"
    assert "visual_confirm aggressive" in node._recognition_why
    assert "2 of 11 wrong localisations" in node._recognition_why


def test_an_unpatched_rtabmap_keeps_the_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """RTAB-Map 0.22.1 drops a node's descriptor when it reloads the node's data for a
    registration and aborts at the next comparison: without the patch's marker, never descriptor."""
    node = _node(UNIFORM, monkeypatch)
    node._rtabmap_keeps = False
    node._switches.set("place_recognition", "descriptor")
    _say(node, SnapshotPlace(True, "service", DIM, TAG))
    _tick(node)
    assert _likelihoods(node) == ["true"]
    assert "rtabmap-keep-global-descriptors.patch" in node._recognition_why
    assert rtabmap_frame.RtabmapFrame()._rtabmap_keeps is False, "no marker on this machine"


@pytest.mark.parametrize(
    ("census", "why"),
    [
        (None, "no census of the database at this start"),
        (Census(169, 22, 0, {(TAG, DIM): 147}), "22 of 169 nodes carry no descriptor"),
        (Census(169, 0, 0, {("boq_r50@0", 16384): 169}), "the snapshots' 12288"),
        (Census(169, 0, 0, {("boq_dinov2@other", DIM): 169}), "described by ['boq_dinov2@other']"),
    ],
)
def test_a_database_that_could_abort_rtabmap_keeps_the_words(
    census: Census | None, why: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(census, monkeypatch)
    node._switches.set("place_recognition", "descriptor")
    _say(node, SnapshotPlace(True, "service", DIM, TAG))
    _tick(node)
    assert _likelihoods(node) == ["true"] and why in node._recognition_why


def test_the_new_flags_are_declared_with_their_kinds() -> None:
    flags = with_knobs(rtabmap_frame.FLAGS, load_knobs("rtabmap_frame"))
    assert flags.flag("registration_backend").choices == ("service", "local", "auto")
    assert flags.flag("registration_backend").env == "PEPIN_REGISTRATION_BACKEND"
    assert flags["registration_timeout_s"] == 1.0 and flags["xfeat_top_k"] == 2048
    assert flags.flag("place_recognition").choices == ("words", "descriptor")
    assert flags["place_recognition"] == "words", "a default flips after a drive, not before"
    assert flags["descriptor_null_share"] == 0.5
    assert flags.flag("descriptor_null_share").range == (0.0, 1.0)
    names = ("registration_backend", "place_recognition", "descriptor_null_share")
    assert all(flags.flag(n).live for n in names)
