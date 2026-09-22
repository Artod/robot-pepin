"""The places node under the ROS stubs: the vocabulary the graph answers for, and the marking.

rclpy, tf2_ros, rtabmap_msgs and RTAB-Map's three label services are faked (``ros_stubs``) and the
node is built and driven here as on the laptop. Every topic and service name is a LITERAL, because
the value of this node is that the board's tool and RTAB-Map both find it at the name it claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()


@pytest.fixture(autouse=True)
def _the_tracker_stack() -> Iterator[None]:
    """Every test in this file describes the arrangement where the BOARD'S TRACKER owns
    ``map -> odom`` (``PEPIN_LOCALIZER=tracker``), which is what these nodes were written for and
    what stays reachable. The other role — RTAB-Map on the laptop owning the frame, no tracker
    anywhere — has its own file, tests/unit/test_one_localiser.py."""
    with ros_stubs.parameters(localizer="tracker"):
        yield


from pepin_bringup.places import (  # noqa: E402
    FLAGS,
    GRAPH_TOPIC,
    HERE,
    LIST_LABELS_SERVICE,
    REMOVE_LABEL_SERVICE,
    SET_LABEL_SERVICE,
    TRACKER_POSE_TOPIC,
    MarkRequest,
    Places,
)

from pepin.places import (  # noqa: E402
    MARK_TOPIC,
    MARKED_TOPIC,
    PLACES_TOPIC,
    GraphPlace,
    load_graph_places,
    save_graph_places,
)
from pepin.watch import DRIVE_SIGMA_M  # noqa: E402

Build = Callable[..., Places]


def _pose_msg(x: float, y: float, yaw_deg: float = 0.0) -> Any:
    yaw = math.radians(yaw_deg)
    return ros_stubs.Pose(
        position=ros_stubs.Point(x=x, y=y),
        orientation=ros_stubs.Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
    )


@pytest.fixture
def build(tmp_path: Path) -> Iterator[Build]:
    """A places node whose book lives in a temporary directory, with RTAB-Map's three services
    ready unless a test says otherwise."""
    made: list[Places] = []

    def make(services: bool = True, book: dict[str, GraphPlace] | None = None, **params: Any):
        database = tmp_path / "rtabmap.db"
        if book is not None:
            save_graph_places(tmp_path / "rtabmap.places.json", book)
        with ros_stubs.parameters(database=str(database), **params):
            node = Places()
        node.book_path = tmp_path / "rtabmap.places.json"  # type: ignore[attr-defined]
        made.append(node)
        for name in (SET_LABEL_SERVICE, LIST_LABELS_SERVICE, REMOVE_LABEL_SERVICE):
            node.service_clients[name].ready = services
        node.service_clients[LIST_LABELS_SERVICE].response = ros_stubs.ListLabels.Response()
        return node

    yield make
    for node in made:
        node.destroy_node()


def graph(node: Places, poses: dict[int, tuple[float, float, float]]) -> None:
    """One /rtabmap/mapGraph: the optimised pose of every node, in the map frame."""
    node.subs[GRAPH_TOPIC][1](
        ros_stubs.MapGraph(
            poses_id=list(poses),
            poses=[_pose_msg(*values) for values in poses.values()],
        )
    )


def cart(node: Places, x: float, y: float, yaw_deg: float = 0.0, sigma_m: float = 0.02) -> None:
    """The board's tracker saying where the cart is and how sharply."""
    msg = ros_stubs.PoseWithCovarianceStamped()
    msg.pose.pose = _pose_msg(x, y, yaw_deg)
    covariance = list(msg.pose.covariance)
    covariance[0] = covariance[7] = sigma_m * sigma_m
    msg.pose.covariance = covariance
    node.subs[TRACKER_POSE_TOPIC][1](msg)


def labelled(node: Places, **labels: int) -> None:
    """What ``list_labels`` will answer with: the ids and the names as two parallel arrays."""
    node.service_clients[LIST_LABELS_SERVICE].response = ros_stubs.ListLabels.Response(
        ids=list(labels.values()), labels=list(labels)
    )


def mark(node: Places, name: str, request_id: str = "tool-1") -> dict[str, Any]:
    """Ask for a mark the way the board's tool does, and read the answer back."""
    node.subs[MARK_TOPIC][1](ros_stubs.String(data=MarkRequest(name, request_id).to_json()))
    answer: dict[str, Any] = json.loads(node.pubs[MARKED_TOPIC].sent[-1].data)
    return answer


def published(node: Places) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(node.pubs[PLACES_TOPIC].sent[-1].data)
    return payload


# ---- the doors -------------------------------------------------------------------------------
def test_the_node_reads_three_topics_and_writes_two_at_the_names_both_ends_spell(
    build: Build,
) -> None:
    node = build()
    assert set(node.subs) == {"/rtabmap/mapGraph", "/tracker_pose", "/places/mark"}
    assert set(node.pubs) == {"/places", "/places/marked"}
    assert (PLACES_TOPIC, MARK_TOPIC, MARKED_TOPIC) == ("/places", "/places/mark", "/places/marked")
    assert node.pubs[PLACES_TOPIC].qos.rest["durability"] == "transient_local", (
        "latched: the last vocabulary must survive a WiFi drop"
    )
    assert node.pubs[MARKED_TOPIC].qos.rest["durability"] == "transient_local"
    assert node.subs[MARK_TOPIC][0] is ros_stubs.String
    assert set(node.service_clients) == {
        "/rtabmap/rtabmap/set_label",
        "/rtabmap/rtabmap/list_labels",
        "/rtabmap/rtabmap/remove_label",
    }
    assert (SET_LABEL_SERVICE, LIST_LABELS_SERVICE, REMOVE_LABEL_SERVICE) == (
        "/rtabmap/rtabmap/set_label",
        "/rtabmap/rtabmap/list_labels",
        "/rtabmap/rtabmap/remove_label",
    )


def test_a_mark_request_that_names_nothing_is_not_a_request(build: Build) -> None:
    node = build()
    for text in ("", "not json", "{}", '{"name": "   "}', "[1]"):
        node.subs[MARK_TOPIC][1](ros_stubs.String(data=text))
    assert not node.pubs[MARKED_TOPIC].sent, "nothing to answer"
    assert MarkRequest.from_json(MarkRequest("home", "id").to_json()) == MarkRequest("home", "id")


# ---- the vocabulary --------------------------------------------------------------------------
def test_the_places_are_recomputed_when_the_graph_moves_them(build: Build) -> None:
    """The whole point of the node: a closure bends the graph, and the place rides its node
    instead of staying on the coordinate it was written at."""
    node = build(book={"printer": GraphPlace("printer", 91, 0.60, 0.0, 0.0)})
    graph(node, {91: (1.0, 2.0, 0.0)})
    node._publish()
    assert published(node)["printer"]["x"] == pytest.approx(1.6)
    assert published(node)["printer"]["node"] == 91

    graph(node, {91: (1.0, 2.5, 0.0)})  # the graph moved the node half a metre
    node._publish()
    assert published(node)["printer"]["y"] == pytest.approx(2.5)


def test_an_unchanged_graph_is_not_republished(build: Build) -> None:
    """The graph arrives about once a second and a place whose node has not moved resolves to the
    same numbers: a latched topic republished every second is traffic for nothing."""
    node = build(book={"home": GraphPlace("home", 3, 0.0, 0.0, 0.0)})
    graph(node, {3: (0.0, 0.0, 0.0)})
    node._publish()
    once = len(node.pubs[PLACES_TOPIC].sent)
    for _ in range(5):
        graph(node, {3: (0.0, 0.0, 0.0)})
        node._publish()
    assert len(node.pubs[PLACES_TOPIC].sent) == once


def test_the_book_is_loaded_from_beside_the_database_at_start(build: Build) -> None:
    node = build(book={"home": GraphPlace("home", 3, 0.1, 0.2, 45.0)})
    assert set(node._places) == {"home"}
    assert "1 in " in node.logger.texts("info")[0]


# ---- the marking -----------------------------------------------------------------------------
def test_a_mark_labels_the_node_here_and_records_the_offset_from_it(build: Build) -> None:
    """The sequence, in RTAB-Map's own terms: remove_label first (a name another node holds is
    REFUSED, not moved), then set_label with node_id 0, then list_labels — the only thing that
    answers WHICH node took it."""
    node = build()
    graph(node, {88429: (2.0, 0.0, 0.0)})
    cart(node, 2.6, 0.0, 90.0)
    labelled(node, bookshelf=88429)
    answer = mark(node, "bookshelf")

    forget = node.service_clients[REMOVE_LABEL_SERVICE].calls[-1]
    assert forget.label == "bookshelf", "so a re-mark is not silently refused"
    label = node.service_clients[SET_LABEL_SERVICE].calls[-1]
    assert (label.node_id, label.node_label) == (0, "bookshelf")
    assert HERE == 0, "rtabmap's own 'the node I am at' (SetLabel.srv:2)"
    assert node.service_clients[LIST_LABELS_SERVICE].calls, "the only way to learn the id"

    assert answer["ok"] is True and answer["id"] == "tool-1" and answer["name"] == "bookshelf"
    assert "60 cm from node 88429" in answer["detail"]
    place = node._places["bookshelf"]
    assert place.node == 88429 and place.reach_m == pytest.approx(0.60)
    assert place.dtheta_deg == pytest.approx(90.0)


def test_a_mark_is_written_beside_the_database_and_published_at_once(build: Build) -> None:
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node, home=7)
    mark(node, "home")
    assert load_graph_places(node.book_path)["home"].node == 7, "a restart keeps it"
    assert published(node)["home"]["x"] == pytest.approx(1.0)


def test_a_mark_rides_the_node_the_moment_the_graph_bends(build: Build) -> None:
    """Marked, then a closure: the published coordinate follows the furniture."""
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node, home=7)
    mark(node, "home")
    graph(node, {7: (0.0, 0.40, 0.0)})
    node._publish()
    assert published(node)["home"]["y"] == pytest.approx(0.40)


def test_a_repeated_request_marks_once(build: Build) -> None:
    """The board's tool republishes until it hears an answer, because a request nobody received is
    a mark that silently never happened. The id is what makes that free."""
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node, home=7)
    mark(node, "home", request_id="tool-9")
    calls = len(node.service_clients[SET_LABEL_SERVICE].calls)
    for _ in range(3):
        node.subs[MARK_TOPIC][1](ros_stubs.String(data=MarkRequest("home", "tool-9").to_json()))
    assert len(node.service_clients[SET_LABEL_SERVICE].calls) == calls
    assert node._marks == 1


# ---- the refusals ----------------------------------------------------------------------------
def test_a_mark_is_refused_where_the_graph_has_not_recognised_the_room(build: Build) -> None:
    """set_label 0 has no node to hang a name on, and a place hung on nothing is worse than a
    refusal."""
    node = build()
    cart(node, 1.0, 0.0)
    answer = mark(node, "home")
    assert answer["ok"] is False and GRAPH_TOPIC in answer["detail"]
    assert not node.service_clients[SET_LABEL_SERVICE].calls, "nothing was labelled"


def test_a_mark_is_refused_where_the_board_is_not_talking(build: Build) -> None:
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    answer = mark(node, "home")
    assert answer["ok"] is False and TRACKER_POSE_TOPIC in answer["detail"]


def test_a_mark_is_refused_on_the_same_bar_a_drive_starts_on(build: Build) -> None:
    """A place marked while the cart does not know where it stands is a place nobody can drive to
    afterwards, so the two thresholds are one number."""
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0, sigma_m=DRIVE_SIGMA_M + 0.05)
    answer = mark(node, "home")
    assert answer["ok"] is False and "over the 0.25 m a mark needs" in answer["detail"]
    assert FLAGS["mark_sigma_m"] == DRIVE_SIGMA_M == 0.25

    cart(node, 1.0, 0.0, sigma_m=DRIVE_SIGMA_M - 0.05)
    labelled(node, home=7)
    assert mark(node, "home", request_id="tool-2")["ok"] is True


def test_a_mark_rtabmap_took_no_label_for_is_refused_and_says_so(build: Build) -> None:
    """``set_label``'s response carries no fields at all, so the only honest test of success is
    that ``list_labels`` now lists the name — and beside a loaded database it labels the node
    nearest its last localisation, of which there may have been none."""
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node)  # nothing came back
    answer = mark(node, "home")
    assert answer["ok"] is False and "does not list 'home'" in answer["detail"]
    assert "home" not in node._places


def test_a_label_on_a_node_the_graph_does_not_hold_is_refused(build: Build) -> None:
    node = build()
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node, home=999)  # a node from a session this graph no longer carries
    answer = mark(node, "home")
    assert answer["ok"] is False and "node 999 carries the label" in answer["detail"]


def test_a_mark_is_refused_where_rtabmaps_services_are_not_there(build: Build) -> None:
    node = build(services=False)
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    answer = mark(node, "home")
    assert answer["ok"] is False and "is not answering" in answer["detail"]


# ---- the flags -------------------------------------------------------------------------------
def test_label_nodes_off_still_asks_which_node_is_current(build: Build) -> None:
    """``list_labels`` is not the label's readback — it is the only way to learn which node
    RTAB-Map considers the current one, so it is called either way."""
    node = build(label_nodes=False)
    graph(node, {7: (0.0, 0.0, 0.0)})
    cart(node, 1.0, 0.0)
    labelled(node, home=7)
    assert mark(node, "home")["ok"] is True
    assert not node.service_clients[SET_LABEL_SERVICE].calls, "nothing was labelled"
    assert node.service_clients[LIST_LABELS_SERVICE].calls


def test_publish_places_off_keeps_the_book_and_publishes_nothing(build: Build) -> None:
    """The A/B between a place that rides its node and a coordinate that does not: off, every
    consumer falls back to the file beside the map."""
    node = build(publish_places=False, book={"home": GraphPlace("home", 3, 0.0, 0.0, 0.0)})
    graph(node, {3: (0.0, 0.0, 0.0)})
    node._publish()
    assert not node.pubs[PLACES_TOPIC].sent
    assert set(node._places) == {"home"}


def test_the_flags_are_the_three_the_report_line_prints(build: Build) -> None:
    node = build()
    assert [flag.name for flag in FLAGS] == ["publish_places", "label_nodes", "mark_sigma_m"]
    node._report()
    line = node.logger.texts("info")[-1]
    assert "publish_places=on" in line and "label_nodes=on" in line
    assert "0 marked, 0 refused" in line and "nothing asked yet" in line


class _Pending:
    """A future that finishes when the test says so — what rclpy's really is. The shared stub's
    future is already done the moment it is made, which is exactly why a race between three
    service calls sent together could not be seen in a test (2026-09-19: live, RTAB-Map answered
    ``list_labels`` 4 ms before it executed ``set_label``, and a mark it had taken read as
    refused)."""

    def __init__(self, result: Any = None) -> None:
        self._result, self._callbacks = result, []

    def add_done_callback(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def finish(self) -> None:
        for callback in self._callbacks:
            callback(self)

    def result(self) -> Any:
        return self._result


def test_the_three_calls_of_a_mark_go_one_after_the_other(build: Build) -> None:
    """``remove_label``, then ``set_label`` on ITS answer, then ``list_labels`` on the answer of
    that: the list is the only thing that says which node took the name, and asked early it does
    not hold the name yet."""
    node = build()
    graph(node, {7: (1.0, 0.0, 0.0)})
    cart(node, 1.2, 0.0)
    asked: list[tuple[str, _Pending]] = []
    for name, client in (
        ("remove", node._forgetter),
        ("set", node._labeller),
        ("list", node._lister),
    ):

        def call_async(request: Any, name: str = name) -> _Pending:
            pending = _Pending(ros_stubs.ListLabels.Response(ids=[7], labels=["desk"]))
            asked.append((name, pending))
            return pending

        client.call_async = call_async  # type: ignore[method-assign]
    node.subs["/places/mark"][1](ros_stubs.String(data=MarkRequest("desk", "tool-9").to_json()))
    assert [name for name, _ in asked] == ["remove"], "nothing else is sent until it answers"
    asked[0][1].finish()
    assert [name for name, _ in asked] == ["remove", "set"]
    asked[1][1].finish()
    assert [name for name, _ in asked] == ["remove", "set", "list"]
