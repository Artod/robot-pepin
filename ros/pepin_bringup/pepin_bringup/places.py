"""The room's vocabulary, kept current against a graph that bends: ``home``, ``bookshelf``,
``printer`` as poses RTAB-Map's own nodes carry.

WHY A COORDINATE IS NOT A PLACE ANY MORE. Until now a place was x, y and a heading written into
``ros/maps/<map>.places.yaml`` beside a frozen grid. Under World R the map IS RTAB-Map's
loop-closed graph, and a closure BENDS it: the wall moves in ``map``, and a coordinate written
before the closure names a spot beside the furniture instead of in front of it. So a place is
:class:`pepin.places.GraphPlace` — THE POSE THE CART HAD RELATIVE TO A LABELLED NODE when it was
marked — and this node is the one thing that turns those back into coordinates, recomputed from
``/rtabmap/mapGraph`` every time the graph moves. A place rides its node; the node is a thing in the
room.

ONE RESPONSIBILITY, TWO DOORS. The node keeps :data:`pepin.places.PLACES_TOPIC` (``/places``,
latched JSON) current and serves the marking on :data:`pepin.places.MARK_TOPIC`, answering on
:data:`pepin.places.MARKED_TOPIC`. TOPICS AND NOT SERVICES, because the tool that marks and the
one that drives run ON THE BOARD, inside its container, and only topics are routed across the zenoh
bridge in this deployment (:mod:`pepin.deployment`'s per-direction lists; the only service of this
laptop's that crosses is the global costmap's, and actions over the bridge aborted the navigation
container in the first place). Latched for ``/places`` and for the answer, so the last known
vocabulary survives a WiFi drop and a tool that starts late still hears it.

HOW A MARK LEARNS WHICH NODE IT LANDED ON — and it has to ask, because RTAB-Map will not say.
``set_label`` with ``node_id`` 0 labels "the last node" while mapping and, beside a loaded
database, the node NEAREST the last localisation within ``RGBD/LocalRadius``
(rtabmap/core/Rtabmap.cpp's ``labelLocation``, the branch on ``Mem/IncrementalMemory``); its
response carries no fields at all (rtabmap_msgs/srv/SetLabel.srv), so success and failure look
identical. And a label another node already holds is REFUSED rather than moved (``Memory.cpp:2718``,
"Another node %d has already label ... cannot set it"). So the sequence is: ``remove_label`` the
name first so a re-mark is not silently refused, then ``set_label`` 0, then ``list_labels`` — which
is the only thing that answers WHICH id now carries it (ListLabels.srv:4-5) — and the offset is
measured against that node's own optimised pose.

THE ID AND THE OFFSET ARE OURS TO KEEP. The label on the node is the courtesy copy and not the
storage: beside a loaded database nothing is written at all, and a label on a node that is in the
working memory only flips a dirty bit until that node leaves it (rtabmap/core/Signature.h:76), so a
label set in a localising session never reaches the file. The book therefore lives beside the
database as ``<database>.places.json`` (:func:`pepin.places.graph_places_path`) and is loaded at
start.

A MARK IS REFUSED ON THE SAME EVIDENCE A DRIVE STARTS ON (:data:`pepin.watch.DRIVE_SIGMA_M`): the
tracker's own error bar at this moment, from ``/tracker_pose``. A place marked while the cart does
not know where it stands is a place nobody can drive to afterwards — and the graph must have
recognised the room at all, or ``set_label`` 0 has no node to hang the name on.

The flags (:data:`FLAGS`, ``ros/flags.sh set places <flag> <value>``): ``publish_places``,
``label_nodes``, ``mark_sigma_m``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import MapGraph
from rtabmap_msgs.srv import ListLabels, RemoveLabel, SetLabel
from std_msgs.msg import String

from pepin.deployment import DEFAULT_LOCALIZER
from pepin.flags import Flag, FlagSet
from pepin.odometry import Pose2D
from pepin.places import (
    MARK_TOPIC,
    MARKED_TOPIC,
    PLACES_TOPIC,
    GraphPlace,
    graph_places_path,
    load_graph_places,
    places_json,
    save_graph_places,
)
from pepin.watch import DRIVE_SIGMA_M
from pepin_bringup.msgs import stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

RATE_HZ = 1.0  # the republish beat: /rtabmap/mapGraph itself comes about once a second
MAP_FRAME = "map"
BASE_FRAME = "base_link"
CART_TF_PERIOD_S = 0.2  # 5 Hz, the rate the tracker published /tracker_pose at
GRAPH_TOPIC = "/rtabmap/mapGraph"
TRACKER_POSE_TOPIC = "/tracker_pose"
# The graph database the places hang on, as the containers see it: the same literal
# pepin_bringup.depth_fusion uses for the volume's own frame. A node id means nothing without it.
DATABASE = "/maps/rtabmap.db"
RTABMAP_NODE = "/rtabmap/rtabmap"
SET_LABEL_SERVICE = f"{RTABMAP_NODE}/set_label"
LIST_LABELS_SERVICE = f"{RTABMAP_NODE}/list_labels"
REMOVE_LABEL_SERVICE = f"{RTABMAP_NODE}/remove_label"
# How old the tracker's word about the cart may be for a mark to stand on it. The tracker publishes
# its pose on every update and its sigma every check period, so three seconds of nothing is the
# board not talking to us — the same patience the goal client's preflight gives it.
BELIEF_FRESH_S = 3.0
# ``node_id`` 0 is RTAB-Map's own "the node I am at" (SetLabel.srv:2, "Set node_id = 0 to set label
# to last node"; beside a loaded database, the node nearest the last localisation).
HERE = 0
MARKS_REMEMBERED = 32  # ids of requests already answered, so a repeat cannot mark twice

FLAGS = FlagSet(
    Flag(
        "publish_places",
        True,
        description=f"the resolved places are published on {PLACES_TOPIC} whenever the graph"
        " moves; off, the book is still kept and marked but nothing is published and every"
        " consumer falls back to the coordinates beside the map",
        why="default by design, unmeasured: it is the one output of this node. The switch exists"
        " to prove which book a drive resolved a name from — with it off, ros/go.sh printer must"
        " print the fallback warning and reach the same furniture, which is the A/B between a"
        " place that rides its node and a coordinate that does not",
        on_when="always: a coordinate written before a loop closure names a spot beside the"
        " furniture rather than in front of it",
        off_when="for that A/B, and if a resolved place is ever seen further from the furniture"
        " than the file's own coordinate",
    ),
    Flag(
        "label_nodes",
        True,
        description="a mark also sets RTAB-Map's own label on the node (set_label), which is what"
        " makes the place a thing in its tools and in its set_goal; off, only our own book records"
        " the node id and the offset",
        why="on, because the label costs one service call and is the only name RTAB-Map itself"
        " understands. It is NOT the storage and this node never reads it back as one: beside a"
        " loaded database nothing is written (Mem/IncrementalMemory false) and a label on a node in"
        " the working memory only flips a dirty bit (Signature.h:76), so a label set while"
        " localising never reaches the file. list_labels is still called with it off — it is the"
        " only way to learn which node RTAB-Map considers the current one",
        on_when="always beside a database that may be written; it costs nothing where it cannot",
        off_when="on a database that must not be touched at all, even by a dirty bit",
    ),
    Flag(
        "mark_sigma_m",
        DRIVE_SIGMA_M,
        range=(0.0, 2.0),
        description="the widest the tracker's own error bar may be, metres, for a mark to be"
        " taken: past it the mark is refused with the reading in the answer",
        why=f"{DRIVE_SIGMA_M}, the same bar a DRIVE starts on (pepin.watch.DRIVE_SIGMA_M): a place"
        " marked while the cart does not know where it stands is a place nobody can drive to"
        " afterwards, and the two thresholds must be one number or a mark can be taken at a pose"
        " no goal would be sent from. Read from the tracker's covariance and not from the lidar's"
        " fit, which is 0.00 by construction on a camera-only stack and refused every mark there"
        " (2026-09-15)",
        on_when="always",
        off_when="raise it only to mark a place in a corner where the pose is never sharp — and"
        " then read the sigma the answer prints before believing the place",
    ),
)


@dataclass(frozen=True)
class MarkRequest:
    """ "Mark where I stand as NAME": what the board's tool publishes on
    :data:`pepin.places.MARK_TOPIC`.

    ``id`` is the tool's own handle for this request, echoed in the answer, so a tool can tell its
    own answer from a latched one of somebody else's and this node can refuse to mark twice on a
    redelivered message.
    """

    name: str
    id: str = ""

    def to_json(self) -> str:
        """The request as the one message that crosses the bridge."""
        return json.dumps({"name": self.name, "id": self.id})

    @classmethod
    def from_json(cls, text: str) -> MarkRequest | None:
        """One such message back, or ``None`` when it is not one or names nothing: a request with
        no name is not a request, and this node acts on nothing it could not read."""
        try:
            heard = json.loads(text)
            name = str(heard["name"]).strip()
        except (TypeError, ValueError, KeyError):
            return None
        return cls(name=name, id=str(heard.get("id", ""))) if name else None


class Places(Node):
    """Keeps ``/places`` current against RTAB-Map's graph, and marks a place where the cart is."""

    def __init__(self) -> None:
        super().__init__("places")
        # Which database the places hang on, declared before the flags the way every node here
        # declares its startup parameters: a node id means nothing without it, so a SLAM session
        # writing its own file gets its own book and can never claim the known map's names.
        self._database = Path(str(self.declare_parameter("database", DATABASE).value))
        # WHERE THE CART'S POSE COMES FROM (PEPIN_LOCALIZER, pepin.deployment.localizer).
        # Declared before the flags, like every startup parameter here: rclpy runs the switches'
        # callback on declarations too and refuses a name outside their table.
        #   "tracker": the board's tracker publishes /tracker_pose with its own covariance, and a
        # mark is measured from it and refused on its error bar. "rtabmap": nothing publishes that
        # topic, so `ros/go.sh mark` would refuse for ever ("no pose on /tracker_pose for N s").
        # The pose is then read from map -> base_link, the same edge every consumer composes —
        # and with it the error bar is GONE, because TF carries no covariance, so the mark_sigma_m
        # gate has nothing to read and a mark rests on the freshness of the transform alone.
        self._localizer = str(self.declare_parameter("localizer", DEFAULT_LOCALIZER).value)
        self._switches = Switches(self, FLAGS)
        self._book = graph_places_path(self._database)
        self._places: dict[str, GraphPlace] = load_graph_places(self._book)
        self._poses: dict[int, Pose2D] = {}  # the graph's optimised node poses, in map
        self._graphs = 0
        self._cart: Pose2D | None = None  # where the board's tracker says the cart is...
        self._sigma_m: float | None = None  # ...and the worse of its two position error bars
        self._cart_at = -math.inf  # ...and when that reached us, by our clock
        self._answered: list[str] = []  # request ids already dealt with
        self._marks = 0  # marks taken...
        self._refused = 0  # ...and refused
        self._last = ""  # the last answer's own words, for the report line
        self._published = ""  # the payload last published, so nothing is republished unchanged
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._places_pub = self.create_publisher(String, PLACES_TOPIC, latched)
        self._marked_pub = self.create_publisher(String, MARKED_TOPIC, latched)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(MapGraph, GRAPH_TOPIC, self._on_graph, 2)
        self.create_subscription(
            PoseWithCovarianceStamped, TRACKER_POSE_TOPIC, self._on_cart, reliable
        )
        # ...and its replacement where no tracker publishes one. On a timer because TF is a
        # lookup, at the rate the tracker spoke at; created only in that role, so a stack with a
        # tracker pays neither the timer nor the /tf subscription behind the listener.
        self._tf: TfLookup | None = None if self._localizer == "tracker" else TfLookup(self)
        if self._tf is not None:
            self.create_timer(CART_TF_PERIOD_S, self._cart_from_tf)
        # NOT latched: a request redelivered to a node that restarted must not mark a second time.
        # The id guards that too, and belt and braces is right for something that writes a file.
        self.create_subscription(String, MARK_TOPIC, self._on_mark, reliable)
        self._labeller = self.create_client(SetLabel, SET_LABEL_SERVICE)
        self._lister = self.create_client(ListLabels, LIST_LABELS_SERVICE)
        self._forgetter = self.create_client(RemoveLabel, REMOVE_LABEL_SERVICE)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"places up: {len(self._places)} in {self._book}, republished on {PLACES_TOPIC}"
            f" whenever {GRAPH_TOPIC} moves them; mark on {MARK_TOPIC}, answered on"
            f" {MARKED_TOPIC}; a place is the cart's pose relative to its labelled node, so it"
            f" rides the node when the graph bends; flags: {self._switches.state()}"
        )

    # ---- inputs ------------------------------------------------------------------------------
    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's optimised graph: the node poses every place is resolved against
        (``poses_id`` beside ``poses``, rtabmap_msgs/msg/MapGraph.msg:11-12)."""
        self._graphs += 1
        self._poses = {
            int(node): Pose2D(
                float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation)
            )
            for node, pose in zip(msg.poses_id, msg.poses, strict=False)
        }

    def _on_cart(self, msg: PoseWithCovarianceStamped) -> None:
        """Where the board's tracker says the cart is, and how sharply: the pose a mark is measured
        from, and the error bar that decides whether it may be taken at all."""
        p = msg.pose.pose
        self._cart = Pose2D(float(p.position.x), float(p.position.y), yaw_of(p.orientation))
        covariance = list(msg.pose.covariance)
        self._sigma_m = math.sqrt(max(max(covariance[0], covariance[7]), 0.0))
        self._cart_at = self._now()

    def _cart_from_tf(self) -> None:
        """The cart's pose from ``map -> base_link``, where no tracker publishes one.

        ``_sigma_m`` stays ``None``: TF carries no covariance and an invented error bar would
        let ``mark_sigma_m`` pass a pose nobody measured. The refusal for a stale transform is
        the same one the tracker's silence produces (:meth:`_refusal`).
        """
        if self._tf is None:
            return
        transform = self._tf.transform(MAP_FRAME, BASE_FRAME, timeout_s=0.0)
        if transform is None:
            return
        t = transform.transform
        self._cart = Pose2D(float(t.translation.x), float(t.translation.y), yaw_of(t.rotation))
        self._sigma_m = None
        self._cart_at = self._now()

    def _now(self) -> float:
        """This node's clock in seconds."""
        return stamp_seconds(self.get_clock().now().to_msg())

    # ---- the marking -------------------------------------------------------------------------
    def _on_mark(self, msg: String) -> None:
        """ "Mark where I stand as NAME": the whole sequence, or one refusal that says why.

        The order matters and each step is RTAB-Map's own contract. ``remove_label`` first, because
        ``set_label`` REFUSES a name another node already holds and says so only in its log
        (``Memory.cpp:2718``) — without this a re-mark would silently keep the old node. Then
        ``set_label`` with :data:`HERE`, which is RTAB-Map's "the node I am at". Then
        ``list_labels``, the only thing that answers WHICH node took it, and the offset is measured
        there (:meth:`_measure`).
        """
        request = MarkRequest.from_json(msg.data)
        if request is None:
            return
        if request.id and request.id in self._answered:
            return  # already dealt with: a redelivered request must not mark twice
        refusal = self._refusal()
        if refusal is not None:
            self._answer(request, ok=False, detail=refusal)
            return
        if self._switches.on("label_nodes") and not (
            self._forgetter.service_is_ready() and self._labeller.service_is_ready()
        ):
            self._answer(request, ok=False, detail=f"{SET_LABEL_SERVICE} is not answering")
            return
        if not self._lister.service_is_ready():
            self._answer(request, ok=False, detail=f"{LIST_LABELS_SERVICE} is not answering")
            return
        if not self._switches.on("label_nodes"):
            self._list(request)
            return
        # ONE AFTER THE OTHER, each on the answer of the one before. Sent together they race:
        # live on 2026-09-19 RTAB-Map logged "List labels service: 1 labels found" 4 ms BEFORE
        # "Set label "home" to last node", so the list did not hold the name yet and a mark
        # RTAB-Map had taken was reported as refused.
        forgotten = self._forgetter.call_async(RemoveLabel.Request(label=request.name))
        forgotten.add_done_callback(lambda _done: self._label(request))

    def _label(self, request: MarkRequest) -> None:
        """Second step of a mark, once the old holder of the name has let go of it."""
        labelled = self._labeller.call_async(
            SetLabel.Request(node_id=HERE, node_label=request.name)
        )
        labelled.add_done_callback(lambda _done: self._list(request))

    def _list(self, request: MarkRequest) -> None:
        """Last step: ask which node holds the name now, and measure the cart against it."""
        future = self._lister.call_async(ListLabels.Request())
        future.add_done_callback(lambda done: self._measure(request, done.result()))

    def _refusal(self) -> str | None:
        """Why this moment may not be marked, in one phrase for the answer, or ``None`` when it
        may: no graph to hang a name on, no word from the tracker, or a pose too uncertain to be
        worth remembering (``mark_sigma_m``, the same bar a drive starts on)."""
        if not self._poses:
            return (
                f"nothing on {GRAPH_TOPIC} yet: the graph has not recognised this room, so"
                " there is no node to hang a name on"
            )
        age = self._now() - self._cart_at
        if self._cart is None or age > BELIEF_FRESH_S:
            heard = TRACKER_POSE_TOPIC if self._localizer == "tracker" else "map -> base_link"
            return f"no pose on {heard} for {age:.1f} s: the board is not talking"
        limit = float(self._switches["mark_sigma_m"])
        if self._sigma_m is not None and self._sigma_m > limit:
            return (
                f"the pose here is known to {self._sigma_m:.2f} m, over the {limit:.2f} m a mark"
                " needs; stand still, or run ros/goto.sh relocalize first"
            )
        return None

    def _measure(self, request: MarkRequest, labels: Any) -> None:
        """``list_labels`` came back: which node carries this name, and what the cart's offset from
        it is. That pair IS the place (:meth:`pepin.places.GraphPlace.measured`)."""
        node = self._labelled(request.name, labels)
        cart = self._cart
        if node is None:
            self._answer(
                request,
                ok=False,
                detail=f"{LIST_LABELS_SERVICE} does not list {request.name!r}: RTAB-Map took no"
                " label here (beside a loaded database it labels the node nearest its last"
                " localisation, and there was none)",
            )
            return
        pose = self._poses.get(node)
        if pose is None or cart is None:
            self._answer(
                request,
                ok=False,
                detail=f"node {node} carries the label but is not in {GRAPH_TOPIC}: nothing to"
                " measure the offset against",
            )
            return
        place = GraphPlace.measured(request.name, node, pose, cart, stamp=self._cart_at)
        self._places[request.name] = place
        save_graph_places(self._book, self._places)
        self._marks += 1
        self._publish(force=True)
        self._answer(
            request,
            ok=True,
            detail=f"marked {request.name!r} at ({cart.x:+.2f}, {cart.y:+.2f},"
            f" {math.degrees(cart.theta):+.0f} deg) as {place.reach_m * 100:.0f} cm from node"
            f" {node}, which is what it will ride when the graph bends; {len(self._places)} places"
            f" in {self._book}",
        )

    @staticmethod
    def _labelled(name: str, labels: Any) -> int | None:
        """Which node carries ``name``, off a ``list_labels`` answer (``ids`` beside ``labels``,
        two parallel arrays), or ``None`` when none does or the service did not answer."""
        if labels is None:
            return None
        for node, label in zip(labels.ids, labels.labels, strict=False):
            if str(label) == name:
                return int(node)
        return None

    def _answer(self, request: MarkRequest, ok: bool, detail: str) -> None:
        """Say what came of a mark on :data:`pepin.places.MARKED_TOPIC`, latched, echoing the
        request's own id — RTAB-Map's ``set_label`` answers with nothing at all, so this is the
        only thing the operator and the board's tool ever hear."""
        if not ok:
            self._refused += 1
        self._last = f"{'marked' if ok else 'REFUSED'} {request.name!r}: {detail}"
        if request.id:
            self._answered = [*self._answered[-(MARKS_REMEMBERED - 1) :], request.id]
        self._marked_pub.publish(
            String(
                data=json.dumps(
                    {"id": request.id, "name": request.name, "ok": ok, "detail": detail}
                )
            )
        )
        self.get_logger().info(f"places: {self._last}")

    # ---- the output --------------------------------------------------------------------------
    def _publish(self, force: bool = False) -> None:
        """Every place the graph can still answer for, as a coordinate in the map right now.

        Published when the payload CHANGES — which is what "recomputed whenever the graph moves"
        means in practice: the graph arrives about once a second and a place whose node has not
        moved resolves to the same numbers, so an unchanged book costs nothing. ``force`` is a fresh
        mark, which must go out even if its numbers happen to match.
        """
        if not self._switches.on("publish_places"):
            return
        payload = places_json(self._places, self._poses)
        if payload == self._published and not force:
            return
        self._published = payload
        self._places_pub.publish(String(data=payload))

    def _report(self) -> None:
        """Every 30 s: how many places are known and how many the graph can answer for, how many
        marks were taken and refused, the last answer, and the graph behind all of it."""
        answerable = len(json.loads(self._published or "{}"))
        self.get_logger().info(
            f"places: {len(self._places)} in {self._book}, {answerable} the graph can place"
            f" ({self._graphs} graphs, {len(self._poses)} nodes); {self._marks} marked,"
            f" {self._refused} refused; last {self._last or 'nothing asked yet'};"
            f" cart {self._cart_text()}; flags: {self._switches.state()}"
        )

    def _cart_text(self) -> str:
        """Where the tracker says the cart is and how sharply, for the report line."""
        cart = self._cart
        if cart is None:
            return f"no {TRACKER_POSE_TOPIC} yet"
        sigma = "no covariance" if self._sigma_m is None else f"+- {self._sigma_m * 100:.0f} cm"
        return (
            f"({cart.x:+.2f}, {cart.y:+.2f}, {math.degrees(cart.theta):+.0f} deg) {sigma},"
            f" {self._now() - self._cart_at:.1f} s ago"
        )


def main() -> None:
    spin_main(Places)


if __name__ == "__main__":
    main()
