"""The one camera channel: camera_stream's rectified left eye, fanned out on this Mac.

camera_stream (pepin-vslam) is the one reader of the board's camera. After it rectifies a frame
it hands the left eye to :class:`pepin.perception.feed.FrameFeed`, whose own thread encodes it
to JPEG and POSTs it to the perception service (:mod:`pepin.perception_service`, a launchd job of
``ros/models.sh`` on 127.0.0.1:8792). The service keeps the last seconds of frames in a ring,
finds faces (SCRFD) and people (Apple Vision) on the newest frame at ``detect_hz``, and answers
everyone who wants the camera: the speaker gaze and person recognition by records (boxes,
landmarks, the head angles that centre each one, the board's exposure stamp), the ``see`` tool by
the ring's JPEG of the first still frame, person recognition's ArcFace by ``/embed`` on demand.
Nothing in the voice process decodes a picture.

The pieces, each importable without the others' dependencies:

* :mod:`pepin.perception.record`: what a frame becomes (:class:`Record`), and the optics that
  name its pixel space;
* :mod:`pepin.perception.ring`: the frames by number, and the newest record, with waits;
* :mod:`pepin.perception.detect`: the detectors behind protocols and the paced detection loop;
* :mod:`pepin.perception.feed`: the sender inside camera_stream (no ROS, no detector);
* :mod:`pepin.perception.client`: the service as consumers see it (:class:`Perception`), its
  HTTP client, the health watch behind the voice loop's vision cell;
* :mod:`pepin.perception.source`: the live ``vision.source`` switch (service or snapshot);
* :mod:`pepin.perception.fakes`: a scripted service for tests.
"""

from __future__ import annotations

PORT = 8792
URL = f"http://127.0.0.1:{PORT}"  # as this Mac's processes reach it
CONTAINER_URL = f"http://host.docker.internal:{PORT}"  # as pepin-vslam reaches it
