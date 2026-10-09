"""The face sensor: who is in front of the robot, where, whether they look at it and talk, and how
their face looks over the last seconds, as one JSON line a second.

A separate process (``scripts/face_sensor.py run``) watches the head camera's left eye at its own
rate (3 fps by default), takes the best face (the widest box) through SCRFD detection, ArcFace
identity against the enrolled family (``data/faces``), EmotiEffLib's expression net and
MediaPipe's landmarker, and rewrites ``data/face/latest.json`` every second::

    {"stamp": 1791486500.64, "faces": 1, "who": {"id": "artem", "cos": 0.84, "votes": "5/5"},
     "face_px": 62, "bearing_deg": {"pan": 4.1, "tilt": -2.0},
     "expr": {"top": "neutral", "p_3s": 0.66, "valence_3s": 0.12, "arousal_3s": -0.25},
     "attention": {"looking": true, "talking": false},
     "quality": {"sharpness": 140, "brightness": 96, "gate": "ok", "det": 0.88}, "fps": 3.0}

``stamp`` is the frame's capture time; ``who.id`` is a name, ``guest-N`` (a face this session
has seen but nobody enrolled) or ``null`` while the vote is undecided; ``bearing_deg`` is where to
turn the neck (pan positive left, tilt positive down, from the eye's optical axis); ``expr`` is
``null`` for a face under 48 px or turned away; every face field is ``null`` without a face.

The voice loop reads it with :func:`latest`, which answers ``None`` for a line older than 3 s.
"""

from pepin.face_sensor.publish import latest

__all__ = ["latest"]
