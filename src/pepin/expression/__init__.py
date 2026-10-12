"""Body expression: the arm, the head, the face and the wheels as one animated character.

The brain names an INTENT inline in its text (``{laugh}``, ``{wave}``, ``{face:smile}``,
``{style:whispering}``; :mod:`pepin.expression.markers`) or through a tool of the same names; the
engine decides HOW. An intent is a composition of parametrised motion atoms, rendered with a
seeded random choice (:mod:`pepin.expression.animation`), placed on the reply's audio at the
marker's word, and played by one executor per body part (:mod:`pepin.expression.executors`).

Every body part keeps its one owner, and expression goes THROUGH that owner:

* the arm: the base server on the board, which owns the servo bus and plays a whole trajectory
  by itself (:mod:`pepin.expression.trajectory`), so a lost laptop costs nothing;
* the head: the gaze arbiter (:mod:`pepin.gaze`), by band, TTL and preemption; the speaker gaze
  holds the head as its base layer (:mod:`pepin.expression.speaker`), a head gesture is a
  relative ``clip`` played on top of whatever holds it (:mod:`pepin.expression.head`);
* the face: the board's head server (:mod:`pepin.head_link`), the newest request wins;
* the wheels: small turns in place and rocks while standing, behind the goal server's guards.

Each part has three layers, as in game animation: a BASE layer (the arm at rest, the head on
the person, the wheels still), an additive GESTURE layer (one performance per part at a time,
the newest wins, faded out when the person starts speaking) and an IDLE layer (micro-motions
while nothing else plays). The voice cascade drives it through
:class:`pepin.expression.conversation.ConversationObserver`; two live switches, ``gestures``
and ``speaker_gaze`` (:class:`pepin.expression.conversation.LiveFlag`), turn the two features
on and off.
"""
