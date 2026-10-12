"""Person recognition: who is speaking to the robot, by face and voice, against the people it
has met; and meeting someone new ("как тебя зовут?" -> "Саша" -> from now on "привет, Саша").

The pieces, each small and testable alone:

* :mod:`pepin.people.store`: the people and their samples behind :class:`PersonStore`
  (SQLite on this Mac; an in-memory twin; the companion's profile database later);
* :mod:`pepin.people.voice`: one speaker embedding per utterance (WeSpeaker ResNet34-LM);
* :mod:`pepin.people.face`: the speaker's face picked, gated and embedded (ArcFace alone);
* :mod:`pepin.people.matcher`: one modality against everyone's centroids, with a margin;
* :mod:`pepin.people.fusion`: face and voice into sure / tentative / new / unknown;
* :mod:`pepin.people.session`: one conversation's speakers, and how a name binds to one;
* :mod:`pepin.people.tag`: the speaker in the tag line (``[говорит: Саша]``);
* :mod:`pepin.people.worker`: all of it in the voice cascade behind the live switch ``people``.

The tools ``remember_person`` and ``forget_person`` are :mod:`pepin.tools.people`. Embeddings
and names never leave this Mac.
"""
