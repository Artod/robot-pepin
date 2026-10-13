"""Person recognition in the voice cascade: the stage that names each utterance's speaker, the
live switch ``people``, the conversation's observer, the tools' door, and the speaker gaze's
face listener, in one object (:class:`People`).

Per utterance (:meth:`People.measure`, on the cascade's ``people`` worker while Whisper reads):
the voice embedding of its loud span when at least ``voice.min_s`` long, and the speaker's
face: the one the speaker gaze centred within ``face.fresh_s``, else a fresh picture from the
shared finder (the dominant face). Then, with the words known (:func:`name` on the event loop,
never for Whisper's noise line): each modality against everyone's centroids
(:mod:`pepin.people.matcher`), the fusion (:mod:`pepin.people.fusion`), the conversation's
tracks (:mod:`pepin.people.session`), the speaker part of the tag line (:mod:`pepin.people.tag`)
and, for a ``sure`` verdict, the person's new samples, written off the event loop.

Costs: nothing between conversations but the store's open file; the voice net (26 MB) stays
loaded while the switch is on; the identity net loads when the switch goes on, is let go after
``face.unload_after_s`` with no conversation, and loads again (0.1 s) at the next wake.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pepin.expression.conversation import Heard
from pepin.expression.markers import Part
from pepin.expression.speaker import FaceFinder, FaceFrame, FaceSighting
from pepin.face_sensor.identity import Gallery, Vector, centroid
from pepin.people.config import FLAG, PeopleConfig
from pepin.people.door import Outcome
from pepin.people.face import (
    FaceNet,
    IdentityNet,
    RoutedIdentity,
    ServiceIdentity,
    Shot,
    gate,
    shots,
    strong,
)
from pepin.people.fusion import Verdict, comparable, fuse
from pepin.people.matcher import Reading, galleries, judge
from pepin.people.session import Observation, Session, Track, learnable
from pepin.people.store import Kind, Modality, Person, PersonStore
from pepin.people.tag import head
from pepin.people.voice import RATE, VoiceEmbedder, samples_of, speech_span
from pepin.voice_cascade.stages import Speaker

if TYPE_CHECKING:
    from pepin.perception.client import Perception
    from pepin.perception.source import VisionSource

logger = logging.getLogger(__name__)

MAX_NAME = 40
MAX_OTHERS = 3  # other faces in view embedded per utterance


def say(line: str) -> None:
    """One line of person recognition in the cascade's output."""
    print(f"  (people: {line})", flush=True)


@dataclass
class Parts:
    """What recognition runs on: the store, the voice net (None: no voice), the identity net
    (None: no faces) and the finder that takes pictures (None: only the gaze's faces)."""

    store: PersonStore
    voice: VoiceEmbedder | None
    faces: FaceNet | None
    finder: FaceFinder | None


@dataclass(frozen=True)
class Measured:
    """One utterance's measurements, before the words are known."""

    obs: Observation
    face_strong: bool = False
    voice_strong: bool = False
    looked: str = "no"  # where the face came from: gaze, look, no
    face_gate: str = ""  # the speaker face's gate verdict ("" without one)


class People:
    """Person recognition: :class:`pepin.voice_cascade.stages.SpeakerId`, the live flag
    ``people``, a :class:`pepin.expression.conversation.ConversationObserver`, the tools' door
    and the gaze's face listener."""

    def __init__(
        self,
        build: Callable[[PeopleConfig], Parts],
        *,
        on: bool | None = None,
        load: Callable[[], PeopleConfig] = PeopleConfig.load,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        report: Callable[[str], None] = say,
        background: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``build`` makes the parts (once, when the switch first goes on); ``on`` None takes
        the file's ``flags.people``; ``background`` False builds and warms in the caller's
        thread (tests)."""
        self._build, self._load, self._clock, self._wall = build, load, clock, wall
        self.report, self.sleep = report, sleep
        self.background = background
        self.config = PeopleConfig()
        self.parts: Parts | None = None
        self._on = False
        self._lock = threading.RLock()
        self._galleries: dict[tuple[Modality, str], Gallery] = {}
        self.session = Session(self.config.session)
        self.conversation = False
        self._gaze: tuple[float, FaceFrame, FaceSighting] | None = None
        self._writer = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="people-store")
        self._unload: threading.Timer | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="people-face")
        if on is None:
            try:
                on = self._load().on
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("config/people.json: %s", exc)
                on = False
        if on:
            why = self.switch(True)
            if why is not None:
                logger.warning("person recognition stays off: %s", why)

    # ---- the switch ----------------------------------------------------------------------------
    @property
    def flag(self) -> str:
        """``people``."""
        return FLAG

    @property
    def on(self) -> bool:
        """Whether recognition runs."""
        return self._on

    def switch(self, on: bool) -> str | None:
        """On re-reads config/people.json and builds what is missing (in the background);
        off lets the identity net go and forgets the conversation's strangers."""
        if not on:
            if self._on:
                self._on = False
                with self._lock:
                    self.session = Session(self.config.session)
                    self._gaze = None
                self._unload_now("the switch is off")
            return None
        try:
            config = self._load()
        except (OSError, ValueError, TypeError) as exc:
            return f"config/people.json: {exc}"
        with self._lock:
            self.config = config
            self.session = Session(config.session)
            self._galleries.clear()
        self._on = True
        if self.background:
            threading.Thread(target=self._start, name="people-start", daemon=True).start()
        else:
            self._start()
        return None

    def _start(self) -> None:
        """Build the parts once, then warm the identity net and the finder."""
        try:
            if self.parts is None:
                self.parts = self._build(self.config)
            parts = self.parts
            known = len(parts.store.people())
            what = [f"{known} people known"]
            if parts.voice is None:
                what.append("no voice net")
            if parts.faces is not None:
                parts.faces.warm()
            else:
                what.append("no identity net")
            warm = getattr(parts.finder, "warm", None)
            if callable(warm):
                warm()
            self.report("on: " + ", ".join(what))
        except Exception as exc:  # a missing model or store: recognition names nobody
            logger.exception("person recognition could not start")
            self.report(f"could not start ({exc}): nobody is named")

    # ---- the stage -----------------------------------------------------------------------------
    def measure(self, pcm: bytes, end_s: float) -> Callable[[str], Speaker]:
        """One utterance's voice and face (on the cascade's worker); the naming step back."""
        parts = self.parts
        if not self._on or parts is None:
            return lambda text: Speaker("")
        try:
            measured = self._observe(parts, pcm, end_s)
        except Exception:  # a broken camera or net: nobody is named, the voice goes on
            logger.exception("person recognition could not measure an utterance")
            measured = Measured(Observation(end_s))
        return lambda text: self.name(measured, text)

    def _observe(self, parts: Parts, pcm: bytes, end_s: float) -> Measured:
        knobs = self.config
        faces = self._pool.submit(self._face, parts) if parts.faces is not None else None
        voice: Vector | None = None
        samples = samples_of(pcm)
        start, end = speech_span(samples, knobs.voice.span_db)
        speech_s = (end - start) / RATE
        if parts.voice is not None and speech_s >= knobs.voice.min_s:
            cut = samples[start : min(end, start + int(knobs.voice.max_s * RATE))]
            voice = parts.voice.embed(cut)
        looked: str = "no"
        speaker: tuple[Shot, Vector] | None = None
        others: tuple[Vector, ...] = ()
        if faces is not None:
            try:
                looked, speaker, others = faces.result()
            except Exception:  # the camera or the net failed: the voice still counts
                logger.exception("person recognition could not look at the speaker")
        face: Vector | None = None
        face_px, face_gate, face_strong = 0.0, "", False
        if speaker is not None:
            face_px = speaker[0].width
            face_gate = "ok"
            face = speaker[1]
            face_strong = strong(speaker[0], knobs.face)
        obs = Observation(end_s, voice, speech_s, face, face_px, others)
        return Measured(obs, face_strong, speech_s >= knobs.voice.strong_s, looked, face_gate)

    def _face(self, parts: Parts) -> tuple[str, tuple[Shot, Vector] | None, tuple[Vector, ...]]:
        """The speaker's face (gated, embedded) and the other faces in view that pass."""
        knobs = self.config.face
        net = parts.faces
        assert net is not None
        picked = self._gaze
        now = self._clock()
        frame: FaceFrame | None
        if picked is not None and now - picked[0] <= knobs.fresh_s:
            looked, frame, speaker = "gaze", picked[1], picked[2]
        elif parts.finder is not None:
            looked, frame, speaker = "look", parts.finder.look(), None
        else:
            return "no", None, ()
        if frame is None:
            return looked, None, ()
        found = shots(frame, speaker, knobs.dominance)
        chosen: tuple[Shot, Vector] | None = None
        others: list[Vector] = []
        for shot in sorted(found, key=lambda s: (not s.speaker, -s.width)):
            if gate(shot, knobs) != "ok":
                continue
            if shot.speaker:
                chosen = (shot, net.embed(shot))
            elif len(others) < MAX_OTHERS:
                others.append(net.embed(shot))
        return looked, chosen, tuple(others)

    def name(self, measured: Measured, text: str) -> Speaker:
        """Who said it, now that the words are known: the verdict, the tracks, the learning."""
        if not self._on or self.parts is None:
            return Speaker("")
        parts = self.parts
        obs = measured.obs
        with self._lock:
            face = self._reading(parts, "face", obs.face, measured.face_strong)
            voice = self._reading(parts, "voice", obs.voice, measured.voice_strong)
            face, voice = comparable(face, voice, lambda r, pid: self._knows(parts, r, pid))
            verdict = fuse(face, voice)
            track = self.session.attribute(obs, verdict)
            person = parts.store.get(verdict.person) if verdict.person is not None else None
        if person is not None:
            self._writer.submit(self._learn, parts, person, verdict, obs)
        speaker_name = person.name if person is not None else None
        log = {
            "who": speaker_name,
            "confidence": verdict.confidence,
            "how": verdict.how,
            "face_cos": None if face is None else round(face.cos, 3),
            "voice_cos": None if voice is None else round(voice.cos, 3),
            "speech_s": round(obs.speech_s, 2),
            "face_px": round(obs.face_px),
            "looked": measured.looked,
            "track": None if track is None else track.id,
        }
        key = verdict.person if verdict.confidence in ("sure", "tentative") else None
        return Speaker(head(verdict, speaker_name), key, log)

    def _reading(
        self, parts: Parts, modality: Modality, embedding: Vector | None, is_strong: bool
    ) -> Reading | None:
        if embedding is None:
            return None
        model = self._model(parts, modality)
        if model is None:
            return None
        t = self.config.face.thresholds if modality == "face" else self.config.voice.thresholds
        return judge(
            self._gallery(parts, modality, model), embedding, t, modality, strong=is_strong
        )

    def _knows(self, parts: Parts, reading: Reading, person: str) -> bool:
        """Whether ``reading``'s modality has samples of ``person``."""
        model = self._model(parts, reading.modality)
        return (
            model is not None and person in self._gallery(parts, reading.modality, model).centroids
        )

    @staticmethod
    def _model(parts: Parts, modality: Modality) -> str | None:
        if modality == "voice":
            return parts.voice.name if parts.voice is not None else None
        return parts.faces.name if parts.faces is not None else None

    def _gallery(self, parts: Parts, modality: Modality, model: str) -> Gallery:
        key = (modality, model)
        gallery = self._galleries.get(key)
        if gallery is None:
            gallery = galleries(parts.store.samples(modality, model))
            self._galleries[key] = gallery
        return gallery

    def _learn(self, parts: Parts, person: Person, verdict: Verdict, obs: Observation) -> None:
        """A recognised person's new samples (sure verdicts only) and when they were seen."""
        try:
            now = self._wall()
            parts.store.seen(person.id, now)
            if verdict.confidence != "sure":
                return
            counts = parts.store.counts(person.id)
            wrote = False
            entries: tuple[tuple[Modality, Reading | None, Vector | None, int], ...] = (
                ("face", verdict.face, obs.face, self.config.face.keep),
                ("voice", verdict.voice, obs.voice, self.config.voice.keep),
            )
            for modality, reading, vector, keep in entries:
                model = self._model(parts, modality)
                if vector is None or model is None:
                    continue
                if learnable(reading, person.id, counts.get(modality, 0) > 0):
                    parts.store.add_samples(
                        person.id, modality, model, [vector], now=now, keep=keep
                    )
                    wrote = True
            if wrote:
                with self._lock:
                    self._galleries.clear()
        except Exception:
            logger.exception("person recognition could not learn from an utterance")

    # ---- the tools' door -------------------------------------------------------------------------
    def remember(self, name: str, family: bool = False) -> Outcome:
        """Bind ``name`` to the stranger who spoke last (or the one new face in view)."""
        clean = " ".join(name.split())
        if not self._on or self.parts is None:
            return Outcome(False, "person recognition is off (the people switch)")
        if not clean or len(clean) > MAX_NAME:
            return Outcome(False, f"a name is 1 to {MAX_NAME} characters, not {name!r}")
        parts = self.parts
        with self._lock:
            known = parts.store.named(clean)
            latest = self.session.latest
            if known:
                if latest is not None and latest.person in {p.id for p in known}:
                    return Outcome(True, payload={"remembered": clean, "already": True})
                met = time.strftime("%H:%M", time.localtime(known[0].created_at))
                return Outcome(
                    False,
                    f"someone else is already called {clean} (met at {met}): ask how to tell"
                    " them apart (a surname or a nickname) and call remember_person with that"
                    " name",
                    {"collision": True},
                )
            track = self.session.to_bind()
            if track is not None:
                voice, face, how = list(track.voice), list(track.face), "the person who spoke"
            else:
                strangers = self.session.strangers_in_view(lambda f: self._known_face(parts, f))
                if len(strangers) > 1:
                    return Outcome(
                        False,
                        "several new faces are in view: ask that person to come closer, or to"
                        " say a few words to the robot, then call remember_person again",
                    )
                voice, face, how = [], strangers, "the new face in view"
        extra = self._looks(parts, face)
        face = [*face, *extra]
        if not voice and not face:
            return Outcome(
                False,
                "nobody new spoke or is in view: ask the person to say a few words to the robot"
                " while facing it, then call remember_person again",
            )
        kind: Kind = "family" if family or self.config.session.kind == "family" else "guest"
        now = self._wall()
        person = parts.store.add_person(clean, kind=kind, now=now)
        counts = {"face": 0, "voice": 0}
        samples: tuple[tuple[Modality, list[Vector], int], ...] = (
            ("face", face, self.config.face.keep),
            ("voice", voice, self.config.voice.keep),
        )
        for modality, vectors, keep in samples:
            model = self._model(parts, modality)
            if vectors and model is not None:
                counts[modality] = parts.store.add_samples(
                    person.id, modality, model, vectors, now=now, keep=keep
                )
        with self._lock:
            if track is not None:
                self.session.bound(track, person.id)
            else:
                bound = Track(len(self.session.tracks) + 1, person.id, face=list(face))
                self.session.tracks.append(bound)
            self._galleries.clear()
        self.report(f"remembered {clean} ({kind}): {counts['face']} face, {counts['voice']} voice")
        payload: dict[str, Any] = {
            "remembered": clean,
            "from": how,
            "face_samples": counts["face"],
            "voice_samples": counts["voice"],
        }
        if not counts["voice"]:
            payload["note"] = "the voice is learned as they speak"
        return Outcome(True, payload=payload)

    def forget(self, name: str) -> Outcome:
        """Delete everything of the person called ``name``."""
        if self.parts is None:
            return Outcome(False, "person recognition has not run in this session")
        parts = self.parts
        with self._lock:
            found = parts.store.named(name)
            if not found:
                return Outcome(False, f"nobody called {' '.join(name.split())} is known")
            if len(found) > 1:
                latest = self.session.latest
                found = [p for p in found if latest is not None and p.id == latest.person]
                if len(found) != 1:
                    return Outcome(False, f"several people are called {name}: ask which one")
            person = found[0]
            counts = parts.store.counts(person.id)
            parts.store.forget(person.id)
            self.session.forget(person.id)
            self._galleries.clear()
        self.report(f"forgot {person.name}")
        return Outcome(True, payload={"forgotten": person.name, "samples_deleted": counts})

    def known(self) -> list[Person]:
        """Everyone the store holds."""
        return [] if self.parts is None else self.parts.store.people()

    def _known_face(self, parts: Parts, face: Vector) -> bool:
        model = self._model(parts, "face")
        if model is None:
            return False
        t = self.config.face.thresholds
        return judge(self._gallery(parts, "face", model), face, t, "face").label == "match"

    def _looks(self, parts: Parts, have: list[Vector]) -> list[Vector]:
        """More face samples of the person being named: ``bind_looks`` fresh pictures, each
        dominant face taken when it resembles the faces in hand (or, with none in hand, when it
        is the only face in view and nobody known)."""
        knobs, net, finder = self.config, parts.faces, parts.finder
        if net is None or finder is None:
            return []
        out: list[Vector] = []
        for i in range(knobs.session.bind_looks):
            if i:
                self.sleep(knobs.session.bind_look_every_s)
            try:
                frame = finder.look()
            except Exception:
                logger.exception("a look for more face samples failed")
                return out
            if frame is None:
                continue
            found = shots(frame, None, knobs.face.dominance)
            speaker = next((s for s in found if s.speaker), None)
            if speaker is None or gate(speaker, knobs.face) != "ok":
                continue
            e = net.embed(speaker)
            ref = have or out
            if ref:
                if float(centroid(np.stack(ref)) @ e) >= knobs.session.track_face:
                    out.append(e)
            elif len(found) == 1 and not self._known_face(parts, e):
                out.append(e)
        return out

    # ---- the gaze's faces ------------------------------------------------------------------------
    def seen_face(self, frame: FaceFrame, face: FaceSighting) -> None:
        """The speaker gaze centred ``face`` in ``frame`` (from the gaze's thread)."""
        if self._on:
            self._gaze = (self._clock(), frame, face)

    # ---- the observer ------------------------------------------------------------------------
    def opened(self, wake: Heard) -> None:
        """A conversation opened: a fresh session; the identity net stays."""
        self.conversation = True
        self._cancel_unload()
        with self._lock:
            self.session = Session(self.config.session)
            self._gaze = None

    def speech_started(self, t: float) -> None:
        """Nothing."""

    def speech_ended(self, heard: Heard) -> None:
        """Nothing: the stage measured it."""

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """Nothing."""

    def voice_started(self, turn: int) -> None:
        """Nothing."""

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """Nothing."""

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """Nothing."""

    def part_spoken(self, turn: int, index: int) -> None:
        """Nothing."""

    def reply_cut(self, turn: int) -> None:
        """Nothing."""

    def reply_done(self, turn: int) -> None:
        """Nothing."""

    def moment(self, name: str) -> None:
        """Nothing."""

    def drive(self, active: bool) -> None:
        """Nothing."""

    def closed(self, reason: str) -> None:
        """The conversation's strangers are forgotten; the identity net is let go later."""
        self.conversation = False
        with self._lock:
            self.session = Session(self.config.session)
            self._gaze = None
        delay = self.config.face.unload_after_s
        if self._on and delay > 0:
            self._cancel_unload()
            timer = threading.Timer(delay, self._unload_now, args=("no conversation",))
            timer.daemon = True
            self._unload = timer
            timer.start()

    def _cancel_unload(self) -> None:
        if self._unload is not None:
            self._unload.cancel()
            self._unload = None

    def _unload_now(self, why: str) -> None:
        parts = self.parts
        if self.conversation and self._on:
            return
        if parts is not None and parts.faces is not None and parts.faces.unload():
            self.report(f"identity net let go ({why})")

    def close(self) -> None:
        """Stop the timer and the workers (the cascade's end)."""
        self._cancel_unload()
        self._writer.shutdown(wait=True)
        self._pool.shutdown(wait=False)
        close = getattr(self.parts.store if self.parts is not None else None, "close", None)
        if callable(close):
            close()


def robot_people(
    finder: FaceFinder | None,
    *,
    on: bool | None = None,
    service: Perception | None = None,
    source: VisionSource | None = None,
) -> People:
    """Person recognition on this Mac: config/people.json's store and nets, faces from
    ``finder`` (the one the speaker gaze shares); built the first time its switch is on. With
    the perception ``service`` (and the live ``source`` that picks the camera path), a face that
    came by frame number is embedded there and the net in this process loads only on the
    snapshot path."""

    def build(config: PeopleConfig) -> Parts:
        from pepin.people.store import SqliteStore

        store = SqliteStore(config.path(config.store))
        voice: VoiceEmbedder | None = None
        try:
            from pepin.people.voice import WeSpeaker

            voice = WeSpeaker(config.path(config.voice.model), threads=config.voice.threads)
        except (OSError, ImportError) as exc:
            say(f"no voice net ({exc}): faces only")
        local: IdentityNet | None = None
        model = config.path(config.face.model)
        if model.exists():

            def load() -> Any:
                from pepin.face_sensor.backends import ArcFace

                cache = config.path("models/face/coreml_cache")
                return ArcFace(model, device=config.face.device, cache=cache)

            local = IdentityNet(load)
        faces: FaceNet | None = local
        if service is not None and source is not None:
            remote = ServiceIdentity(service, model.stem)
            faces = RoutedIdentity(local, remote, source, model.stem)
        elif local is None:
            say(f"no identity net at {model}: voices only")
        return Parts(store, voice, faces, finder)

    return People(build, on=on)
