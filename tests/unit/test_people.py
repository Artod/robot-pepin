"""Person recognition: the store (in memory and SQLite), the matcher's margins, the fusion
table, the tag text, the voice's loud span and filterbanks, the face gate, and the whole stage
on fakes: a stranger named and recognised, the name bound to the right speaker, collisions,
forgetting, learning only from sure verdicts, and the switch off doing nothing."""

import stat
from pathlib import Path

import numpy as np
import pytest

from pepin.expression.speaker import FaceFrame
from pepin.face_sensor.identity import Gallery
from pepin.people.config import PeopleConfig, SessionKnobs, VoiceKnobs
from pepin.people.face import IdentityNet, Shot, gate, shots
from pepin.people.fakes import (
    FakeFinder,
    FakeIdentity,
    FakeVoiceNet,
    detection,
    frame,
    near,
    random_unit,
    speech,
    textured,
)
from pepin.people.fusion import comparable, fuse
from pepin.people.matcher import Reading, Thresholds, galleries, judge
from pepin.people.session import Observation, Session, learnable
from pepin.people.store import MemoryStore, PersonStore, SqliteStore
from pepin.people.tag import head
from pepin.people.voice import RATE, fbank, mel_banks, samples_of, speech_span
from pepin.people.worker import Measured, Parts, People
from pepin.tools import TOOLS
from pepin.tools.fakes import FakePeople, fake_robot
from pepin.voice_cascade.config import ProsodyConfig
from pepin.voice_cascade.stages import Prosody, Speaker
from pepin.voice_cascade.tagger import ProsodyTagger

T = Thresholds(accept=0.45, margin=0.08, stranger=0.25, sure=0.55)


# -- the store --------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> PersonStore:
    if request.param == "memory":
        return MemoryStore()
    return SqliteStore(tmp_path / "people" / "people.db")


def test_a_store_keeps_people_and_the_newest_samples_by_modality_and_model(
    store: PersonStore,
) -> None:
    sasha = store.add_person(" Саша ", kind="guest", now=10.0)
    masha = store.add_person("Маша", kind="family", now=20.0)
    assert sasha.name == "Саша" and [p.name for p in store.people()] == ["Саша", "Маша"]
    vs = [random_unit(i, 8) for i in range(5)]
    assert store.add_samples(sasha.id, "face", "arc", vs, now=11.0, keep=3) == 3
    assert store.add_samples(sasha.id, "voice", "wes", vs[:2], now=11.0, keep=3) == 2
    assert store.add_samples(masha.id, "face", "other-net", vs[:1], now=21.0, keep=3) == 1
    faces = store.samples("face", "arc")
    assert set(faces) == {sasha.id}  # another net's samples never compare
    np.testing.assert_allclose(faces[sasha.id], np.stack(vs[2:]), atol=1e-6)  # the newest three
    assert store.counts(sasha.id) == {"face": 3, "voice": 2}
    assert [p.id for p in store.named("саша")] == [sasha.id]  # case folded, blanks ignored
    store.seen(sasha.id, 99.0)
    seen = store.get(sasha.id)
    assert seen is not None and seen.last_seen_at == 99.0 and seen.kind == "guest"
    with pytest.raises(KeyError):
        store.add_samples("nobody", "face", "arc", vs, now=1.0, keep=3)
    assert store.forget(sasha.id) and not store.forget(sasha.id)
    assert store.samples("voice", "wes") == {} and store.get(sasha.id) is None
    assert store.counts(sasha.id) == {"face": 0, "voice": 0}


def test_the_sqlite_store_round_trips_privately_and_overwrites_what_it_forgets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "people" / "people.db"
    store = SqliteStore(path)
    sasha = store.add_person("Саша", kind="guest", now=1.0)
    v = random_unit(7, 16)
    store.add_samples(sasha.id, "voice", "wes", [v], now=1.0, keep=5)
    store.close()
    again = SqliteStore(path)
    (person,) = again.people()
    assert person == sasha
    np.testing.assert_array_equal(again.samples("voice", "wes")[sasha.id][0], v)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert again._db.execute("PRAGMA secure_delete").fetchone()[0] == 1
    again.forget(sasha.id)
    again.close()
    assert v.astype("<f4").tobytes() not in path.read_bytes()  # zeroed, not just unlinked


# -- the matcher and the fusion -------------------------------------------------------------------


def test_a_match_needs_the_threshold_and_the_margin_over_the_runner_up() -> None:
    a, b = random_unit(1), random_unit(2)
    gallery = Gallery({"a": a, "b": b})
    reading = judge(gallery, near(a, 0.7, 3), T, "face")
    assert (reading.label, reading.person, reading.sure) == ("match", "a", True)
    assert judge(gallery, near(a, 0.5, 4), T, "face").sure is False
    twins = Gallery({"a": a, "a2": near(a, 0.97, 5)})  # two people alike: abstain, not guess
    assert judge(twins, near(a, 0.8, 6), T, "face").label == "unsure"
    assert judge(gallery, near(a, 0.35, 7), T, "face").label == "unsure"
    assert judge(gallery, random_unit(8), T, "face").label == "stranger"
    assert judge(Gallery(), a, T, "voice").label == "stranger"  # nobody to compare with


def test_a_gallery_is_the_centroid_of_each_person_s_samples() -> None:
    a = random_unit(1)
    rows = np.stack([near(a, 0.8, s) for s in range(10, 20)])
    centre = galleries({"a": rows}).centroids["a"]
    assert float(centre @ a) > 0.95


def reading(label: str, person: str | None = None, *, sure: bool = False, strong: bool = False,
            modality: str = "face") -> Reading:  # fmt: skip
    return Reading(modality, person, 0.6, 0.1, label, sure, strong)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("face", "voice", "person", "confidence"),
    [
        (reading("match", "a"), reading("match", "a"), "a", "sure"),
        (reading("match", "a", strong=True), reading("match", "b"), "a", "tentative"),
        (reading("match", "a"), reading("match", "b"), "b", "tentative"),
        (reading("match", "a", sure=True), None, "a", "sure"),
        (reading("match", "a"), reading("unsure", "b"), "a", "tentative"),
        (None, reading("match", "b", sure=True), "b", "sure"),
        (None, reading("match", "b"), "b", "tentative"),
        (reading("match", "a", sure=True), reading("stranger"), "a", "tentative"),
        (reading("stranger"), reading("match", "b", sure=True), "b", "tentative"),
        (reading("stranger"), reading("stranger"), None, "new"),
        (reading("stranger"), None, None, "new"),
        (reading("unsure", "a"), reading("stranger"), None, "new"),
        (reading("unsure", "a"), None, None, "unknown"),
        (None, None, None, "unknown"),
    ],
)
def test_the_fusion_table(
    face: Reading | None, voice: Reading | None, person: str | None, confidence: str
) -> None:
    verdict = fuse(face, voice)
    assert (verdict.person, verdict.confidence) == (person, confidence)


def test_a_modality_that_never_met_the_person_is_no_evidence_against_them() -> None:
    face, voice = reading("match", "sasha", sure=True), reading("stranger", modality="voice")
    fair_face, fair_voice = comparable(face, voice, lambda r, pid: False)
    assert fair_voice is not None and fair_voice.label == "unsure" and fair_face == face
    assert fuse(fair_face, fair_voice).confidence == "sure"
    assert comparable(face, voice, lambda r, pid: True) == (face, voice)  # met: a real conflict


def test_the_tag_names_the_speaker_and_carries_the_prosody_words() -> None:
    sure = fuse(reading("match", "a"), reading("match", "a"))
    assert head(sure, "Саша") == "говорит: Саша"
    assert head(fuse(reading("match", "a"), None), "Саша") == "говорит: похоже, Саша"
    assert head(fuse(reading("stranger"), None), None) == "говорит: кто-то новый"
    assert head(fuse(None, None), None) == ""
    assert Speaker("говорит: Саша").line("тише обычного") == "[говорит: Саша; тише обычного]"
    assert Speaker("говорит: Саша").line("") == "[говорит: Саша]"
    assert Speaker("").line("") == ""


def test_prosody_baselines_are_kept_per_person() -> None:
    tagger = ProsodyTagger(ProsodyConfig(min_baseline=2))
    loud, quiet = Prosody(2.0, 70.0, 7.0, 6.0), Prosody(2.0, 50.0, 7.0, 6.0)
    for _ in range(2):
        tagger.describe(loud, "раз два три", "artem")
    assert tagger.describe(quiet, "раз два три", "artem") == "тише обычного"
    assert tagger.describe(quiet, "раз два три", "sasha") == ""  # Sasha has no baseline yet
    assert tagger.line(loud, "раз два три") == ""  # the shared one is untouched


# -- the voice ------------------------------------------------------------------------------------


def test_the_loud_span_drops_the_lead_in_and_the_tail() -> None:
    pcm = speech(3, 1.5, lead_s=0.5)
    start, end = speech_span(samples_of(pcm))
    assert start == pytest.approx(0.5 * RATE, abs=320) and (end - start) / RATE == pytest.approx(
        1.5, abs=0.03
    )
    assert speech_span(np.zeros(RATE, dtype=np.int16)) == (0, 0)


def test_the_filterbanks_are_kaldi_shaped_and_mean_free() -> None:
    feats = fbank(samples_of(speech(5, 1.0, lead_s=0.0)))
    assert feats.shape == (98, 80) and feats.dtype == np.float32
    np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-3)
    banks = mel_banks()
    assert banks.shape == (80, 257) and (banks.sum(axis=1) > 0).all() and not banks[:, -1].any()
    assert fbank(np.zeros(100, dtype=np.int16)).shape == (0, 80)


# -- the face -------------------------------------------------------------------------------------

KNOBS = PeopleConfig().face


def test_a_face_counts_only_when_big_sure_frontal_and_sharp() -> None:
    image = textured()
    assert gate(Shot(image, detection(100, 80), True), KNOBS) == "ok"
    assert gate(Shot(image, detection(100, 30), True), KNOBS) == "small"
    assert gate(Shot(image, detection(100, 80, score=0.6), True), KNOBS) == "doubtful"
    assert gate(Shot(image, detection(100, 80, yaw=60), True), KNOBS) == "turned"
    flat = np.full((600, 800, 3), 128, dtype=np.uint8)
    assert gate(Shot(flat, detection(100, 80), True), KNOBS) == "blurry"
    dark = (textured() // 8).astype(np.uint8)
    assert gate(Shot(dark, detection(100, 80), True), KNOBS) == "dark"


def test_the_speaker_s_face_is_the_gaze_s_or_a_clearly_dominant_one() -> None:
    near_face, far_face = detection(100, 100), detection(400, 50)
    marked = shots(frame(near_face, far_face), None, 1.4)
    assert [s.speaker for s in marked] == [True, False]
    crowd = shots(frame(detection(100, 80), detection(400, 70)), None, 1.4)
    assert not any(s.speaker for s in crowd)  # alike faces: nobody is the speaker
    f = frame(near_face, far_face)
    gazed = shots(f, f.faces[1], 1.4)  # the gaze centred the far face: it is the speaker's
    assert [s.speaker for s in gazed] == [False, True]
    assert shots(FaceFrame(0.0, f.faces), None, 1.4) == []  # no picture kept, nothing to embed


def test_the_identity_net_loads_once_on_demand_and_lets_go() -> None:
    made: list[int] = []

    def load() -> FakeIdentity:
        made.append(1)
        return FakeIdentity({100.0: random_unit(1)})

    net = IdentityNet(load)
    assert not net.loaded
    shot = Shot(textured(), detection(100, 80), True)
    net.embed(shot)
    net.embed(shot)
    assert net.loaded and len(made) == 1 and net.name == "fake-face"
    assert net.unload() and not net.loaded and not net.unload()
    assert net.name == "fake-face" and not net.loaded  # the name is kept, no load for it
    net.embed(shot)
    assert len(made) == 2


# -- the session --------------------------------------------------------------------------------


def test_a_stranger_s_utterances_join_one_track_and_another_starts_a_new_one() -> None:
    session = Session(SessionKnobs())
    v1, v2 = random_unit(1), random_unit(2)
    new = fuse(reading("stranger"), None)
    t1 = session.attribute(Observation(1.0, voice=v1), new)
    t1b = session.attribute(Observation(2.0, voice=near(v1, 0.6, 9)), new)
    t2 = session.attribute(Observation(3.0, voice=v2), new)
    assert t1 is t1b and t2 is not t1 and len(t1.voice) == 2
    assert session.to_bind() is t2
    # a short "Саша" with no face: nothing to tell by, so the latest speaker goes on
    assert session.attribute(Observation(4.0), fuse(None, None)) is None
    assert session.to_bind() is t2
    session.attribute(Observation(5.0, voice=v1), fuse(None, reading("match", "artem")))
    assert session.to_bind() is None  # a known person spoke last


def test_a_sure_verdict_learns_the_agreeing_modality_and_the_missing_one() -> None:
    assert learnable(reading("match", "a"), "a", has_samples=True)
    assert not learnable(reading("match", "b"), "a", has_samples=True)
    assert not learnable(reading("stranger", "a"), "a", has_samples=True)
    assert learnable(reading("stranger"), "a", has_samples=False)  # a voice learned from a face
    assert not learnable(None, "a", has_samples=False)


# -- the whole stage on fakes ---------------------------------------------------------------------


class Rig:
    """People on fakes: voices by loudness, faces by box, pictures from a script."""

    def __init__(self, *, on: bool = True, store: PersonStore | None = None) -> None:
        self.voices = {k: random_unit(100 + k) for k in range(1, 6)}
        self.faces = {float(x): random_unit(200 + x) for x in (100, 400, 600)}
        self.voice_net = FakeVoiceNet(dict(self.voices))
        self.face_net = FakeIdentity(dict(self.faces))
        self.finder = FakeFinder()
        self.store = store or MemoryStore()
        self.lines: list[str] = []
        self.wall = [1000.0]
        parts = Parts(self.store, self.voice_net, IdentityNet(lambda: self.face_net), self.finder)
        self.people = People(
            lambda config: parts,
            on=on,
            load=lambda: PeopleConfig(on=on),
            wall=lambda: self.wall[0],
            report=self.lines.append,
            background=False,
            sleep=lambda s: None,
        )

    def hear(self, voice: int | None, *faces: float, seconds: float = 2.0) -> Speaker:
        """One utterance: speaker ``voice`` talking (None: a short "да"), these faces in view."""
        self.finder.show([frame(*(detection(x, 100 if i == 0 else 50) for i, x in
                                  enumerate(faces)))] if faces else [])  # fmt: skip
        pcm = speech(voice, seconds) if voice is not None else speech(1, 0.4)
        speaker = self.people.measure(pcm, 0.0)("слова")
        self.people._writer.submit(lambda: None).result()  # the learning written
        return speaker


def test_the_switch_off_measures_nothing_and_names_nobody() -> None:
    rig = Rig(on=False)
    assert rig.people.measure(speech(1, 2.0), 0.0)("привет") == Speaker("")
    assert rig.voice_net.calls == 0 and rig.finder.looks == 0 and rig.face_net.calls == 0
    assert not rig.people.remember("Саша").ok
    assert rig.people.switch(True) is None and rig.people.on
    assert rig.people.switch(False) is None and not rig.people.on


def test_a_stranger_is_named_and_then_recognised_by_face_and_voice() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    first = rig.hear(1, 100.0)
    assert first.head == "говорит: кто-то новый" and first.key is None
    assert rig.hear(1, 100.0).head == "говорит: кто-то новый"  # the same stranger's track
    outcome = rig.people.remember("Саша")
    assert outcome.ok and outcome.payload["voice_samples"] == 2
    assert outcome.payload["face_samples"] >= 2  # the track's two and the extra looks
    rig.people.closed("idle")
    rig.people.opened(None)  # type: ignore[arg-type]
    again = rig.hear(1, 100.0)
    (sasha,) = rig.store.named("Саша")
    assert again.head == "говорит: Саша" and again.key == sasha.id
    assert again.log is not None and again.log["confidence"] == "sure"
    assert rig.hear(None, 100.0).head == "говорит: Саша"  # a short "да": the face alone, sure
    assert rig.hear(None).head == ""  # nothing to go on: no tag at all


def test_the_name_binds_to_the_stranger_who_spoke_last() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    rig.hear(2, 400.0)  # one guest, then another
    rig.hear(3, 600.0)
    assert rig.people.remember("Миша").ok
    rig.hear(2, 400.0)
    rig.people.closed("idle")
    rig.people.opened(None)  # type: ignore[arg-type]
    assert rig.hear(3, 600.0).head == "говорит: Миша"
    assert rig.hear(2, 400.0).head == "говорит: кто-то новый"


def test_someone_introduced_is_bound_by_the_one_new_face_in_view() -> None:
    rig = Rig()
    artem = rig.store.add_person("Артём", kind="family", now=1.0)
    rig.store.add_samples(artem.id, "voice", "fake-voice", [rig.voices[1]], now=1.0, keep=20)
    rig.store.add_samples(artem.id, "face", "fake-face", [rig.faces[100.0]], now=1.0, keep=30)
    rig.people.opened(None)  # type: ignore[arg-type]
    said = rig.hear(1, 100.0, 400.0)  # Artem speaks, a guest's face beside his
    assert said.head == "говорит: Артём"
    outcome = rig.people.remember("Саша")
    assert outcome.ok and outcome.payload["from"] == "the new face in view"
    assert outcome.payload["voice_samples"] == 0 and "voice is learned" in outcome.payload["note"]
    rig.people.closed("idle")
    rig.people.opened(None)  # type: ignore[arg-type]
    first = rig.hear(4, 400.0)  # Sasha speaks for the first time: his face is sure alone
    assert first.head == "говорит: Саша"
    (sasha,) = rig.store.named("Саша")
    assert rig.store.counts(sasha.id)["voice"] == 1  # his voice learned from the sure face


def test_a_taken_name_is_refused_with_what_to_ask() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    rig.hear(2, 400.0)
    assert rig.people.remember("Миша").ok
    rig.hear(3, 600.0)
    taken = rig.people.remember("миша")
    assert not taken.ok and taken.payload == {"collision": True} and "nickname" in taken.why
    again = rig.people.remember("Миша Петров")
    assert again.ok
    rig.hear(3, 600.0)
    assert rig.people.remember("Миша Петров").payload.get("already") is True


def test_nobody_new_means_no_binding() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    assert not rig.people.remember("Саша").ok  # nobody spoke, nothing in view
    rig.hear(2, 400.0, 600.0)
    assert rig.hear(None).head == ""  # "Саша", short, looking away: names nobody
    assert rig.people.remember("Саша").ok  # the speaker's track, not the face beside
    rig.people.closed("idle")
    rig.people.opened(None)  # type: ignore[arg-type]
    assert rig.hear(4, 600.0, seconds=0.4).head == "говорит: кто-то новый"


def test_forgetting_deletes_the_person_and_they_are_new_again() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    rig.hear(2, 400.0)
    rig.people.remember("Саша")
    gone = rig.people.forget("саша")
    assert gone.ok and gone.payload["forgotten"] == "Саша"
    assert rig.store.people() == [] and rig.store.samples("face", "fake-face") == {}
    assert not rig.people.forget("Саша").ok
    assert rig.hear(2, 400.0).head == "говорит: кто-то новый"


def test_only_sure_verdicts_teach_and_the_conversation_s_end_forgets_strangers() -> None:
    rig = Rig()
    sasha = rig.store.add_person("Саша", kind="guest", now=1.0)
    rig.store.add_samples(sasha.id, "voice", "fake-voice", [rig.voices[2]], now=1.0, keep=20)
    rig.people.opened(None)  # type: ignore[arg-type]
    tentative = rig.hear(2)  # the voice alone, no face: tentative, learns nothing
    assert tentative.head == "говорит: Саша"  # the fake voice is exact: cos 1.0 >= sure
    rig.people.config = PeopleConfig(on=True, voice=VoiceKnobs(sure=1.01))
    assert rig.hear(2).head == "говорит: похоже, Саша"
    assert rig.store.counts(sasha.id) == {"face": 0, "voice": 2}  # only the sure one learned
    rig.hear(5, 600.0)
    assert rig.people.session.to_bind() is not None
    rig.people.closed("idle")
    assert rig.people.session.to_bind() is None


def test_the_gaze_s_fresh_face_is_used_without_a_new_picture() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    f = frame(detection(400, 90))
    rig.people.seen_face(f, f.faces[0])
    speaker = rig.people.measure(speech(2, 2.0), 0.0)("слова")
    assert rig.finder.looks == 0 and speaker.log is not None and speaker.log["looked"] == "gaze"


def test_a_measured_utterance_is_named_by_its_own_numbers() -> None:
    rig = Rig()
    obs = Observation(0.0, voice=rig.voices[1], speech_s=2.0)
    assert rig.people.name(Measured(obs), "да").head == "говорит: кто-то новый"


# -- the tools ------------------------------------------------------------------------------------


def test_the_tools_go_through_the_door_and_say_why_not() -> None:
    people = FakePeople()
    robot = fake_robot(people=people)
    assert TOOLS.call("remember_person", {"name": "Саша"}, robot)["remembered"] == "Саша"
    taken = TOOLS.call("remember_person", {"name": "Саша"}, robot)
    assert not taken["ok"] and taken["collision"]
    assert TOOLS.call("forget_person", {"name": "Саша"}, robot)["forgotten"] == "Саша"
    assert not TOOLS.call("forget_person", {"name": "Саша"}, robot)["ok"]
    absent = TOOLS.call("remember_person", {"name": "Саша"}, fake_robot(people=None))
    assert not absent["ok"] and "voice cascade" in absent["why"]
    schema = TOOLS["remember_person"].input_schema()
    assert schema["required"] == ["name"] and schema["properties"]["family"]["type"] == "boolean"
    assert not TOOLS["remember_person"].moves and not TOOLS["forget_person"].moves


def test_the_real_door_through_the_tool() -> None:
    rig = Rig()
    rig.people.opened(None)  # type: ignore[arg-type]
    rig.hear(2, 400.0)
    robot = fake_robot(people=rig.people)
    result = TOOLS.call("remember_person", {"name": "Саша", "family": True}, robot)
    assert result["ok"] and rig.store.named("Саша")[0].kind == "family"


def test_the_shipped_config_reads_and_is_off() -> None:
    config = PeopleConfig.load(Path(__file__).resolve().parents[2] / "config" / "people.json")
    assert (
        not config.on
        and config.voice.min_s == 1.0
        and config.face.thresholds == Thresholds(0.45, 0.08, 0.25, 0.55)
    )
    with pytest.raises(ValueError):
        PeopleConfig.from_dict({"voice": {"accept": 0.9, "sure": 0.5}})
