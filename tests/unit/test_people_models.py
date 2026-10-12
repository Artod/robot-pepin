"""Person recognition's real nets, where their weights are installed (models/ is gitignored):
the voice net embeds an utterance in tens of milliseconds and the identity net alone embeds an
aligned face; skipped on a checkout without the weights."""

import time

import numpy as np
import pytest

from pepin.people.config import PeopleConfig
from pepin.people.fakes import detection, speech, textured
from pepin.people.voice import samples_of

pytestmark = pytest.mark.slow
CONFIG = PeopleConfig()
VOICE = CONFIG.path(CONFIG.voice.model)
FACE = CONFIG.path(CONFIG.face.model)


@pytest.mark.skipif(not VOICE.exists(), reason=f"no voice net at {VOICE}")
def test_the_voice_net_embeds_four_seconds_within_its_budget() -> None:
    from pepin.people.voice import WeSpeaker

    net = WeSpeaker(VOICE, threads=CONFIG.voice.threads)
    x = samples_of(speech(3, 4.0, lead_s=0.0))
    net.embed(x)
    t0 = time.perf_counter()
    e = net.embed(x)
    assert (time.perf_counter() - t0) < 0.1
    assert e.shape == (256,) and float(np.linalg.norm(e)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not FACE.exists(), reason=f"no identity net at {FACE}")
def test_the_identity_net_alone_embeds_an_aligned_face() -> None:
    from pepin.face_sensor.backends import ArcFace

    net = ArcFace(FACE, device="cpu", cache=CONFIG.path("models/face/coreml_cache"))
    e = net.embed(textured(), detection(300, 120))
    assert e.shape == (512,) and float(np.linalg.norm(e)) == pytest.approx(1.0, abs=1e-5)
