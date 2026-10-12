"""ros/tools/doa_calibrate.py: utterances found in the array's frames and directions, and the
fit written into config/expression.json without disturbing the rest of the file."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from pepin.audio_link import FRAME_SAMPLES, AudioFrame, DoaReading
from pepin.expression.doa import fit_doa_map

REPO = Path(__file__).resolve().parents[2]


def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "doa_calibrate", REPO / "ros/tools/doa_calibrate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


class Link:
    """The audio link: 20 ms frames, loud where ``voiced`` says, and directions every 0.1 s."""

    def __init__(self, voiced: list[tuple[float, float, int]], seconds: float) -> None:
        self.voiced, self.seconds = voiced, seconds

    def _voice(self, t: float) -> int | None:
        return next((deg for start, end, deg in self.voiced if start <= t < end), None)

    def frames(self) -> Iterator[AudioFrame]:
        loud = (8000).to_bytes(2, "little", signed=True) * FRAME_SAMPLES
        quiet = bytes(2 * FRAME_SAMPLES)
        for k in range(int(self.seconds / 0.02)):
            t = k * 0.02
            yield AudioFrame(k, t, loud if self._voice(t) is not None else quiet)

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        out = []
        for k in range(int(t0 * 10), int(t1 * 10) + 1):
            t = k / 10
            if t0 <= t <= t1:
                deg = self._voice(t)
                out.append(DoaReading(t, deg if deg is not None else 300, deg is not None))
        return out


def test_utterances_are_the_loud_spells_with_their_speech_directions() -> None:
    link = Link([(0.5, 1.5, 180), (2.0, 2.3, 90), (3.0, 4.2, 270)], seconds=5.0)
    heard = list(tool().utterances(link, threshold_db=-40.0, hold_s=0.5, gap_s=0.4))
    assert [round(u.deg) for u in heard] == [180, 270], "0.3 s is too short to count"
    assert heard[0].speech >= 9 and heard[0].loud_db > -15


def test_the_fit_is_written_with_its_date_and_the_rest_of_the_file_stays(tmp_path: Path) -> None:
    path = tmp_path / "expression.json"
    shutil.copy(REPO / "config/expression.json", path)
    before = json.loads(path.read_text())
    readings = [182.0, 271.0, 3.0, 92.0]
    tool().write(fit_doa_map(*readings), readings, path)
    after = json.loads(path.read_text())
    doa = after["speaker"]["doa"]
    assert (doa["front_deg"], doa["sign"]) == (182.0, 1) and doa["measured"].startswith("MEASURED")
    assert "front 182, left 271, behind 3, right 92" in doa["measured"]
    for key in ("flags", "timing", "parts"):
        assert after[key] == before[key]
    assert tool().spread_deg([350.0, 10.0], 0.0) == 10.0
