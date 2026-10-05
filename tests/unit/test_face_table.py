"""config/face.json and the two copies generated from it (pepin.face)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pepin.face import (
    HEADER,
    SCRIPT,
    load_face_table,
    main,
    mouth_level,
    parse_face_table,
    render_header,
    rgb565,
    stale,
)

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"


def raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FACE.read_text())
    return data


def test_the_generated_tables_are_current() -> None:
    """firmware/head_esp32/src/face_table.h and sim/face_table.js are config/face.json's: a
    stale copy would draw another face on the head than in the simulator."""
    assert stale(load_face_table(FACE)) == [], "uv run python -m pepin.face --write"
    assert main(["--check"]) == 0


def test_artems_expressions_and_the_robots_events_are_all_there() -> None:
    table = load_face_table(FACE)
    names = "neutral smile grin clenched sad thinking worried surprised struggling flat sleepy"
    for name in [*names.split(), "happy", "focused", "listening"]:
        table.id_of(name)
    assert table.id_of("neutral") == 0
    assert set(table.events) >= {"goal_accepted", "recovery", "arrived", "goal_failed",
                                 "listening", "thinking", "speaking", "brain_lost"}  # fmt: skip
    assert table.event("recovery").hold_s == 3.0 and table.event("goal_accepted").hold_s is None
    with pytest.raises(KeyError, match="the face knows neutral, smile"):
        table.id_of("smug")


def test_ids_are_positions_so_the_wire_never_reorders() -> None:
    """An 'E' frame carries the index: the header's names are the table's, in its order."""
    table = load_face_table(FACE)
    header = HEADER.read_text()
    for i, name in enumerate(table.names):
        assert f"// {i} {name}" in header
    assert '"name": "neutral"' in SCRIPT.read_text()


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda d: d["expressions"][1]["params"].update(open=1.5), "smile: open 1.5 outside"),
        (lambda d: d["expressions"][1]["params"].update(grin=1), "unknown params grin"),
        (lambda d: d["expressions"].insert(0, {"name": "x", "params": {}}), "expression 0 is"),
        (lambda d: d["events"]["recovery"].update(name="nope"), "no expression 'nope'"),
        (lambda d: d["events"]["recovery"].update(hold_s=0), "hold_s is positive"),
        (lambda d: d["colors"].update(lip="cyan"), "#RRGGBB"),
        (lambda d: d["lipsync"].update(full_db=-60), "floor_db < full_db"),
        (lambda d: d["events"]["recovery"].update(min_gap_s=-1), "min_gap_s is not negative"),
        (lambda d: d["idle"].update(expressions=["smug"]), "idle: no expression smug"),
        (lambda d: d["idle"].update(every_s=[20, 8]), "0 < min <= max"),
        (lambda d: d["idle"]["gestures"][0].update(params={"grin": 1}), "unknown params grin"),
        (lambda d: d["idle"]["gestures"][0].update(weight=0), "weight and duration_s"),
        (lambda d: d["idle"].update(gestures=[]), "enabled with no gesture"),
    ],
)
def test_a_broken_table_is_refused_with_the_reason(edit: Any, why: str) -> None:
    data = copy.deepcopy(raw())
    edit(data)
    with pytest.raises(ValueError, match=why):
        parse_face_table(data)


def test_colors_and_levels() -> None:
    assert rgb565("#FFFFFF") == 0xFFFF and rgb565("#000000") == 0
    assert rgb565("#FF0000") == 0xF800 and rgb565("#00FF00") == 0x07E0
    table = load_face_table(FACE)
    assert mouth_level(-60.0, table) == 0.0
    assert mouth_level(-15.0, table) == 1.0
    assert mouth_level(-31.5, table) == pytest.approx(0.5)
    assert "constexpr int kNeutral = 0;" in render_header(table)


def test_the_idle_block_reaches_the_firmware_as_a_mask_and_gestures() -> None:
    table = load_face_table(FACE)
    idle = table.idle
    assert idle.enabled and idle.expressions == (table.id_of("neutral"), table.id_of("smile"))
    assert 0 < idle.every_s[0] <= idle.every_s[1] and idle.settle_s > 0
    names = [g.name for g in idle.gestures]
    assert names == ["grin", "smirk", "hmm", "tilt", "blink"]
    tilt = idle.gestures[names.index("tilt")]
    assert tilt.mirror and tilt.deltas[[p.name for p in table.params].index("asym")] > 0
    header = render_header(table)
    assert "constexpr uint32_t kIdleExpressionMask = 0x00000003u;" in header
    assert f"kIdleGestureCount = {len(names)};" in header
    off = raw()
    off["idle"]["enabled"] = False
    assert "constexpr bool kIdleOn = false;" in render_header(parse_face_table(off))
    del off["idle"]
    assert not parse_face_table(off).idle.enabled  # an older table: no idle, one empty gesture
    assert table.event("speaking").min_gap_s == 0.0 and "note" not in table.events
