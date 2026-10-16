"""config/face.json's ``eyes`` block and the two copies generated from it (pepin.face)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pepin.face import (
    EYE_EFFECTS,
    EYE_ELEMENTS,
    EYE_HEADER,
    EYE_OVERRIDABLE,
    EYE_SCRIPT,
    EYE_SETS,
    load_face_table,
    parse_face_table,
    render_eye_header,
    render_eye_script,
    stale,
)

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"
STYLES = load_face_table(FACE).style_names


def raw() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FACE.read_text())
    return data


def test_the_generated_eye_tables_are_current() -> None:
    """firmware/head_esp32/src/eye_table.h and sim/eye_table.js are config/face.json's."""
    table = load_face_table(FACE)
    assert table.eyes is not None
    assert stale(table) == [], "uv run python -m pepin.face --write"
    assert EYE_HEADER.read_text() == render_eye_header(table)
    assert EYE_SCRIPT.read_text() == render_eye_script(table)


def test_the_chain_has_both_rings_of_both_eyes_and_a_layout_id() -> None:
    eyes = load_face_table(FACE).eyes
    assert eyes is not None
    assert eyes.led_count == 48 and len(eyes.rings) == 4
    assert sorted((r.eye, r.ring) for r in eyes.rings) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [r.first for r in eyes.rings] == [0, 8, 24, 32]
    assert 0 <= eyes.layout_id <= 255
    angles = eyes.led_angles()
    assert len(angles) == 48 and all(0.0 <= a < 360.0 for a in angles)
    assert angles[:3] == (0.0, 45.0, 90.0) and angles[8:10] == (0.0, 22.5)
    # a ring wound the other way runs its angles down
    data = raw()
    data["eyes"]["layout"]["rings"][0]["clockwise"] = False
    data["eyes"]["layout"]["rings"][0]["first_deg"] = 90
    other = parse_face_table(data).eyes
    assert other is not None
    assert other.led_angles()[:3] == (90.0, 45.0, 0.0)
    assert other.layout_id != eyes.layout_id


def test_every_expression_has_a_look_per_style_and_the_map_of_emotions_holds() -> None:
    """The 'Y' frame's params exist; happy lifts the lower lids, sad puts the brows outward,
    sinister brings them to the nose in red, surprised raises them wide, thinking lifts one
    brow and looks aside, the sleeper's lids sink and its colour greys, the cat's eyes are
    closed arcs, the kiss winks, the terminator never blinks and has no catchlight."""
    table = load_face_table(FACE)
    eyes = table.eyes
    assert eyes is not None
    at = {p.name: i for i, p in enumerate(eyes.params)}
    assert set(EYE_OVERRIDABLE) <= set(at)
    assert len(eyes.looks) == len(table.expressions)
    assert all(len(per_style) == len(STYLES) for per_style in eyes.looks)

    def v(name: str, param: str, style: int = 0) -> float:
        return float(eyes.looks[table.id_of(name)][style].values[at[param]])

    def c(name: str, element: str, style: int = 0) -> str:
        return str(eyes.looks[table.id_of(name)][style].colors[EYE_ELEMENTS.index(element)])

    assert v("happy", "lid_bottom") > v("neutral", "lid_bottom") and v("happy", "cheek") > 0
    assert v("sad", "brow_pos") > 0.5 and v("sad", "lid_top") > v("neutral", "lid_top")
    assert v("sinister", "brow_pos") < -0.5 and c("sinister", "iris").startswith("#FF")
    assert c("sinister", "iris") != c("neutral", "iris")
    assert v("surprised", "brow_raise") == 1.0 and v("surprised", "lid_top") == 0.0
    assert v("thinking", "brow_asym") != 0.0 and v("thinking", "gaze_x") != 0.0
    assert v("sleepy", "lid_top") >= 0.5 and v("sleepy", "lid_bottom") >= 0.4  # a slit
    assert v("sleepy", "blink_rate") < 1.0 and v("sleepy", "glow") < v("neutral", "glow")
    assert v("neutral", "brow_glow") == 1.0 and v("neutral", "glow") < 0.7  # the brow leads
    assert c("sleepy", "iris") != c("neutral", "iris")
    assert v("cat", "lid_top") == 1.0 and v("cat", "brow_width") > 0.8
    assert v("kiss", "wink") == 1.0 and v("kiss_heart", "wink") == 1.0  # like: carried
    assert v("terminator", "blink_rate") == 0.0 and v("terminator", "highlight") == 0.0
    # the effects: the eye language's defaults, carried by `like` and into every style
    fx = {n: eyes.looks[table.id_of(n)][0].effect for n in table.names}
    assert fx["thinking"] == "spin" and fx["listening"] == "pulse"
    assert fx["struggling"] == "flicker" and fx["sleepy"] == "doze" and fx["tired"] == "doze"
    assert fx["surprised"] == "flash" and fx["kiss_heart"] == "sparkle"
    assert fx["neutral"] == "none" and fx["happy"] == "none" and fx["kiss"] == "none"
    assert fx["smile_wide"] == fx["smile"] and fx["happy_plain"] == fx["happy"]  # like
    assert fx["shocked"] == "flash"
    assert all(look.effect == fx[n] for n, per in zip(table.names, eyes.looks, strict=True)
               for look in per)  # fmt: skip
    assert EYE_EFFECTS == ("none", "sparkle", "flash", "spin", "pulse", "flicker", "doze", "tear")
    assert eyes.effects["spin_hz"] == 1.0 and eyes.timing["flutter"] == 0.12

    # the later styles: the same numbers in their own palette
    for style in range(1, len(STYLES)):
        assert eyes.looks[0][style].values == eyes.looks[0][0].values
        assert c("neutral", "iris", style) != c("neutral", "iris", 0)
        assert c("sinister", "iris", style) == c("sinister", "iris", 0)  # red everywhere


def test_the_kawaii_set_speaks_the_kawaii_language() -> None:
    """Two look sets over the same params: the kawaii set's defaults (a full body with the
    shine), closed happy eyes as the upper arc alone, squeezed and sleepy eyes as the side
    segments, x x by `cross`, the tear on sad, the hollow shocked ring (no shine) and the
    angry brows; a blink in that set closes into the arc; the kawaii face style follows the
    kawaii set, the others classic, and `style_sets` maps a style by name."""
    table = load_face_table(FACE)
    eyes = table.eyes
    assert eyes is not None
    assert EYE_SETS == ("classic", "kawaii") and [s.name for s in eyes.sets] == list(EYE_SETS)
    classic, kawaii = eyes.sets
    assert not classic.shut_arc and kawaii.shut_arc and kawaii.shut_lid == 0.72
    assert kawaii.defaults["glow"] == 0.85 and kawaii.defaults["highlight"] == 1.0
    assert eyes.style_sets == tuple(int(s == "kawaii") for s in STYLES)
    assert eyes.looks is eyes.set_looks[0] and len(eyes.set_looks) == 2
    at = {p.name: i for i, p in enumerate(eyes.params)}

    def k(name: str, param: str) -> float:
        return float(eyes.set_looks[1][table.id_of(name)][0].values[at[param]])

    assert k("neutral", "glow") == 0.85 and k("neutral", "lid_top") == 0.05
    assert k("smile", "lid_bottom") == 0.65 and k("happy", "lid_bottom") == 0.75
    assert k("cat", "lid_bottom") == 0.8 and k("cat", "highlight") == 0.0
    assert k("clenched", "lid_top") == k("clenched", "lid_bottom") == 0.42
    assert k("sleepy", "lid_top") == k("sleepy", "lid_bottom") == 0.5
    assert k("tired", "cross") == 1.0 and k("tired", "highlight") == 0.0
    assert k("kiss", "wink") == 1.0 and k("kiss_heart", "wink") == 1.0
    assert eyes.set_looks[1][table.id_of("sad")][0].effect == "tear"
    assert eyes.set_looks[0][table.id_of("sad")][0].effect == "none"
    assert eyes.set_looks[1][table.id_of("thinking")][0].effect == "spin"
    # the kawaii look is not the classic one with a few numbers over it: its own defaults
    assert k("sad", "glow") == 0.7 and eyes.looks[table.id_of("sad")][0].values[at["glow"]] == 0.45
    shocked = eyes.set_looks[1][table.id_of("shocked")][0]
    assert shocked.effect == "flash" and shocked.values[at["highlight"]] == 0.0
    assert shocked.colors[EYE_ELEMENTS.index("cheek")] == "#7A9CFF"
    classic_shocked = eyes.set_looks[0][table.id_of("shocked")][0]
    assert classic_shocked.values[at["highlight"]] == 1.0 and classic_shocked.effect == "flash"
    assert k("angry", "brow_pos") == -0.9 and k("angry", "glow") == 0.9
    assert eyes.by_name == ()
    # a style mapped to the kawaii set by name; a by_name look applies to an expression of
    # that name when one exists, a name without one is ignored
    data = raw()
    data["eyes"]["style_sets"]["rose"] = "kawaii"
    data["eyes"]["by_name"] = {
        "startled": {"effect": "flash", "kawaii": {"effect": "flash", "highlight": 0.0}},
        "nobody": {"glow": 0.1},
    }
    data["expressions"].append({"name": "startled", "brief": "startled", "params": {}})
    other = parse_face_table(data)
    assert other.eyes is not None and other.eyes.style_sets[1] == 1
    assert other.eyes.by_name == ("startled", "nobody")
    startled = other.eyes.set_looks[1][other.id_of("startled")][0]
    assert startled.effect == "flash" and startled.values[at["highlight"]] == 0.0
    assert other.eyes.set_looks[0][other.id_of("startled")][0].effect == "flash"
    header = render_eye_header(other)
    sets = ", ".join("1" if s in ("rose", "kawaii") else "0" for s in STYLES)
    assert f"constexpr uint8_t kEyeSetOfStyle[kStyleCount] = {{{sets}}};" in header
    assert "kSetKawaii = 1," in header and "kFxTear = 7," in header
    assert "constexpr bool kEyeSetShutArc[kEyeSetCount] = {false, true};" in header


def test_the_header_and_the_script_carry_the_table() -> None:
    table = load_face_table(FACE)
    eyes = table.eyes
    assert eyes is not None
    header = render_eye_header(table)
    assert "constexpr int kLedCount = 48;" in header
    assert f"constexpr uint8_t kLayoutId = 0x{eyes.layout_id:02X};" in header
    assert f"constexpr int kParamCount = {len(eyes.params)};" in header
    assert "constexpr uint8_t kBrightnessDefault = 64;" in header
    assert "constexpr float kCapMa = 800.0f;" in header
    assert "constexpr uint8_t kGammaTable[256] = {" in header
    assert "kFxSpin = 3," in header and "constexpr float kSpinHz = 1.0f;" in header
    assert "constexpr float kFlutter = 0.12f;" in header
    assert eyes.gamma[0] == 0 and eyes.gamma[255] == 255 and eyes.gamma[128] == 56
    for i, name in enumerate(table.names):
        assert f"// {i} {name}" in header
    script = render_eye_script(table)
    assert script.startswith("// Generated") and "const EYE_TABLE = {" in script
    payload = json.loads(script.split("const EYE_TABLE = ", 1)[1].rsplit(";\nif", 1)[0])
    assert payload["layout_id"] == eyes.layout_id and len(payload["gamma"]) == 256
    assert len(payload["looks"]) == 2 and len(payload["looks"][0]) == len(table.expressions)
    assert payload["leds"]["sin"][2] == 1.0 and payload["leds"]["cos"][2] == 0.0
    assert payload["timing"]["blink_min_s"] == 3.0 and payload["power"]["cap_ma"] == 800


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda d: d["eyes"]["params"]["glow"].update(default=2), "default 2.0 outside"),
        (lambda d: d["expressions"][1]["eyes"].update(glow=1.5), "eyes glow 1.5 outside"),
        (lambda d: d["expressions"][1]["eyes"].update(nose=1), "eyes: unknown params nose"),
        (lambda d: d["expressions"][1]["eyes"].update(colors={"iris": "red"}), "#RRGGBB"),
        (lambda d: d["expressions"][1]["eyes"].update(colors={"nose": "#FF0000"}), "no element"),
        (lambda d: d["expressions"][1]["rose"].update(eyes={"wink": 3}), "wink 3.0 outside"),
        (lambda d: d["eyes"]["layout"]["rings"].pop(), "each eye's inner and outer ring once"),
        (lambda d: d["eyes"]["layout"]["rings"][0].update(eye="middle"), "eye of"),
        (lambda d: d["eyes"]["power"].update(cap_ma=0), "cap_ma > 0"),
        (lambda d: d["eyes"]["power"].pop("gamma"), "missing \\['gamma'\\]"),
        (lambda d: d["eyes"]["timing"].update(blink_every_s=[6, 3]), "0 < min <= max"),
        (lambda d: d["eyes"]["geometry"].update(spin=1), "unknown \\['spin'\\]"),
        (lambda d: d["eyes"]["colors"].pop("brow"), "no default for brow"),
        (lambda d: d["styles"]["rose"]["eyes"].update(glow=1), "eyes carries colors only"),
        (lambda d: d["eyes"]["params"].pop("gaze_x"), "'Y' frame's params are missing: gaze_x"),
        (lambda d: d["expressions"][8]["eyes"].update(effect="strobe"), "eyes effect 'strobe'"),
        (lambda d: d["eyes"]["effects"].update(names=["none"]), "effects: names are"),
        (lambda d: d["eyes"]["effects"].pop("spin_hz"), "missing \\['spin_hz'\\]"),
        (lambda d: d["eyes"]["timing"].update(flutter=2), "flutter is a share"),
        (lambda d: d["eyes"]["style_sets"].update(rose="neon"), "no set 'neon'"),
        (
            lambda d: d["eyes"].update(by_name={"angry": {"effect": "x"}}),
            "by_name angry: eyes effect",
        ),
        (lambda d: d["expressions"][1]["eyes"]["kawaii"].update(cross=2), "cross 2.0 outside"),
        (lambda d: d["eyes"]["sets"]["kawaii"].update(glow=1), "eyes set kawaii: unknown glow"),
        (lambda d: d["eyes"]["sets"].update(names=["kawaii", "classic"]), "sets: names are"),
    ],
)
def test_a_broken_eyes_block_is_refused_with_the_reason(edit: Any, why: str) -> None:
    data = raw()
    edit(data)
    with pytest.raises(ValueError, match=why):
        parse_face_table(data)


def test_a_table_without_eyes_still_loads_and_generates_the_mouth_alone() -> None:
    data = raw()
    del data["eyes"]
    for e in data["expressions"]:
        e.pop("eyes", None)
        for style in STYLES[1:]:
            if style in e:
                e[style].pop("eyes", None)
    for style in data["styles"].values():
        if isinstance(style, dict):
            style.pop("eyes", None)
    table = parse_face_table(copy.deepcopy(data))
    assert table.eyes is None
    from pepin.face import generated

    assert EYE_HEADER not in generated(table) and EYE_SCRIPT not in generated(table)
