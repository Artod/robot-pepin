"""config/face.json and the two copies generated from it (pepin.face)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pepin.face import (
    BRIEF_WORDS,
    ELEMENTS,
    HEADER,
    SCRIPT,
    STYLE_GEOMETRY,
    face_lines,
    load_face_table,
    main,
    mouth_level,
    parse_face_table,
    render_header,
    render_script,
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
    fun = "sinister kiss kiss_heart pout thinking_flat moustache o_mouth vampire"
    for name in [*fun.split(), "terminator", "blush", "tease", "tired", "cat", "cat_blush",
                 "smile_wide", "happy_plain", "angry", "shocked"]:  # fmt: skip
        table.id_of(name)
    assert table.id_of("neutral") == 0
    # 2026-10-14 night: the lipstick print's kiss_mark (29) and kiss_full (30) went; tease and
    # tired moved up, the cats came after them (the firmware and face.json flash together).
    # 2026-10-15: sinister_red (22) and the three dimpled faces (26-28) went after the kawaii
    # review; vampire to cat_blush moved up, the smile's and happy's variants and the two new
    # faces came after them. 2026-10-15 night: tongue (18) went, tease took its drawing and
    # its numbers; thinking_flat to shocked moved up one
    for gone in ("kiss_mark", "kiss_full", "sinister_red", "smile_dimples", "happy_dimples",
                 "grin_dimples", "tongue"):  # fmt: skip
        assert gone not in table.names
    moved = (
        "thinking_flat",
        "vampire",
        "terminator",
        "blush",
        "tease",
        "tired",
        "cat",
        "cat_blush",
    )
    assert [table.id_of(n) for n in moved] == [18, 21, 22, 23, 24, 25, 26, 27]
    assert [table.id_of(n) for n in ("smile_wide", "happy_plain", "angry", "shocked")] == [
        28,
        29,
        30,
        31,
    ]
    assert len(table.expressions) == 32
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
        (lambda d: d["colors"].pop("tongue"), "no default for tongue"),
        (lambda d: d["expressions"][1].update(colors={"nose": "#FF0000"}), "no element: nose"),
        (lambda d: d["expressions"][1].update(colors={"lip": "red"}), "color lip #RRGGBB"),
        (lambda d: d["expressions"][1].update(like="kiss"), "like 'kiss' is no earlier"),
        (
            lambda d: d["expressions"][1].update(brief="one two three four five six seven"),
            "a brief of at most 6 words",
        ),
        (lambda d: d["expressions"][1].update(effects=["sparkle"]), "effect 'sparkle'"),
        (lambda d: d["expressions"][1].update(effects=["heart", "heart"]), "an effect repeats"),
        (lambda d: d["expressions"][1].update(variants=["smug"]), "variant 'smug' is no other"),
        (lambda d: d["expressions"][1].update(variants=["smile"]), "variant 'smile' is no other"),
        (lambda d: d["expressions"][1].update(variants=["thinking"]), "variants of its own"),
        (
            lambda d: d["expressions"][1].update(variants=["sad", "flat", "grin", "happy"]),
            "at most 3 variants",
        ),
        (lambda d: d.update(style="neon"), "style 'neon': one of pepin, rose, manga, anime"),
        (lambda d: d["styles"].update(neon={"like": "neon"}), "like 'neon' is no earlier style"),
        (lambda d: d["styles"].update({"Neon": {}}), "style 'Neon': a lower_snake_case"),
        (lambda d: d["styles"]["anime"].update(engine="paint"), "style anime: engine 'paint'"),
        (lambda d: d["styles"]["manga"].update(shapes="x"), "style manga: unknown shapes"),
        (lambda d: d["styles"]["rose"].update(like="manga"), "like 'manga' is no earlier style"),
        (lambda d: d["styles"].pop("pepin"), "styles: pepin first"),
        (lambda d: d["styles"]["manga"].update(glow=1), "style manga: unknown glow"),
        (lambda d: d["styles"]["manga"].update(shade=1.5), "style manga: shade in 0..1"),
        (lambda d: d["styles"]["manga"]["colors"].update(nose="#FF0000"), "no element: nose"),
        (
            lambda d: d["expressions"][1].update(manga={"params": {"grin": 1}}),
            "manga unknown params grin",
        ),
        (
            lambda d: d["expressions"][1].update(manga={"params": {"open": 3}}),
            "manga open 3.0 outside",
        ),
        (
            lambda d: d["expressions"][1].update(manga={"colors": {"lip": "red"}}),
            "manga: color lip",
        ),
        (lambda d: d["expressions"][1].update(manga={"mood": 1}), "manga has no mood"),
        (lambda d: d["expressions"][1].update(effects=["mark"]), "effect 'mark', one of heart, z"),
        (lambda d: d["expressions"][1].update(manga={"effects": ["mark"]}), "manga: effect 'mark'"),
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
    assert names == ["grin", "smirk", "hmm", "tilt", "blink", "twitch", "eps", "breath"]
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


def test_every_expression_has_a_brief_for_the_brains_list() -> None:
    """The brain's list of faces (face_lines): one line an expression, at most six words each."""
    table = load_face_table(FACE)
    for e in table.expressions:
        assert 0 < len(e.brief.split()) <= BRIEF_WORDS, e.name
    lines = face_lines(table)
    assert len(lines) == len(table.expressions)
    assert "face:kiss_heart — blows a kiss, a heart flies" in lines
    assert main(["--lines"]) == 0


def test_the_old_faces_keep_their_monochrome_and_the_new_ones_inherit() -> None:
    """Colour per element: the defaults are today's palette, so the fifteen faces of 2026-10-13
    draw as before; `like` copies an earlier face's numbers and colours under its own."""
    table = load_face_table(FACE)
    old = {"lip": "#7FE8FF", "cavity": "#08313C", "teeth": "#F4FBFF"}
    first_new = [p.name for p in table.params].index("bow")  # the decorations: none on them
    for e in table.expressions[: table.id_of("sinister") + 1]:
        colors = dict(zip(ELEMENTS, e.colors, strict=True))
        assert {k: colors[k] for k in old} == old, e.name
        defaults = tuple(p.default for p in table.params[first_new:])
        assert e.values[first_new:] == defaults, e.name
    kiss, blown = (
        table.expressions[table.id_of("kiss")],
        table.expressions[table.id_of("kiss_heart")],
    )
    assert blown.values == kiss.values and blown.colors == kiss.colors
    assert blown.effects == ("heart",) and kiss.effects == ()
    # the variants are their bases in every style but the kawaii drawings
    for base, variant in (("smile", "smile_wide"), ("happy", "happy_plain")):
        a, b = (table.expressions[table.id_of(n)] for n in (base, variant))
        assert b.looks == a.looks and b.brief != a.brief
    at = {p.name: i for i, p in enumerate(table.params)}
    angry, shocked = (table.expressions[table.id_of(n)] for n in ("angry", "shocked"))
    assert angry.values[at["teeth"]] == 1.0 and angry.values[at["smile"]] < 0  # gritted, down
    assert shocked.values[at["cheeks"]] == 1.0 and shocked.values[at["stripes"]] == 1.0
    assert shocked.values[at["open"]] > 0 and shocked.values[at["width"]] < 0.2
    assert all(e.values[at["stripes"]] == 0.0 for e in table.expressions if e.name != "shocked")


def test_thinking_draws_its_variants_in_the_firmware() -> None:
    table = load_face_table(FACE)
    thinking = table.expressions[table.id_of("thinking")]
    assert thinking.variants == ("thinking_flat",)  # the tongue is the tasty face since 10-15
    header = render_header(table)
    ids = f"{{{table.id_of('thinking_flat')}, -1, -1}},  // 8 thinking"
    assert ids in header
    block = header.split("kExpressionEffects[kStyleCount][kExpressionCount] = {")[1]
    by_style = {}
    for line in block.split("};")[0].strip().splitlines():
        by_style[line.split("// ")[1]] = line.split("{")[1].split("}")[0].split(", ")
    assert by_style["pepin"][table.id_of("kiss_heart")] == "kEffectHeart"
    assert by_style["pepin"][table.id_of("kiss")] == "0"
    assert by_style["pepin"][table.id_of("sleepy")] == "0"  # the "z" is a look's, not pepin's
    assert by_style["rose"][table.id_of("sleepy")] == "0"
    assert by_style["manga"][table.id_of("sleepy")] == "kEffectZ"
    assert by_style["anime"][table.id_of("sleepy")] == "kEffectZ"
    assert by_style["manga"][table.id_of("kiss_heart")] == "kEffectHeart"


def test_the_cat_mouth_is_a_standing_wave_on_a_smile() -> None:
    """The anime cat's w: the smile's corners up and one and a half periods of a standing wave
    (two dips, a peak between); cat_blush is it with the cheeks."""
    table = load_face_table(FACE)
    at = {p.name: i for i, p in enumerate(table.params)}
    cat = table.expressions[table.id_of("cat")].values
    assert cat[at["smile"]] > 0 and cat[at["wave"]] > 0.5 and cat[at["wave_n"]] == 1.5
    assert cat[at["wave_speed"]] == 0.0 and cat[at["open"]] == 0.0
    blushing = table.expressions[table.id_of("cat_blush")].values
    names = [p.name for p in table.params]
    assert [n for n, x, y in zip(names, cat, blushing, strict=True) if x != y] == ["cheeks"]


def test_the_styles_are_the_blocks_order_the_same_ids_and_the_old_set_untouched() -> None:
    """The styles are config/face.json's `styles` block in its order (the 'C' byte): pepin,
    rose, manga, anime. Every expression has a look per style; the pepin look is the
    expression's own numbers (the later styles' parameters at their defaults there); a later
    look starts from them, takes its block's params and that style's palette; `like` carries
    the blocks; the switch defaults to pepin; the anime style is the shape engine's (its block
    names the keyframes' directory and the engine's numbers, its looks carry the few numbers
    the engine reads)."""
    table = load_face_table(FACE)
    assert table.style_names[:4] == ("pepin", "rose", "manga", "anime")
    assert table.style == "pepin" and table.style_id == 0
    assert table.style_id_of("manga") == 2 and table.style_id_of("anime") == 3
    with pytest.raises(KeyError, match="no face style 'neon'; the face has pepin, rose"):
        table.style_id_of("neon")
    at = {p.name: i for i, p in enumerate(table.params)}
    later_only = ("hooks", "heart_lips", "bulb", "fang_side", "drool", "patch", "wag", "fang_at")
    pepin, rose, manga, anime = table.styles[:4]
    assert (pepin.engine, rose.engine, manga.engine, anime.engine) == ("params",) * 3 + ("shapes",)
    assert table.shape_styles[0] == "anime" and anime.like == "" and rose.like == ""
    assert anime.own["shapes"] == "anime" and anime.own["line_px"] > 0 and anime.own["drool_s"] > 0
    assert anime.numbers["flap_shapes"] == 3.0 and anime.colors["background"] == "#FFF5F0"
    for name in ("worried", "sinister", "blush", "cat", "tease"):  # the wobble is drawn
        assert table.expressions[table.id_of(name)].looks[3].values[at["wave"]] == 0.0
    assert table.expressions[table.id_of("struggling")].looks[3].values[at["wave"]] > 0  # crawls
    assert table.expressions[table.id_of("tease")].looks[3].values[at["wag"]] > 0
    assert table.expressions[table.id_of("sleepy")].looks[3].effects == ("z",)
    assert table.expressions[table.id_of("sleepy")].looks[3].colors[0] == "#C04860"  # pink, 10-15
    for e in table.expressions:
        assert len(e.looks) == len(table.styles) and e.looks[0].values == e.values
        for name in later_only:
            assert e.values[at[name]] == table.params[at[name]].default, (e.name, name)
    for key in STYLE_GEOMETRY:
        assert pepin.numbers[key] == table.geometry[key]
    assert pepin.numbers["shade"] == 0.0 and pepin.numbers["deco_line_px"] == 0.0
    assert pepin.numbers["flap_shapes"] == 0.0 and pepin.numbers["teeth_span"] == 1.0
    assert pepin.colors["background"] == table.colors["background"] == "#000000"
    assert rose.colors["background"] == manga.colors["background"] == "#FFF5F0"
    assert rose.numbers["tooth_gap_px"] == 0.0 and 0 < rose.numbers["shade"] < 1
    assert manga.numbers["flap_shapes"] == 3.0 and manga.numbers["teeth_span"] < 1.0
    assert manga.numbers["tongue_w_px"] > pepin.numbers["tongue_w_px"]
    assert manga.numbers["thick_max_px"] < pepin.numbers["thick_max_px"]
    neutral = table.expressions[0]
    assert neutral.looks[1].values[at["hooks"]] == 0.5 and neutral.values[at["hooks"]] == 0.0
    assert neutral.looks[2].values[at["hooks"]] < 0 and neutral.looks[2].values[at["upper"]] == 0
    for e, style in ((table.expressions[table.id_of("cat")], 1),
                     (table.expressions[table.id_of("cat")], 2)):  # fmt: skip
        palette = dict(zip(ELEMENTS, e.looks[style].colors, strict=True))
        assert palette == {k: table.styles[style].colors[k] for k in ELEMENTS}
    smile, wide = (table.expressions[table.id_of(n)] for n in ("smile", "smile_wide"))
    for style in (1, 2):  # the variant carries its base's blocks
        assert wide.looks[style].values[at["hooks"]] == smile.looks[style].values[at["hooks"]]
        assert wide.looks[style].values[at["hooks"]] < 0
    vampire = dict(zip(ELEMENTS, table.expressions[table.id_of("vampire")].looks[2].colors,
                       strict=True))  # fmt: skip
    assert vampire["cavity"] == "#C9344A" and vampire["cheek"] == manga.colors["cheek"]
    for style in (1, 2):
        blown = table.expressions[table.id_of("kiss_heart")].looks[style].values
        assert blown[at["heart_lips"]] == 1.0 and blown[at["pucker"]] == 0.0
    tired = table.expressions[table.id_of("tired")].looks[1].values  # rose: the dumbbell
    assert tired[at["bulb"]] == 1.0 and tired[at["tongue"]] == 0.0
    header = render_header(table)
    assert f"constexpr int kStyleCount = {len(table.styles)};" in header
    assert "kStyleDefault = 0;" in header
    assert "kStyleAnime = 3," in header and "kEffectZ = 2," in header
    assert "constexpr float kZS = 1.8f;" in header and "kZEveryS = 2.4f;" in header
    assert "kExpressions[kStyleCount][kExpressionCount][kParamCount]" in header
    assert "kExpressionColors[kStyleCount][kExpressionCount][kElementCount]" in header
    assert "  {  // manga" in header and "0xFFBE},  // manga" in header  # the cream card
    script = render_script(table)
    assert '"looks"' in script and '"style": 0' in script and '"styles"' in script
    switched = raw()
    switched["style"] = "manga"
    assert "kStyleDefault = 2;" in render_header(parse_face_table(switched))
    switched["style"] = "pepin"
    del switched["styles"]  # a table without the block: the base style alone
    with pytest.raises(ValueError, match="idle gesture twitch: styles are of pepin"):
        parse_face_table(switched)  # a gesture of the manga and anime styles: no such styles
    for g in switched["idle"]["gestures"]:
        g.pop("styles", None)
    plain = parse_face_table(switched)
    assert plain.style_names == ("pepin",) and plain.styles[0].colors["background"] == "#000000"
    assert all(len(e.looks) == 1 for e in plain.expressions)


def test_a_style_like_another_starts_from_it_in_the_block_the_expressions_and_the_gestures() -> (
    None
):
    """A style `like` an earlier one takes that one's numbers, palette, engine and own keys
    (never its drawings' directory), and in every expression and idle gesture that style's
    block under its own; it plays the gestures the liked style plays."""
    data = raw()
    data["styles"]["glow"] = {
        "like": "anime", "colors": {"lip": "#112233"}, "line_px": 2.0, "note": "a test",
    }  # fmt: skip
    data["expressions"][1]["glow"] = {"params": {"wag": 0.25}}  # smile
    data["idle"]["gestures"][0]["glow"] = {"sx": 1.5}  # grin: toward smile stays
    table = parse_face_table(data)
    n = table.style_id_of("glow")
    assert table.style_names[-1] == "glow" and n == len(table.styles) - 1 >= 5
    glow = table.styles[-1]
    assert glow.like == "anime" and glow.engine == "shapes" and glow.own["line_px"] == 2.0
    assert glow.own["talk_tween_ms"] == 60 and "shapes" not in glow.own  # its own directory
    assert glow.colors["lip"] == "#112233" and glow.colors["cavity"] == "#EC7488"
    assert glow.numbers["flap_shapes"] == 3.0
    at = {p.name: i for i, p in enumerate(table.params)}
    sleepy = table.expressions[table.id_of("sleepy")]
    assert sleepy.looks[n].effects == sleepy.looks[3].effects == ("z",)  # the anime block's
    assert sleepy.looks[n].values[at["breathe"]] == sleepy.looks[3].values[at["breathe"]]
    assert sleepy.looks[n].colors[0] == "#112233"  # the palette is glow's own
    tongue = table.expressions[table.id_of("tease")]  # the old tongue face, tease since 10-15
    assert tongue.looks[n].values[at["wag"]] == tongue.looks[3].values[at["wag"]] > 0
    smile = table.expressions[1]
    assert smile.looks[n].values[at["wag"]] == 0.25 and smile.looks[3].values[at["wag"]] == 0.0
    by_name = {g.name: g for g in table.idle.gestures}
    assert by_name["grin"].shapes["glow"] == {"toward": "smile", "amount": 0.7, "sx": 1.5}
    assert by_name["tilt"].shapes["glow"] == by_name["tilt"].shapes["anime"]
    breath = by_name["breath"].styles  # listed for anime: glow and every style like it play it
    assert breath[0] == "anime" and breath[-1] == "glow" and "manga" not in breath
    assert f"kStyleGlow = {n}," in render_header(table) and "glow" in render_script(table)
    data["styles"]["glow"] = {"like": "pepin"}  # like the base: the block's defaults
    table = parse_face_table(data)
    assert table.styles[-1].engine == "params" and table.styles[-1].numbers["flap_shapes"] == 0
    grin = {"toward": "smile", "amount": 0.7, "sx": 1.08}
    shapes = table.idle.gestures[0].shapes  # the anime block, the styles like it; no glow
    assert shapes["anime"] == grin and "glow" not in shapes
    assert all(v == grin for v in shapes.values())


def test_the_manga_set_follows_the_guides_not_the_pepin_curves() -> None:
    """The anime conventions (the README cites the guides): an open mouth is a bowl under a
    fixed upper lip with one teeth band and the tongue curve (happy, grin), the grin tilted
    with one sharp tooth on the raised side (sinister, no zigzag), the curious mouth a rounded
    rectangle (listening), upset a series of waves (worried, closed), teasing the cat line with
    the tongue hanging down, the O with teeth and tongue (surprised); the lip flap has three
    shapes; the idle twitches of the style play in it alone."""
    table = load_face_table(FACE)
    at = {p.name: i for i, p in enumerate(table.params)}
    manga = table.style_id_of("manga")

    def look(name: str) -> tuple[float, ...]:
        return table.expressions[table.id_of(name)].looks[manga].values

    for name in ("happy", "grin"):
        bowl = look(name)
        assert bowl[at["upper"]] == 0.0 and bowl[at["open"]] > 0.2
        assert bowl[at["teeth"]] > 0 and bowl[at["patch"]] > 0
    sinister = look("sinister")
    assert sinister[at["wave"]] == 0.0 and sinister[at["fangs"]] > 0
    assert sinister[at["fang_side"]] == 1.0 and sinister[at["asym"]] > 0
    assert sinister[at["fang_at"]] > 0.6 and sinister[at["upper"]] == 0.0
    assert look("listening")[at["corner_exp"]] >= 4.0 and look("listening")[at["teeth"]] > 0
    worried = look("worried")
    assert worried[at["wave"]] > 0 and worried[at["open"]] == 0.0 and worried[at["asym"]] != 0
    tease = look("tease")  # the old tongue face's numbers (2026-10-15 night)
    assert tease[at["tongue"]] == 0.9 and tease[at["asym"]] > 0 and tease[at["wag"]] == 0.5
    surprised = look("surprised")
    assert (
        surprised[at["teeth"]] > 0 and surprised[at["patch"]] > 0 and surprised[at["width"]] < 0.2
    )
    sleepy = table.expressions[table.id_of("sleepy")]
    assert sleepy.looks[manga].effects == ("z",) and sleepy.effects == ()
    assert sleepy.looks[1].effects == () and look("sleepy")[at["drool"]] == 0.0
    tired = look("tired")
    assert tired[at["tongue"]] == 1.0 and tired[at["bulb"]] == 0.0 and tired[at["tongue_x"]] != 0
    assert look("pout")[at["fangs"]] == 0.0 and look("thinking_flat")[at["pucker"]] == 0.0
    for name in ("neutral", "smile", "cat"):  # the closed lines open as bowls when they talk
        closed = look(name)
        assert closed[at["open"]] == 0.0 and closed[at["upper"]] == 0.0
        assert closed[at["teeth"]] > 0 and closed[at["patch"]] > 0
    assert look("kiss")[at["thick"]] > table.expressions[table.id_of("kiss")].values[at["thick"]]
    names = [g.name for g in table.idle.gestures]
    assert names == ["grin", "smirk", "hmm", "tilt", "blink", "twitch", "eps", "breath"]
    by_name = {g.name: g for g in table.idle.gestures}
    assert by_name["twitch"].styles == by_name["eps"].styles
    assert by_name["twitch"].styles[:2] == ("manga", "anime")
    assert by_name["breath"].styles[0] == "anime" and by_name["breath"].shapes["anime"]["sy"] > 1.0
    assert by_name["eps"].shapes["anime"]["toward"] == "eps"
    assert by_name["grin"].shapes["anime"]["toward"] == "smile"
    assert all(by_name[n].styles == table.style_names for n in names[:5])
    header = render_header(table)
    bits = ", ".join(
        f"0x{sum(1 << table.style_names.index(s) for s in g.styles):02X}"
        for g in table.idle.gestures
    )
    assert f"kIdleGestureStyles[kIdleGestureCount] = {{{bits}}};" in header
    assert bits.startswith("0x3F, 0x3F") and bits.endswith(
        "0x3C, 0x3C, 0x38"
    )  # twitch, eps: manga, anime, kawaii, comics; breath: anime, kawaii, comics
    bad = raw()
    bad["idle"]["gestures"][0]["styles"] = ["neon"]
    with pytest.raises(ValueError, match="idle gesture grin: styles are of"):
        parse_face_table(bad)
