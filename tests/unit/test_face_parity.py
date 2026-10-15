"""The firmware's face and protocol, compiled for this computer, against the simulator and the
board's Python: the head must draw what the simulator shows and read what the server writes.

firmware/head_esp32/test_native/dump.cpp runs src/face_model.cpp, src/face_render.cpp,
src/face_anime.cpp and src/protocol.cpp; sim/dump.js runs sim/face.js and sim/anime.js under
node. Skipped without a C++ compiler or node.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from pepin.face import ELEMENTS, FaceTable, load_face_table, rgb565
from pepin.face_shapes import KEYS, load_shapes
from pepin.head_link import EXPRESSION, IMU, INFO, PING, encode_frame, encode_info

REPO = Path(__file__).resolve().parents[2]
FIRMWARE = REPO / "firmware" / "head_esp32"
W, H = 320, 170
TABLE = load_face_table(REPO / "config" / "face.json")
STYLES = TABLE.style_names
PARAMETRIC = [i for i, s in enumerate(TABLE.styles) if s.engine == "params"]
SHAPES = [TABLE.style_id_of(s) for s in TABLE.shape_styles]  # the shape engine's styles
ANIME = TABLE.style_id_of("anime")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path_factory.mktemp("face") / "dump"
    sources = ["test_native/dump.cpp", "src/face_model.cpp", "src/face_render.cpp",
               "src/face_anime.cpp", "src/protocol.cpp"]  # fmt: skip
    # -ffp-contract=off: no fused multiply-adds, each float operation rounded as the simulator's
    # Math.fround rounds it
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-ffp-contract=off", "-Wall", "-Wno-unused-result",
         "-I", "src", *sources, "-o", str(binary)],
        cwd=FIRMWARE, check=True, capture_output=True,
    )  # fmt: skip
    return binary


@pytest.fixture(scope="module")
def node() -> str:
    found = shutil.which("node")
    if found is None:
        pytest.skip("no node")
    return found


def run(command: list[str], script: str) -> bytes:
    return subprocess.run(command, input=script.encode(), capture_output=True, check=True).stdout


def rendered(shape: list[float], colors: list[int]) -> str:
    """An ``R`` line: the parameters and the phase, the heart, the "z", the colours."""
    return "R " + " ".join(f"{v:.6f}" for v in shape) + " " + " ".join(map(str, colors)) + "\n"


@pytest.mark.parametrize("style", PARAMETRIC, ids=[STYLES[i] for i in PARAMETRIC])
def test_every_expression_and_random_shapes_draw_the_same_pixels(
    native: Path, node: str, style: int
) -> None:
    """Every expression at rest in this style (its heart in flight for kiss_heart), and random
    shapes with every decoration at once in random colours."""
    table = load_face_table(REPO / "config" / "face.json")
    shapes = [
        [
            *e.looks[style].values,
            0.0,
            0.3 if "heart" in e.looks[style].effects else -1.0,
            0.3 if "z" in e.looks[style].effects else -1.0,
        ]
        for e in table.expressions
    ]
    colors = [[rgb565(c) for c in e.looks[style].colors] for e in table.expressions]
    rng = random.Random(7 + style)
    for _ in range(60):
        flights = [rng.random() if rng.random() < 0.5 else -1.0 for _ in range(2)]
        shapes.append([rng.uniform(p.lo, p.hi) for p in table.params] + [rng.random(), *flights])
        colors.append([rng.randrange(1 << 16) for _ in ELEMENTS])
    script = f"Y {style}\n" + "".join(rendered(s, c) for s, c in zip(shapes, colors, strict=True))
    a = run([str(native)], script)
    b = run([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) == len(shapes) * W * H * 2
    frame = W * H * 2
    background = rgb565(table.styles[style].colors["background"]).to_bytes(2, "little")
    for i in range(len(shapes)):
        fa, fb = a[i * frame : (i + 1) * frame], b[i * frame : (i + 1) * frame]
        differing = sum(1 for k in range(0, frame, 2) if fa[k : k + 2] != fb[k : k + 2])
        assert differing <= 20, f"shape {i}: {differing} pixels differ"
        if i < len(table.expressions):
            assert fa.count(background) < frame // 2, f"{table.names[i]} drew nothing"


def test_the_models_tween_and_lip_sync_alike(native: Path, node: str) -> None:
    script = "\n".join(
        [
            "S 0", "E 2 1.0 280 10", "S 50", "S 150", "S 290", "M 0.8 300", "S 320", "S 400",
            "E 9 0.5 500 420", "S 500", "S 700", "S 1000", "M 0.3 1000", "S 1100", "S 1500",
            "E 8 1.0 0 1600", "S 1600", "S 2600", "S 9000",
        ]
    )  # fmt: skip
    a = run([str(native)], script).decode().split("\n")
    b = run([node, str(FIRMWARE / "sim" / "dump.js")], script).decode().split("\n")
    assert len(a) == len(b) == 15  # 14 S lines and the empty end
    for line_a, line_b in zip(a, b, strict=True):
        for x, y in zip(line_a.split(), line_b.split(), strict=True):
            assert float(x) == pytest.approx(float(y), abs=2e-4)


def anime_script(table: FaceTable, style: int = ANIME) -> tuple[list[str], list[str]]:
    """A script through the shape engine in ``style``, and a label per D frame: every
    expression at rest and at the three talk drawings, cut into the next one half way through
    its transition, at the speech level's attack, with its heart or z in flight; then two
    minutes of the resting smile (the idle gestures, among them the morph toward the eps
    drawing and the breath), a stretch of the drooling, wagging faces, and the switch from
    the other shape styles (the drawings morph) and from a parametric one."""
    script = [f"C {style} 0"]
    labels = []
    t = 0
    for e in table.expressions:
        script += [f"E {table.id_of(e.name)} 1.0 280 {t}"]
        # a frame at the cut (the engine's talk tween starts when it sees the level) and one
        # with the drawing settled
        for dt, level in ((140, 0.0), (300, 0.0), (400, 1.0), (480, 1.0), (600, 0.5), (680, 0.5),
                          (800, 0.0), (900, 0.0)):  # fmt: skip
            if level > 0:
                script += [f"M {level} {t + dt - 60}", f"M {level} {t + dt - 20}"]
            script += [f"D {t + dt}"]
            labels.append(f"{e.name} +{dt} ms level {level}")
        t += 1000
    script += [f"E {table.id_of('smile')} 1.0 280 {t}"]
    for k in range(0, 120_000, 400):
        script += [f"D {t + k}"]
        labels.append(f"smile idle {k} ms")
    t += 120_000
    for name in ("tired", "tease", "sleepy", "kiss_heart"):
        script += [f"E {table.id_of(name)} 1.0 280 {t}"]
        for k in range(300, 4000, 230):
            script += [f"D {t + k}"]
            labels.append(f"{name} {k} ms")
        t += 4000
    for other in [*(s for s in SHAPES if s != style), PARAMETRIC[0]]:
        script += [f"C {other} {t}", f"E {table.id_of('tease')} 1.0 0 {t}", f"C {style} {t + 500}"]
        for k in (520, 640, 900):
            script += [f"D {t + k}"]
            labels.append(f"style {other} -> {style} {k - 500} ms")
        t += 2000
    return script, labels


@pytest.mark.parametrize("style", SHAPES, ids=[STYLES[i] for i in SHAPES])
def test_a_shapes_style_draws_the_same_pixels_on_the_head_and_in_the_simulator(
    native: Path, node: str, style: int
) -> None:
    table = load_face_table(REPO / "config" / "face.json")
    script, labels = anime_script(table, style)
    text = "\n".join(script) + "\n"
    a = run([str(native)], text)
    b = run([node, str(FIRMWARE / "sim" / "dump.js")], text)
    frame = W * H * 2
    assert len(a) == len(b) == len(labels) * frame
    background = rgb565(table.styles[style].colors["background"]).to_bytes(2, "little")
    for i, label in enumerate(labels):
        fa, fb = a[i * frame : (i + 1) * frame], b[i * frame : (i + 1) * frame]
        differing = sum(1 for k in range(0, frame, 2) if fa[k : k + 2] != fb[k : k + 2])
        assert differing <= 20, f"{label}: {differing} pixels differ"
        assert frame // 2 - fa.count(background) > 40, f"{label} drew nothing"  # a thin line


def test_variants_colours_and_the_heart_come_alike_on_the_head_and_in_the_simulator(
    native: Path, node: str
) -> None:
    """Thinking asked for 60 times shows itself and both variants (the same draws on both), the
    status names thinking throughout; the kiss's colours tween in; its heart flies once, a
    resend in flight does not restart it, one after it lands does."""
    table = load_face_table(REPO / "config" / "face.json")
    thinking, kiss_heart = table.id_of("thinking"), table.id_of("kiss_heart")
    script = []
    for i in range(60):
        script += [f"E {thinking} 1.0 0 {100 * i}", f"S {100 * i}"]
    start = 6000
    script += [f"E {kiss_heart} 1.0 280 {start}"]
    script += [f"S {start + t}" for t in (0, 140, 280, 500, 900)]
    script += [f"E {kiss_heart} 1.0 280 {start + 1000}", f"S {start + 1000}", f"S {start + 1700}"]
    script += [f"S {start + 1801}", f"E {kiss_heart} 1.0 280 {start + 2000}", f"S {start + 2000}"]
    a = shapes([str(native)], script)
    b = shapes([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) == 60 + 9
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a == pytest.approx(row_b, abs=2e-4)
    n = len(table.params)
    heart_at, colors_at = n + 1, n + 3
    shown = [int(row[-2]) for row in a[:60]]
    assert set(shown) == {thinking, *(table.id_of(v) for v in table.expressions[thinking].variants)}
    assert all(int(row[-1]) == thinking for row in a[:60])
    flight = a[60:]
    kiss = [rgb565(c) for c in table.expressions[kiss_heart].colors]
    neutral = [rgb565(c) for c in table.expressions[0].colors]
    lip = colors_at + ELEMENTS.index("lip")
    assert flight[0][lip] != kiss[0] and flight[2][lip] == kiss[0]  # the tween's start and end
    assert neutral[0] != kiss[0]
    hearts = [row[heart_at] for row in flight]
    heart_s = table.timing["heart_s"]
    assert hearts[:5] == pytest.approx([0.0, 0.14 / heart_s, 0.28 / heart_s, 0.5 / heart_s,
                                        0.9 / heart_s], abs=1e-4)  # fmt: skip
    assert hearts[5] == pytest.approx(1.0 / heart_s, abs=1e-4)  # the resend at 1 s: not restarted
    assert hearts[6] == pytest.approx(1.7 / heart_s, abs=1e-4)
    assert hearts[7] == -1.0 and hearts[8] == 0.0  # landed; the next one flies anew


def test_the_style_switches_live_and_alike_on_the_head_and_in_the_simulator(
    native: Path, node: str
) -> None:
    """A 'C' frame's style byte: the face shown tweens into its manga look (the manga palette's
    lip colour arrives over the default transition), the variant drawn and a heart in flight
    stay, the status names the style; back to pepin the same way; an unknown style is ignored;
    the cat's standing wave sits at its home phase after a crawl."""
    table = load_face_table(REPO / "config" / "face.json")
    thinking, kiss_heart, cat = (table.id_of(n) for n in ("thinking", "kiss_heart", "cat"))
    struggling = table.id_of("struggling")
    manga = TABLE.style_id_of("manga")
    script = [f"E {thinking} 1.0 0 0", "S 0", f"C {manga} 100", "S 100", "S 240", "S 400"]
    script += [f"E {kiss_heart} 1.0 0 1000", "S 1000", "S 1500", "C 0 1600", "S 1600", "S 1900"]
    script += ["C 7 2000", "S 2000", f"C {manga} 2100", "S 2500"]
    script += [f"E {struggling} 1.0 0 3000", *(f"S {t}" for t in range(3000, 3701, 100))]
    script += [f"E {cat} 1.0 0 3700", *(f"S {t}" for t in range(3700, 6001, 100))]
    sleepy = table.id_of("sleepy")
    script += [f"C {ANIME} 6100", f"E {sleepy} 1.0 0 6200", *(f"S {t}" for t in (6200, 7000, 8700))]
    script += [f"E {cat} 1.0 0 8800", "S 8800"]
    a = shapes([str(native)], script)
    b = shapes([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) == 46
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a == pytest.approx(row_b, abs=2e-4)
    n = len(table.params)
    lip = n + 3 + ELEMENTS.index("lip")
    style_at, shown_at, asked_at = -3, -2, -1
    shown = int(a[0][shown_at])
    pepin_lip = rgb565(table.expressions[shown].looks[0].colors[0])
    manga_lip = rgb565(table.expressions[shown].looks[manga].colors[0])
    assert a[0][style_at] == 0 and a[0][lip] == pepin_lip
    assert a[1][style_at] == manga and a[1][lip] == pepin_lip  # the tween has just started
    assert a[2][lip] not in (pepin_lip, manga_lip)  # half way
    assert a[3][lip] == manga_lip and int(a[3][shown_at]) == shown  # arrived, the variant kept
    assert all(int(row[asked_at]) == thinking for row in a[:4])
    heart_at = n + 1
    assert a[4][heart_at] == 0.0 and 0.0 < a[5][heart_at] < a[6][heart_at] < a[7][heart_at]
    assert a[6][style_at] == 0 and a[6][heart_at] == pytest.approx(
        0.6 / table.timing["heart_s"], abs=1e-4
    )
    assert a[8][style_at] == 0  # style 7: no such style, nothing changes
    assert a[9][style_at] == manga
    phase_at = n
    assert any(row[phase_at] > 0.05 for row in a[10:18])  # the crawl moves the phase
    assert a[-5][phase_at] == 0.0  # the cat's w sits at its home phase, snapped exactly
    assert a[-5][style_at] == manga and int(a[-5][asked_at]) == cat
    # the anime style: the sleeper's z flies again every z_every_s and stops with the face
    z_at = n + 2
    assert a[-4][style_at] == ANIME and a[-4][z_at] == 0.0
    assert a[-3][z_at] == pytest.approx(0.8 / table.timing["z_s"], abs=1e-4)
    assert a[-2][z_at] == pytest.approx((2.5 - table.timing["z_every_s"]) / table.timing["z_s"],
                                        abs=1e-4)  # fmt: skip
    assert a[-1][z_at] == -1.0


def extreme(table: FaceTable, values: list[float], level: float, b: float, d: float) -> list[float]:
    """``values`` as the model's shape() draws them at speech ``level``, the breath at ``b``
    (-1..1) and the drift at ``d`` (-1..1)."""
    geo = table.geometry
    at = {p.name: i for i, p in enumerate(table.params)}
    p = list(values)
    breathe = values[at["breathe"]]
    width = values[at["width"]] * (1 + 0.035 * breathe * b)
    p[at["width"]] = width * (1 - geo["speech_narrow"] * level)
    opened = values[at["open"]] + breathe * 0.045 * (0.5 + 0.5 * b)
    p[at["open"]] = min(1.0, opened + geo["speech_open"] * level)
    p[at["shift_x"]] = values[at["shift_x"]] + values[at["drift"]] / 3 * d
    return p


@pytest.mark.parametrize("style", SHAPES, ids=[STYLES[i] for i in SHAPES])
def test_the_drawn_faces_stay_inside_the_screen_with_their_margin(native: Path, style: int) -> None:
    """The shape engine's frames of test_a_shapes_style_draws_the_same_pixels...: nothing
    within margin_px of the screen's edges (the drool is shortened, the heart and the z held
    in, the gestures' deformations and the breath leave room)."""
    import numpy as np

    table = load_face_table(REPO / "config" / "face.json")
    script, labels = anime_script(table, style)
    raw = run([str(native)], "\n".join(script) + "\n")
    frames = np.frombuffer(raw, dtype="<u2").reshape(-1, H, W)
    m = int(table.geometry["margin_px"])
    edge = np.zeros((H, W), dtype=bool)
    edge[:m, :] = edge[-m:, :] = edge[:, :m] = edge[:, -m:] = True
    background = rgb565(table.styles[style].colors["background"])
    out = [labels[i] for i in range(len(frames)) if (frames[i][edge] != background).any()]
    assert len(frames) == len(labels) and out == [], out[:5]
    shapes = load_shapes(table, style=STYLES[style])
    assert shapes is not None and len(KEYS) == 3


@pytest.mark.parametrize("style", PARAMETRIC, ids=[STYLES[i] for i in PARAMETRIC])
def test_every_face_stays_inside_the_screen_with_its_margin(native: Path, style: int) -> None:
    """Every expression in this style at its extremes (the speech level fully open and closed,
    the breath's peaks, the wave's phases, the drift either way, an idle gesture on the resting
    faces, its heart in flight) draws nothing within margin_px of the screen's edges
    (2026-10-14: happy with the speech level opened past the bottom edge on the head)."""
    import itertools

    import numpy as np

    table = load_face_table(REPO / "config" / "face.json")
    at = {p.name: i for i, p in enumerate(table.params)}
    lines, labels = [], []
    for i, e in enumerate(table.expressions):
        values = e.looks[style].values
        gestures = [tuple(0.0 for _ in table.params)]
        if i in table.idle.expressions:
            gestures = [
                tuple(s * v for v in g.deltas) for g in table.idle.gestures for s in (1, -1)
            ]
        phases = (0.0, 0.25, 0.5, 0.75) if values[at["wave"]] > 0 else (0.0,)
        drifts = (1.0, -1.0) if values[at["drift"]] > 0 else (0.0,)
        effects = e.looks[style].effects
        flights = (0.0, 0.5, 1.0) if effects else (-1.0,)
        colors = [rgb565(x) for x in e.looks[style].colors]
        combos = itertools.product((0.0, 1.0), (1.0, -1.0), phases, drifts, gestures, flights)
        for level, b, ph, d, delta, k in combos:
            p = extreme(table, list(values), level, b, d)
            p = [min(max(v + dv, q.lo), q.hi) if dv else v
                 for v, dv, q in zip(p, delta, table.params, strict=True)]  # fmt: skip
            heart, z = (k if name in effects else -1.0 for name in ("heart", "z"))
            lines.append(rendered([*p, ph, heart, z], colors))
            labels.append(f"{e.name} level {level} breath {b} phase {ph} drift {d} k {k}")
    raw = run([str(native)], f"Y {style}\n" + "".join(lines))
    frames = np.frombuffer(raw, dtype="<u2").reshape(-1, H, W)
    m = int(table.geometry["margin_px"])
    edge = np.zeros((H, W), dtype=bool)
    edge[:m, :] = edge[-m:, :] = edge[:, :m] = edge[:, -m:] = True
    background = rgb565(table.styles[style].colors["background"])
    out = [labels[i] for i in range(len(frames)) if (frames[i][edge] != background).any()]
    assert len(frames) >= 300 and out == [], out[:5]


def shapes(binary: list[str], script: list[str]) -> list[list[float]]:
    """The model's shape at each ``S`` of ``script``: the parameters, the phase, the heart, the
    "z", the colours, the style, the expression shown and the one asked for."""
    out = run(binary, "\n".join(script) + "\n").decode().split("\n")
    return [[float(v) for v in line.split()] for line in out if line.strip()]


def every(step_ms: int, until_ms: int, start_ms: int = 0) -> list[str]:
    return [f"S {t}" for t in range(start_ms, until_ms + 1, step_ms)]


IDLE_PARAMS = ("smile", "asym", "shift_x", "thick")  # what a gesture moves and nothing else does


def gesturing(table: FaceTable, base: list[float], shape: list[float]) -> bool:
    names = [p.name for p in table.params]
    return any(abs(shape[names.index(n)] - base[names.index(n)]) > 0.005 for n in IDLE_PARAMS)


def test_the_idle_gestures_come_alike_on_the_head_and_in_the_simulator(
    native: Path, node: str
) -> None:
    """Two minutes of the resting smile with a stretch of speech and an event in it: both models
    (the same seed) draw the same gestures at the same moments."""
    script = ["E 1 1.0 280 0", *every(100, 40_000)]
    for t in range(40_100, 50_000, 100):  # ten seconds of speech
        script += [f"M 0.5 {t}", f"S {t}"]
    script += [*every(100, 60_000, 50_000), "E 12 1.0 280 60000", *every(100, 70_000, 60_100)]
    script += ["E 1 1.0 280 70000"]
    script += every(100, 120_000, 70_100)
    a = shapes([str(native)], script)
    b = shapes([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) > 1000
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a == pytest.approx(row_b, abs=2e-4)


def test_the_resting_face_comes_alive_and_never_while_speaking_or_on_an_event(
    native: Path,
) -> None:
    table = load_face_table(REPO / "config" / "face.json")
    lo, hi = table.idle.every_s
    rest = shapes([str(native)], ["E 1 1.0 280 0", *every(50, 180_000)])
    base = rest[100]  # 5 s in: the smile, still (breath and drift leave these four alone)
    moving = [gesturing(table, base, s) for s in rest]
    starts = [i for i in range(1, len(moving)) if moving[i] and not moving[i - 1]]
    assert len(starts) >= 180 / (hi + 2.2) - 1  # a gesture every 8-20 s
    ends = [i for i in range(1, len(moving)) if moving[i - 1] and not moving[i]]
    for end, start in zip(ends, starts[1:], strict=False):
        assert (start - end) * 0.05 >= lo - 0.1  # never sooner than every_s after the last
    assert not any(moving[8 : int(table.idle.settle_s / 0.05)])  # after the tween: settling
    # speech all along: the level opens the mouth, no gesture comes
    speech = ["E 1 1.0 280 0"]
    for t in range(0, 120_000, 50):
        speech += [f"M 0.4 {t}", f"S {t}"]
    talking = shapes([str(native)], speech)
    assert not any(gesturing(table, base, s) for s in talking[20:])
    # an event's expression (focused) gets none either
    focused = shapes([str(native)], ["E 12 1.0 280 0", *every(100, 120_000)])
    assert not any(gesturing(table, focused[10], s) for s in focused[10:])


def test_a_gesture_under_way_leaves_within_its_fade_when_an_event_comes(native: Path) -> None:
    table = load_face_table(REPO / "config" / "face.json")
    rest = shapes([str(native)], ["E 1 1.0 280 0", *every(50, 60_000)])
    base = rest[100]
    start = next(i for i in range(1, len(rest)) if gesturing(table, base, rest[i]))
    cut_ms = start * 50 + 200  # 200 ms into the first gesture, the goal server's focus comes
    script = ["E 1 1.0 280 0", *every(50, cut_ms), f"E 12 1.0 0 {cut_ms}"]
    script += every(10, cut_ms + 600, cut_ms + 10)
    after = shapes([str(native)], script)[len(every(50, cut_ms)) :]
    focus = shapes([str(native)], ["E 12 1.0 0 0", "S 0"])[0]
    names = [p.name for p in table.params]
    settled = int(table.idle.fade_s * 100) + 1  # 10 ms steps
    for row in after[settled:]:
        for n in IDLE_PARAMS:
            assert row[names.index(n)] == pytest.approx(focus[names.index(n)], abs=1e-3), n


def test_the_firmware_frames_and_parses_what_the_server_writes(native: Path) -> None:
    payloads = [(PING, b"\x01\x02\x03\x04"), (EXPRESSION, b"\x0d\x99\x18\x01"),
                (IMU, bytes(range(34))), (INFO, b"")]  # fmt: skip
    script = "".join(f"F {kind} {payload.hex() or '-'}\n" for kind, payload in payloads)
    lines = run([str(native)], script).decode().split()
    for i, (kind, payload) in enumerate(payloads):
        assert lines[2 * i] == encode_frame(kind, payload).hex()
        assert lines[2 * i + 1] == "ok"
    info = encode_info(
        [{"text": "Сервы: температура"}, {"key": "left", "value": "41 C"},
         {"bar": "battery", "frac": 0.72, "value": "72%"}, {"text": "ж" * 40}],
        7.5,
    )  # fmt: skip
    parsed = run([str(native)], f"T {info.hex()}\n").decode().split("\n")
    assert parsed[0] == "7500 4"
    assert parsed[1] == "0|0|" + "|Сервы: температура"
    assert parsed[2] == "1|0|left|41 C"
    assert parsed[3] == "2|184|battery|72%"
    assert parsed[4] == "0|0||" + "ж" * 23
    assert run([str(native)], "T 0100\n").decode() == "bad\n"


def test_the_sleepers_z_floats_up_again_and_again_in_the_manga_style(
    native: Path, node: str
) -> None:
    """In the manga style sleepy sends a "z" up at once and another every z_every_s while it
    shows, none in the pepin style; a switch to manga starts them, a switch back or another
    expression ends the one in flight at once; both models alike."""
    table = load_face_table(REPO / "config" / "face.json")
    sleepy, smile = table.id_of("sleepy"), table.id_of("smile")
    manga = TABLE.style_id_of("manga")
    life, every = (round(1000 * table.timing[k]) for k in ("z_s", "z_every_s"))
    script = [f"E {sleepy} 1.0 0 0", "S 0", f"S {life // 2}", f"S {life + 1}", f"S {every - 1}",
              f"S {every}", f"S {every + 10}", f"C {manga} 5000", "S 5000", "S 5100",
              "C 0 6000", "S 6000", f"S {6000 + life + 1}", "S 8500",
              f"C {manga} 9000", "S 9000", f"E {smile} 1.0 0 9100", "S 9100",
              f"S {9100 + life + 1}", f"S {9100 + 2 * every}"]  # fmt: skip
    a = shapes([str(native)], script)
    b = shapes([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) == 15
    for row_a, row_b in zip(a, b, strict=True):
        assert row_a == pytest.approx(row_b, abs=2e-4)
    z = [row[len(table.params) + 2] for row in a]
    assert z[:6] == [-1.0] * 6  # the pepin sleepy: no "z"
    assert z[6] == 0.0 and 0.0 < z[7] < 0.1  # manga: one at once
    assert z[8] == z[9] == z[10] == -1.0  # back in pepin: the one in flight goes, none comes
    assert z[11] == 0.0  # manga again: a fresh one
    assert z[12] == z[13] == z[14] == -1.0  # the smile: it goes at once, none after


def test_the_sleepers_z_repeats_while_sleepy_shows(native: Path) -> None:
    import itertools

    table = load_face_table(REPO / "config" / "face.json")
    sleepy, manga = table.id_of("sleepy"), TABLE.style_id_of("manga")
    life, every = (round(1000 * table.timing[k]) for k in ("z_s", "z_every_s"))
    script = [f"C {manga} 0", f"E {sleepy} 1.0 0 0", *every_step(50, 4 * every)]
    rows = shapes([str(native)], script)
    z = [row[len(table.params) + 2] for row in rows]
    launches = [i for i in range(1, len(z)) if z[i] == 0.0 and z[i - 1] != 0.0]
    assert len(launches) >= 4
    assert all(b - a == every // 50 for a, b in itertools.pairwise(launches))
    flying = sum(1 for v in z if v >= 0.0)
    assert abs(flying / len(z) - life / every) < 0.08  # in flight life / every of the time


def every_step(step_ms: int, until_ms: int, start_ms: int = 0) -> list[str]:
    return [f"S {t}" for t in range(start_ms, until_ms + 1, step_ms)]
