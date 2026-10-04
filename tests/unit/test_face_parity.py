"""The firmware's face and protocol, compiled for this computer, against the simulator and the
board's Python: the head must draw what the simulator shows and read what the server writes.

firmware/head_esp32/test_native/dump.cpp runs src/face_model.cpp, src/face_render.cpp and
src/protocol.cpp; sim/dump.js runs sim/face.js under node. Skipped without a C++ compiler or
node.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from pepin.face import load_face_table
from pepin.head_link import EXPRESSION, IMU, INFO, PING, encode_frame, encode_info

REPO = Path(__file__).resolve().parents[2]
FIRMWARE = REPO / "firmware" / "head_esp32"
W, H = 320, 170

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    binary = tmp_path_factory.mktemp("face") / "dump"
    sources = ["test_native/dump.cpp", "src/face_model.cpp", "src/face_render.cpp",
               "src/protocol.cpp"]  # fmt: skip
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-Wall", "-Wno-unused-result", "-I", "src",
         *sources, "-o", str(binary)],
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


def test_every_expression_and_random_shapes_draw_the_same_pixels(native: Path, node: str) -> None:
    table = load_face_table(REPO / "config" / "face.json")
    shapes = [[*e.values, 0.0] for e in table.expressions]
    rng = random.Random(7)
    for _ in range(40):
        shapes.append([rng.uniform(p.lo, p.hi) for p in table.params] + [rng.random()])
    script = "".join("R " + " ".join(f"{v:.6f}" for v in shape) + "\n" for shape in shapes)
    a = run([str(native)], script)
    b = run([node, str(FIRMWARE / "sim" / "dump.js")], script)
    assert len(a) == len(b) == len(shapes) * W * H * 2
    frame = W * H * 2
    for i in range(len(shapes)):
        fa, fb = a[i * frame : (i + 1) * frame], b[i * frame : (i + 1) * frame]
        differing = sum(1 for k in range(0, frame, 2) if fa[k : k + 2] != fb[k : k + 2])
        assert differing <= 20, f"shape {i}: {differing} pixels differ"
        if i < len(table.expressions):
            assert fa.count(0) < frame, f"{table.names[i]} drew nothing"


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
    assert len(a) == len(b) == 15
    for line_a, line_b in zip(a, b, strict=True):
        for x, y in zip(line_a.split(), line_b.split(), strict=True):
            assert float(x) == pytest.approx(float(y), abs=2e-4)


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
