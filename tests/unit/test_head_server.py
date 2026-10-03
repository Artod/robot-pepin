"""The head server against a fake ESP32: the face's arbiter, the clock map, the IMU lines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from fake_head import FakeClient, FakeClock, FakeHead, FakeServer

from pepin.face import FaceTable, load_face_table
from pepin.head_server import (
    BrainLeases,
    ClockMap,
    FaceArbiter,
    HeadService,
    HeadSettings,
    MicrosUnwrapper,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def table() -> FaceTable:
    return load_face_table(REPO / "config" / "face.json")


def make(table: FaceTable, **head: Any) -> tuple[FakeClock, FakeHead, FakeServer, HeadService]:
    clock = FakeClock()
    fake = FakeHead(clock, **head)
    server = FakeServer()
    service = HeadService(lambda: fake, server, table, HeadSettings(), clock=clock)
    return clock, fake, server, service


def run(service: HeadService, clock: FakeClock, seconds: float, dt: float = 0.01) -> None:
    end = clock.now + seconds
    while clock.now < end:
        clock.now += dt
        service.step(0.0)


def ask(service: HeadService, server: FakeServer, message: dict[str, Any]) -> dict[str, Any]:
    client = FakeClient()
    server.inbox.append((client, message))
    service.step(0.0)
    return server.replies[-1][1]


# -- the pieces ------------------------------------------------------------------------------------


def test_micros_unwrap_across_the_wrap_small_steps_back_and_a_reboot() -> None:
    u = MicrosUnwrapper()
    assert u.unwrap(4_294_967_000) == (4_294_967_000, False)
    assert u.unwrap(500) == (4_294_967_796, False)  # wrapped
    assert u.unwrap(400) == (4_294_967_696, False)  # a pong that crossed a batch
    assert u.unwrap(3_000_000_000) == (3_000_000_000, True)  # 1.3 billion us back: rebooted
    assert u.resets == 1


def test_the_clock_map_finds_offset_and_skew_under_jittery_transit() -> None:
    """The ESP32's crystal 40 ppm fast, 7 s apart, transit 1 ms plus an exponential jitter
    (mean 2 ms) and, for a sample, 1.4 ms more that is known (its read and its frame's bytes);
    pings every 0.5 s. After 20 s the map is within 0.1 ms everywhere."""
    rng = random.Random(3)
    clock_map = ClockMap(bucket_s=0.5, window_s=20.0)
    transit = 0.001
    known = 0.0014

    def esp_us(host: float) -> int:
        return round(host * 1e6 * (1 + 40e-6)) + 7_000_000

    host = 100.0
    while host < 125.0:
        host += 0.005
        arrived = host + known + transit + rng.expovariate(500.0)
        clock_map.observe(esp_us(host), arrived, known)
        if round(host * 1000) % 500 == 0:
            out = transit + rng.expovariate(1000.0)
            back = transit + rng.expovariate(1000.0)
            clock_map.observe_rtt(out + back)
            clock_map.observe(esp_us(host + out), host + out + back)
    for probe in (110.0, 120.0, 124.9, 125.2):
        assert clock_map.to_host(esp_us(probe)) == pytest.approx(probe, abs=1e-4)
    state = clock_map.state()
    assert state["esp_fast_ppm"] == pytest.approx(40.0, abs=3.0)
    assert state["min_rtt_ms"] == pytest.approx(2.0, abs=0.1)


def test_the_newest_request_wins_and_a_timed_one_lapses_back() -> None:
    a = FaceArbiter()
    assert a.showing(0.0) is None
    a.set("goal", "focused", intensity=1.0, transition_ms=280, hold_s=None, now=0.0)
    a.set("goal", "struggling", intensity=1.0, transition_ms=280, hold_s=3.0, now=1.0)
    a.set("voice", "listening", intensity=1.0, transition_ms=280, hold_s=None, now=2.0)
    assert a.showing(2.5).name == "listening"  # type: ignore[union-attr]
    a.clear("voice")
    assert a.showing(2.5).name == "struggling"  # type: ignore[union-attr]
    assert a.showing(4.0).name == "focused"  # type: ignore[union-attr]
    assert a.sources() == ["goal"]


def test_a_brain_is_lost_only_once_one_was_there() -> None:
    leases = BrainLeases()
    assert not leases.lost(0.0)
    leases.lease("goal_server", 6.0, 0.0)
    assert leases.alive(5.0) == ["goal_server"] and not leases.lost(5.0)
    assert leases.lost(6.5)


# -- the service -----------------------------------------------------------------------------------


def test_a_fresh_port_gets_the_config_the_face_and_pings(table: FaceTable) -> None:
    clock, head, _, service = make(table)
    run(service, clock, 1.2)
    assert head.config_id == 1
    assert head.sent("E")[0][0] == table.id_of("neutral")
    assert len(head.sent("Q")) in (2, 3)  # 2 Hz
    assert service.clock_map.ready  # the pongs alone map the clock


def test_expressions_events_and_their_lapses(table: FaceTable) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    reply = ask(service, server, {"cmd": "event", "source": "goal", "name": "goal_accepted"})
    assert reply == {"type": "ack", "cmd": "event", "showing": "focused", "by": "goal"}
    assert head.expression == table.id_of("focused")
    ask(service, server, {"cmd": "event", "source": "goal", "name": "recovery"})
    assert head.expression == table.id_of("struggling")
    run(service, clock, 3.1)
    assert head.expression == table.id_of("focused")  # the recovery's 3 s are over
    ask(service, server, {"cmd": "event", "source": "goal", "name": "arrived", "end": True})
    assert head.expression == table.id_of("happy")
    run(service, clock, 4.1)
    assert head.expression == table.id_of("neutral")  # "end" cleared the drive's focus
    sent = [p[0] for p in head.sent("E")]
    assert sent == [table.id_of(n) for n in ("neutral", "focused", "struggling", "focused",
                                             "happy", "neutral")]  # fmt: skip
    speaking = ask(service, server, {"cmd": "express", "source": "voice", "name": "smile",
                                     "intensity": 0.6, "transition_ms": 100})  # fmt: skip
    assert speaking["showing"] == "smile"
    assert head.sent("E")[-1] == bytes((table.id_of("smile"), 153, 100, 0))


def test_refusals_name_what_is_known(table: FaceTable) -> None:
    _, _, server, service = make(table)
    service.step(0.0)
    reply = ask(service, server, {"cmd": "express", "source": "x", "name": "smug"})
    assert reply["type"] == "error" and "neutral, smile" in reply["error"]
    assert ask(service, server, {"cmd": "event", "name": "nope"})["type"] == "error"
    assert "unknown command" in ask(service, server, {"cmd": "dance"})["error"]


def test_mouth_show_and_config_go_straight_down(table: FaceTable) -> None:
    _, head, server, service = make(table)
    service.step(0.0)
    ask(service, server, {"cmd": "mouth", "level": 0.5})
    assert head.sent("M") == [bytes((128,))]
    reply = ask(service, server, {"cmd": "show", "text": "Temps\nleft: 41/70 C", "seconds": 5})
    assert reply == {"type": "ack", "cmd": "show", "items": 2, "sent": True}
    assert head.sent("T")[0][:3] == bytes((0x88, 0x13, 2))
    reply = ask(service, server, {"cmd": "config", "imu_rate_hz": 500, "brightness": 90})
    assert reply["config"]["id"] == 2 and reply["config"]["rate_hz"] == 500
    assert head.sent("C")[-1] == bytes((2, 0xF4, 0x01, 3, 1, 1, 90))


def test_subscribers_get_samples_on_the_board_clock_across_the_wrap(table: FaceTable) -> None:
    """1 kHz in batches of 5, the ESP32's micros 1 s from wrapping and 40 ppm fast, 1 ms each
    way: every line carries both clocks, the board time within 0.3 ms of the truth (less the
    filter delay) once the map has a few seconds of pongs."""
    clock, head, server, service = make(
        table, offset_us=(1 << 32) - 1_000_000 - 100_000_000, fast_ppm=40.0
    )
    sub = FakeClient()
    server.inbox.append((sub, {"cmd": "subscribe", "imu": True}))
    service.step(0.0)
    truths: list[list[float]] = []
    next_batch = clock.now
    end = clock.now + 4.0
    while clock.now < end:
        clock.now += 0.001
        if clock.now >= next_batch + 0.005:
            times = [next_batch + 0.001 * (i + 1) for i in range(5)]
            head.imu(times)
            truths.append(times)
            next_batch = times[-1]
        service.step(0.0)
    lines = [json.loads(line) for line in sub.posted]
    assert len(lines) >= len(truths) - 2
    last = lines[-1]
    assert last["type"] == "imu" and last["cfg"] == 1 and last["rate_hz"] == 1000
    assert last["delay_s"] == 0.0048
    stamps = [s[1] for line in lines for s in line["s"]]
    assert stamps == sorted(stamps) and stamps[-1] > 1 << 32  # unwrapped, past the wrap
    by_line = {round(line["s"][-1][1]): line for line in lines}
    truth_of = {round(head.esp_us(t[-1])): t for t in truths}
    checked = 0
    for esp, line in by_line.items():
        if esp in truth_of and truth_of[esp][-1] > clock.now - 1.0:
            for sample, t in zip(line["s"], truth_of[esp], strict=True):
                assert sample[0] == pytest.approx(t - 0.0048, abs=3e-4)
            checked += 1
    assert checked > 50
    assert service.imu_gaps == 0


def test_a_reboot_resends_the_config_and_the_face(table: FaceTable) -> None:
    clock, head, server, service = make(table, offset_us=900_000_000)
    run(service, clock, 1.0)
    ask(service, server, {"cmd": "express", "source": "llm", "name": "grin"})
    assert len(head.sent("C")) == 1
    head.offset_us = 300_000 - round(clock.now * 1e6)  # micros start over
    head.config_id, head.expression = 0, 0
    run(service, clock, 1.0)
    assert len(head.sent("C")) == 2 and head.config_id == 1
    assert head.expression == table.id_of("grin")


def test_a_status_that_names_another_config_or_face_is_corrected(table: FaceTable) -> None:
    clock, head, server, service = make(table)
    run(service, clock, 0.1)
    head.status(config_id=0)
    run(service, clock, 0.1)
    head.status(config_id=0)
    run(service, clock, 0.1)
    assert len(head.sent("C")) == 2
    status = server.broadcasts[-1]
    assert status["esp"]["who_am_i"] == 0x72 and status["link"] == "up"
    n = len(head.sent("E"))
    for _ in range(2):
        head.status(expression=5)
        run(service, clock, 0.1)
    assert len(head.sent("E")) == n + 1  # the lost face sent again


def test_every_lease_lapsing_puts_the_face_to_sleep(table: FaceTable) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    ask(service, server, {"cmd": "event", "source": "goal", "name": "goal_accepted"})
    ask(service, server, {"cmd": "lease", "name": "goal_server", "seconds": 1.0})
    run(service, clock, 1.1)
    assert head.expression == table.id_of("sleepy")
    assert service.status(clock.now)["brain_lost"]
    ask(service, server, {"cmd": "lease", "name": "goal_server", "seconds": 6.0})
    assert head.expression == table.id_of("focused")


def test_an_absent_port_is_retried_and_reported(table: FaceTable) -> None:
    clock = FakeClock()
    head = FakeHead(clock)
    plugged = {"in": False}

    def open_link() -> FakeHead:
        if not plugged["in"]:
            raise FileNotFoundError("[Errno 2] No such file or directory: '/dev/pepin-head'")
        return head

    server = FakeServer()
    service = HeadService(open_link, server, table, HeadSettings(), clock=clock)
    service.step(0.0)
    assert service.status(clock.now)["link"].startswith("down: [Errno 2]")
    assert "link down" in service.report(clock.now)
    plugged["in"] = True
    clock.now += 1.1
    service.step(0.0)
    assert service.status(clock.now)["link"] == "up" and head.config_id == 1


def test_the_serial_port_is_a_raw_tty() -> None:
    """Over a pseudo-terminal: raw bytes both ways (no echo, no line discipline, 0x0D and 0x03
    untouched), an empty read on a timeout."""
    import os

    from pepin.head_server import SerialPort

    master, slave = os.openpty()
    port = SerialPort(os.ttyname(slave), 115200)
    try:
        frame = bytes((0xA5, ord("Q"), 4, 0, 0x0D, 0x03, 0x0A, 0x11, 0x42))
        os.write(master, frame)
        got = b""
        for _ in range(20):
            got += port.read(0.05)
            if len(got) >= len(frame):
                break
        assert got == frame
        assert port.read(0.0) == b""
        port.write(b"\x00\xa5\x0d")
        assert os.read(master, 16) == b"\x00\xa5\x0d"
    finally:
        port.close()
        os.close(master)
        os.close(slave)


def test_a_damaged_frame_is_counted_not_fatal(table: FaceTable) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    head.raw(b"\xa5I\x11\x00" + bytes(17) + b"\x00")  # a bad CRC
    head.status()
    run(service, clock, 0.1)
    status = server.broadcasts[-1]
    assert status["crc_errors"] == 1 and status["esp"] is not None


@pytest.mark.slow
def test_clients_over_real_sockets(table: FaceTable) -> None:
    """The door end to end: a HeadClient's fire-and-forget lines and ask()'s answers against
    the service stepping on its own thread behind a JsonLinesServer."""
    import threading
    import time

    from pepin.head_link import HeadClient, ask
    from pepin.streams import JsonLinesServer

    clock = FakeClock(time.monotonic())
    head = FakeHead(clock)
    server = JsonLinesServer(0, outbox_size=400).start()
    service = HeadService(lambda: head, server, table, HeadSettings(), clock=clock)
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            clock.now = time.monotonic()
            service.step(0.002)
            time.sleep(0.002)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    client = HeadClient("127.0.0.1", server.port, source="voice").start()
    try:
        deadline = time.monotonic() + 3.0
        while not client.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        client.express("thinking")
        client.lease(5.0)
        client.mouth(0.5)
        deadline = time.monotonic() + 3.0
        while head.expression != table.id_of("thinking") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert head.expression == table.id_of("thinking")
        assert head.sent("M") == [bytes((128,))]
        status = ask({"cmd": "status"}, "127.0.0.1", server.port)
        assert status["showing"] == "thinking" and status["leases"] == ["voice"]
        reply = ask({"cmd": "event", "source": "goal", "name": "arrived"}, "127.0.0.1", server.port)
        assert reply["showing"] == "happy"
    finally:
        client.close()
        stop.set()
        thread.join(2.0)
        server.close()
