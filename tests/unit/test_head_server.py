"""The head server against a fake ESP32: the face's arbiter, the clock map, the IMU lines."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fake_head import FakeClient, FakeClock, FakeHead, FakeServer

from pepin.face import FaceTable, load_face_table
from pepin.head_link import EYE_RELEASE, EYE_STANDING, Eyes, EyesFrame, decode_eyes
from pepin.head_server import (
    BrainLeases,
    ClockMap,
    EyesArbiter,
    FaceArbiter,
    HeadService,
    HeadSettings,
    MicrosUnwrapper,
    StyleArbiter,
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


def test_the_newest_style_wins_and_a_timed_one_lapses_back_to_the_standing_one() -> None:
    a = StyleArbiter()
    assert a.showing(0.0) is None
    a.set("expression", "rose", hold_s=None, now=0.0)
    a.set("llm", "manga", hold_s=2.0, now=1.0)
    assert a.showing(2.5).name == "manga"  # type: ignore[union-attr]
    assert a.showing(3.0).name == "rose" and a.sources() == ["expression"]  # type: ignore[union-attr]
    a.set("expression", "anime", hold_s=1.0, now=4.0)  # its gag over its own standing one
    assert a.showing(4.5).name == "anime" and a.showing(5.0).name == "rose"  # type: ignore[union-attr]
    a.clear("expression")
    assert a.showing(5.0) is None


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


def test_a_moment_repeated_sooner_than_its_gap_is_dropped_per_source(table: FaceTable) -> None:
    clock, _, server, service = make(table)
    service.step(0.0)
    gap = table.event("stall_look").min_gap_s
    assert gap > 0.0
    first = ask(service, server, {"cmd": "event", "source": "gaze", "name": "stall_look"})
    assert first["showing"] == "surprised" and "dropped" not in first
    run(service, clock, 2.0)  # its 1.5 s hold is over
    again = ask(service, server, {"cmd": "event", "source": "gaze", "name": "stall_look"})
    assert "dropped" in again and again["showing"] != "surprised"
    other = ask(service, server, {"cmd": "event", "source": "voice", "name": "stall_look"})
    assert other["showing"] == "surprised"  # another source's moment is its own
    run(service, clock, gap)
    later = ask(service, server, {"cmd": "event", "source": "gaze", "name": "stall_look"})
    assert later["showing"] == "surprised" and "dropped" not in later
    assert service.status(clock())["moments_dropped"] == 1
    ask(service, server, {"cmd": "event", "source": "gaze", "name": "goal_accepted"})
    ask(service, server, {"cmd": "event", "source": "gaze", "name": "goal_accepted"})
    assert service.moments_dropped == 1  # a moment without a gap is never dropped


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
    assert reply["style"] == table.style == "pepin"
    assert head.sent("C")[-1] == bytes((2, 0xF4, 0x01, 3, 1, 1, 90, 0, 64, 0))  # eyes 64, follow
    # the face's style is live: named in a config, sent as the 'C' frame's eighth byte
    reply = ask(service, server, {"cmd": "config", "style": "manga"})
    assert reply["style"] == "manga" and head.sent("C")[-1][7] == 2  # pepin 0, rose 1, manga 2
    assert head.sent("C")[-1][:7] == bytes((2, 0xF4, 0x01, 3, 1, 1, 90))  # the IMU's id kept
    assert ask(service, server, {"cmd": "status"})["style"] == "manga"
    bad = ask(service, server, {"cmd": "config", "style": "neon"})
    assert bad["type"] == "error" and "no face style 'neon'" in bad["error"]
    assert ask(service, server, {"cmd": "status"})["style"] == "manga"


def test_a_style_goes_down_once_in_the_config_and_a_gag_lapses_back(table: FaceTable) -> None:
    """The mouth's style per source, one standing and one timed, the newest drawing; only a
    change sends a 'C' (the IMU's config and id as they stand, the style its eighth byte); a gag
    lapses back to the standing style, a release to the base; the face's expressions and a
    config's base leave the requests alone; an unknown style is refused with the known ones."""
    index = {s.name: i for i, s in enumerate(table.styles)}
    clock, head, server, service = make(table)
    service.step(0.0)
    n = len(head.sent("C"))
    reply = ask(service, server, {"cmd": "style", "source": "expression", "name": "rose",
                                  "ms": None})  # fmt: skip
    assert reply == {"type": "ack", "cmd": "style", "style": "rose", "by": "expression",
                     "sent": True}  # fmt: skip
    assert len(head.sent("C")) == n + 1 and head.sent("C")[-1][7] == index["rose"]
    assert head.sent("C")[-1][0] == 1  # the IMU's config id unchanged
    again = ask(service, server, {"cmd": "style", "source": "expression", "name": "rose"})
    assert again["sent"] is False and len(head.sent("C")) == n + 1
    reply = ask(service, server, {"cmd": "style", "source": "llm", "name": "manga", "ms": 2000})
    assert reply["style"] == "manga" and head.sent("C")[-1][7] == index["manga"]
    ask(service, server, {"cmd": "express", "source": "doze", "name": "sleepy"})
    ask(service, server, {"cmd": "clear", "source": "expression"})
    status = ask(service, server, {"cmd": "status"})
    assert (status["style"], status["style_by"]) == ("manga", "llm")
    assert status["style_sources"] == ["expression", "llm"] and status["sources"] == ["doze"]
    assert "style manga (llm)" in service.report(clock.now)
    run(service, clock, 2.1)  # the gag is over: the standing rose again
    assert head.sent("C")[-1][7] == index["rose"] and len(head.sent("C")) == n + 3
    reply = ask(service, server, {"cmd": "config", "style": "anime"})  # the base, under them
    assert (reply["style"], reply["style_by"]) == ("rose", "expression")
    assert head.sent("C")[-1][7] == index["rose"]
    reply = ask(service, server, {"cmd": "style_release", "source": "expression"})
    assert reply == {"type": "ack", "cmd": "style_release", "style": "anime", "by": None,
                     "sent": True}  # fmt: skip
    assert head.sent("C")[-1][7] == index["anime"]
    assert "style anime (the base)" in service.report(clock.now)
    bad = ask(service, server, {"cmd": "style", "source": "llm", "name": "neon"})
    assert bad["type"] == "error" and bad["error"] == (
        f"no face style 'neon': {', '.join(s.name for s in table.styles)}"
    )


def test_a_reboot_resends_the_style_that_draws(table: FaceTable) -> None:
    clock, head, server, service = make(table, offset_us=900_000_000)
    run(service, clock, 1.0)
    ask(service, server, {"cmd": "style", "source": "llm", "name": "manga"})
    head.offset_us = 300_000 - round(clock.now * 1e6)  # micros start over
    head.config_id = 0
    run(service, clock, 1.0)
    assert head.config_id == 1 and head.sent("C")[-1][7] == [s.name for s in table.styles].index(
        "manga"
    )


def test_subscribers_get_samples_on_the_board_clock_across_the_wrap(table: FaceTable) -> None:
    """200 Hz in 20 ms frames, the ESP32's micros 1 s from wrapping and 40 ppm fast, 1 ms each
    way: an imu_config line first, then every line carries both clocks and SI values (gyro, then
    accel), the board time of each sample's data-ready edge within 0.3 ms of the truth once the
    map has a few seconds of pongs; the filter delay is the consumer's to take off."""
    clock, head, server, service = make(
        table, offset_us=(1 << 32) - 1_000_000 - 100_000_000, fast_ppm=40.0
    )
    sub = FakeClient()
    server.inbox.append((sub, {"cmd": "subscribe", "imu": True}))
    service.step(0.0)
    truths: list[list[float]] = []
    next_batch = clock.now
    end = clock.now + 5.0
    while clock.now < end:
        clock.now += 0.00025  # finer than the transit: a coarse grid biases the round trip
        if clock.now >= next_batch + 0.02:
            times = [next_batch + 0.005 * (i + 1) for i in range(4)]
            head.imu(times)
            truths.append(times)
            next_batch = times[-1]
        service.step(0.0)
    lines = [json.loads(line) for line in sub.posted]
    assert lines[0] == {"type": "imu_config", "cfg": 1, "rate_hz": 200, "dlpf": 3,
                        "gyro_fs_dps": 500, "accel_fs_g": 4, "filter_delay_s": 0.0048}  # fmt: skip
    lines = lines[1:]
    assert len(lines) >= len(truths) - 2
    last = lines[-1]
    assert set(last) == {"type", "cfg", "samples"} and last["type"] == "imu" and last["cfg"] == 1
    _, _, gx, gy, gz, ax, ay, az = last["samples"][1]  # the fake's sample i has gyro i,-i,2i
    assert (gx, gy, gz) == pytest.approx((math.radians(1 / 65.5), -math.radians(1 / 65.5),
                                          2 * math.radians(1 / 65.5)), rel=1e-5)  # fmt: skip
    assert (ax, ay, az) == pytest.approx((0.0, 0.0, 9.80665), rel=1e-5)  # 8192 counts at 4 g
    stamps = [s[1] for line in lines for s in line["samples"]]
    assert stamps == sorted(stamps) and stamps[-1] > 1 << 32  # unwrapped, past the wrap
    by_line = {round(line["samples"][-1][1]): line for line in lines}
    truth_of = {round(head.esp_us(t[-1])): t for t in truths}
    checked = 0
    for esp, line in by_line.items():
        if esp in truth_of and truth_of[esp][-1] > clock.now - 1.0:
            for sample, truth in zip(line["samples"], truth_of[esp], strict=True):
                assert sample[0] == pytest.approx(truth, abs=3e-4)
            checked += 1
    assert checked > 40
    assert service.imu_gaps == 0
    assert service.status(clock.now)["chip_rate_hz"] == pytest.approx(200.0 * (1 + 40e-6), rel=1e-3)


def test_a_config_change_reaches_the_subscribers(table: FaceTable) -> None:
    _, _, server, service = make(table)
    sub = FakeClient()
    server.inbox.append((sub, {"cmd": "subscribe", "imu": True}))
    service.step(0.0)
    ask(service, server, {"cmd": "config", "imu_rate_hz": 1000, "gyro_fs": 2})
    change = json.loads(sub.posted[-1])
    assert change["type"] == "imu_config" and change["cfg"] == 2
    assert change["rate_hz"] == 1000 and change["gyro_fs_dps"] == 1000


def test_only_a_sampling_change_takes_a_new_imu_config_id(table: FaceTable) -> None:
    """A config of the style, the eye set or a brightness alone keeps the IMU's config id (the
    firmware leaves the chip running for the same id and sampling) and tells the subscribers
    nothing; a rate, filter or range change takes the next id and tells them; a refused value
    applies nothing."""
    _, head, server, service = make(table)
    sub = FakeClient()
    server.inbox.append((sub, {"cmd": "subscribe", "imu": True}))
    service.step(0.0)
    posted = len(sub.posted)
    for change in ({"style": "manga"}, {"eye_set": "kawaii"}, {"brightness": 90},
                   {"eyes_brightness": 0}, {"imu_rate_hz": 200, "dlpf": 3}):  # fmt: skip
        reply = ask(service, server, {"cmd": "config", **change})
        assert reply["config"]["id"] == 1 and head.sent("C")[-1][0] == 1, change
    assert len(sub.posted) == posted and head.sent("C")[-1][6:] == bytes((90, 2, 0, 2))
    for i, change in enumerate(({"imu_rate_hz": 500}, {"dlpf": 4}, {"accel_fs": 2},
                                {"gyro_fs": 2}), start=2):  # fmt: skip
        reply = ask(service, server, {"cmd": "config", **change})
        assert reply["config"]["id"] == i and head.sent("C")[-1][0] == i, change
        assert json.loads(sub.posted[-1])["cfg"] == i
    bad = ask(service, server, {"cmd": "config", "brightness": 10, "style": "neon"})
    assert bad["type"] == "error" and service.status(0.0)["brightness"] == 90


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
    report = service.report(clock.now)
    assert "face 49.5 fps (face), imu 200.1 Hz interrupt (who 0x72)" in report
    assert "ms, spread" in report and "showing neutral (nobody)" in report
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


# -- the eyes ------------------------------------------------------------------------------------


def frames(head: FakeHead) -> list[EyesFrame]:
    return [decode_eyes(p) for p in head.sent("Y")]


def test_each_field_of_each_eye_shows_the_newest_override_and_a_timed_one_lapses() -> None:
    a = EyesArbiter()
    assert a.showing(0.0) == (Eyes(), Eyes())
    a.set("expression", Eyes(brow_raise=40, lid_top=30), side=0, hold_s=None, now=0.0)
    a.set("llm", Eyes(brow_raise=-60), side=0, hold_s=2.0, now=1.0)
    a.set("gaze", Eyes(gaze_x=50, brow_raise=90), side=0, hold_s=None, now=1.5)
    a.set("llm", Eyes(lid_top=200), side=1, hold_s=None, now=1.6)  # the left lid, standing
    left, right = a.showing(1.7)
    # the gaze layer is under everyone, however new: its brow loses, its gaze shows
    assert right == Eyes(brow_raise=-60, lid_top=30, gaze_x=50)
    assert left == Eyes(brow_raise=-60, lid_top=200, gaze_x=50)
    left, right = a.showing(3.0)  # the llm's 2 s are over: back to what stood
    assert right == Eyes(brow_raise=40, lid_top=30, gaze_x=50)
    assert left.brow_raise == 40 and left.lid_top == 200
    a.clear("llm")
    assert a.showing(3.0)[0] == a.showing(3.0)[1]
    assert a.sources() == ["expression", "gaze"]


def test_an_eyes_override_goes_down_once_and_lapses_back(table: FaceTable) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    assert frames(head) == [EyesFrame(mode=EYE_RELEASE)]  # a fresh port: nothing overridden
    reply = ask(service, server, {"cmd": "eyes", "source": "voice", "brow_raise": 0.4})
    assert reply == {"type": "ack", "cmd": "eyes", "eyes": {"brow_raise": 0.4}, "by": ["voice"]}
    ask(service, server, {"cmd": "eyes", "source": "llm", "lid_top": 0.5, "brow_raise": -0.6,
                          "ms": 1500})  # fmt: skip
    assert frames(head)[-1] == EyesFrame(Eyes(brow_raise=-60, lid_top=127), EYE_STANDING)
    n = len(head.sent("Y"))
    run(service, clock, 1.0)
    assert len(head.sent("Y")) == n  # only a change goes down the wire
    run(service, clock, 0.6)
    assert frames(head)[-1] == EyesFrame(Eyes(brow_raise=40))  # the 1.5 s are over
    ask(service, server, {"cmd": "eyes", "source": "voice", "brow_raise": 0.4})
    assert len(head.sent("Y")) == n + 1  # the same again: nothing new
    reply = ask(service, server, {"cmd": "eyes_release", "source": "voice"})
    assert reply["eyes"] == {} and reply["by"] == []
    assert frames(head)[-1] == EyesFrame(mode=EYE_RELEASE)


def test_eyes_unlike_go_as_one_frame_each(table: FaceTable) -> None:
    _, head, server, service = make(table)
    service.step(0.0)
    reply = ask(service, server, {"cmd": "eyes", "source": "llm", "lid_top": 0.8, "side": "left"})
    assert reply["left"] == {"lid_top": 0.8} and reply["right"] == {}
    # the right eye left to its look by a frame of sentinels: a release frees both eyes
    assert frames(head)[-2:] == [
        EyesFrame(Eyes(lid_top=203), EYE_STANDING, side=1),
        EyesFrame(mode=EYE_STANDING, side=2),
    ]
    ask(service, server, {"cmd": "eyes", "source": "llm", "lid_top": 0.8})  # both again
    assert frames(head)[-1] == EyesFrame(Eyes(lid_top=203), EYE_STANDING, side=0)


def test_blinks_and_winks_go_at_once_on_top_of_the_override(table: FaceTable) -> None:
    _, head, server, service = make(table)
    service.step(0.0)
    ask(service, server, {"cmd": "eyes", "source": "llm", "glow": 0.3})
    reply = ask(service, server, {"cmd": "blink"})
    assert reply == {"type": "ack", "cmd": "blink", "blink": "both", "sent": True}
    assert frames(head)[-1] == EyesFrame(Eyes(glow=76), EYE_STANDING, blink=1)
    assert ask(service, server, {"cmd": "wink", "side": "right"})["blink"] == "right"
    assert frames(head)[-1].blink == 3 and frames(head)[-1].eyes == Eyes(glow=76)
    assert "left or right" in ask(service, server, {"cmd": "wink"})["error"]
    assert "side 'up'" in ask(service, server, {"cmd": "blink", "side": "up"})["error"]
    assert service.status(0.0)["eyes"]["blinks"] == 2


def test_the_gaze_is_a_layer_under_every_override_and_centred_on_a_person(
    table: FaceTable,
) -> None:
    _, head, server, service = make(table)
    service.step(0.0)
    reply = ask(service, server, {"cmd": "gaze", "x": 0.4, "y": -0.25, "tracking": False})
    assert reply["eyes"] == {"gaze_x": 0.4, "gaze_y": -0.25}
    ask(service, server, {"cmd": "eyes", "source": "llm", "brow_raise": 0.3})
    assert frames(head)[-1].eyes == Eyes(brow_raise=30, gaze_x=40, gaze_y=-25)
    ask(service, server, {"cmd": "gaze", "x": 0.6, "y": -0.25})  # newer, still under the llm's
    ask(service, server, {"cmd": "eyes", "source": "llm", "gaze_x": -1.0})
    assert frames(head)[-1].eyes == Eyes(gaze_x=-100, gaze_y=-25)
    ask(service, server, {"cmd": "eyes_release", "source": "llm"})
    ask(service, server, {"cmd": "gaze", "x": 0.6, "y": -0.25, "tracking": True})
    assert frames(head)[-1].eyes == Eyes(gaze_x=0, gaze_y=0)  # the head points at them
    assert service.status(0.0)["eyes"]["gaze"] == {"x": 0.6, "y": -0.25, "tracking": True}
    ask(service, server, {"cmd": "eyes_release", "source": "gaze"})
    assert frames(head)[-1] == EyesFrame(mode=EYE_RELEASE)
    assert service.status(0.0)["eyes"]["gaze"] is None


def test_the_effect_is_arbitrated_beside_the_look_newest_first_and_a_timed_one_lapses() -> None:
    a = EyesArbiter()
    assert a.effect(0.0) is None
    a.set_effect("voice", "pulse", hold_s=None, now=0.0)
    a.set("voice", Eyes(lid_top=30), side=0, hold_s=None, now=0.5)  # its look stays apart
    a.set_effect("llm", "spin", hold_s=2.0, now=1.0)
    running = a.effect(1.5)
    assert running is not None and (running.source, running.name) == ("llm", "spin")
    running = a.effect(3.0)  # the llm's 2 s are over: the voice's pulse again
    assert running is not None and running.name == "pulse"
    assert a.showing(3.0)[0] == Eyes(lid_top=30) and a.sources() == ["voice"]
    a.end_effect("voice")
    assert a.effect(3.0) is None and a.showing(3.0)[0] == Eyes(lid_top=30)
    a.set_effect("voice", "doze", hold_s=None, now=4.0)
    a.clear("voice")  # a release ends the effect too
    assert a.effect(4.0) is None and a.sources() == []


def test_an_effect_goes_down_once_in_the_long_frame_and_lapses_to_the_look_s_own(
    table: FaceTable,
) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    reply = ask(service, server, {"cmd": "eyes_effect", "source": "llm", "name": "spin",
                                  "ms": 1500})  # fmt: skip
    assert reply == {"type": "ack", "cmd": "eyes_effect", "eyes": {}, "effect": "spin",
                     "by": ["llm"]}  # fmt: skip
    # an effect over no look field: a standing frame of sentinels, never a release (it ends it)
    assert frames(head)[-1] == EyesFrame(Eyes(effect="spin", effect_ms=1500), EYE_STANDING)
    n = len(head.sent("Y"))
    ask(service, server, {"cmd": "eyes", "source": "voice", "glow": 0.5})
    assert frames(head)[-1] == EyesFrame(Eyes(glow=127), EYE_STANDING)  # 16 bytes: kept
    assert len(head.sent("Y")) == n + 1
    run(service, clock, 1.6)  # the 1.5 s are over: back to the look's own effect
    assert frames(head)[-1] == EyesFrame(Eyes(glow=127, effect="none"), EYE_STANDING)
    ask(service, server, {"cmd": "eyes_effect", "source": "voice", "name": "doze"})
    ask(service, server, {"cmd": "eyes_effect", "source": "voice", "name": "doze"})
    assert [f.eyes.effect for f in frames(head)[-2:]] == ["doze", "doze"]  # each asks afresh
    assert ask(service, server, {"cmd": "eyes_effect", "source": "voice", "name": None})["by"] == [
        "voice"
    ]  # its look still stands
    assert frames(head)[-1] == EyesFrame(Eyes(glow=127, effect="none"), EYE_STANDING)
    ask(service, server, {"cmd": "eyes_effect", "source": "voice", "name": "pulse"})
    ask(service, server, {"cmd": "eyes_release", "source": "voice"})
    assert frames(head)[-1] == EyesFrame(mode=EYE_RELEASE)  # the release ends the effect


def test_a_one_shot_asked_for_without_a_time_holds_its_own_length(table: FaceTable) -> None:
    """A sparkle ends on the firmware by itself after its twinkles and their fade (1.2 s +
    2 x 0.14 s): the request lapses with it, so the look's own effect comes back."""
    clock, head, server, service = make(table)
    service.step(0.0)
    ask(service, server, {"cmd": "eyes_effect", "source": "llm", "name": "sparkle"})
    assert frames(head)[-1].eyes == Eyes(effect="sparkle", effect_ms=1480)
    run(service, clock, 1.5)
    assert frames(head)[-1] == EyesFrame(mode=EYE_RELEASE)
    assert "no eye effect 'twirl'" in ask(service, server, {"cmd": "eyes_effect",
                                                           "name": "twirl"})["error"]  # fmt: skip


def test_eyes_refusals_name_what_is_known(table: FaceTable) -> None:
    _, _, server, service = make(table)
    service.step(0.0)
    bad = ask(service, server, {"cmd": "eyes", "source": "x", "brow": 0.3})
    assert bad["type"] == "error" and "no eye field brow; the eyes have brow_pos" in bad["error"]
    empty = ask(service, server, {"cmd": "eyes", "source": "x", "ms": 500})
    assert "sets a field at least" in empty["error"]


def test_the_eyes_brightness_rides_the_config_to_a_firmware_with_eyes(table: FaceTable) -> None:
    """Firmware 3 drops a 'C' of nine bytes: its status asks for the config again and gets
    eight; a reflash to 4 is sent the brightness at its first status."""
    assert table.eyes is not None and table.eyes.power["brightness"] == 64  # as shipped
    brighter = replace(table, eyes=replace(table.eyes, power={**table.eyes.power,
                                                             "brightness": 90}))  # fmt: skip
    clock = FakeClock()
    head = FakeHead(clock)
    head.version = 3
    server = FakeServer()
    service = HeadService(lambda: head, server, brighter, HeadSettings(), clock=clock)
    run(service, clock, 0.1)
    assert head.sent("C")[-1][8] == 90 and head.config_id == 0  # dropped
    for _ in range(2):
        head.status()
        run(service, clock, 0.1)
    assert len(head.sent("C")[-1]) == 8 and head.config_id == 1
    head.version = 4  # reflashed: rebooted, but within a second of the last micros
    head.status()
    run(service, clock, 0.1)
    assert head.sent("C")[-1][8] == 90 and head.eyes_brightness == 90
    n = len(head.sent("C"))
    head.status()
    run(service, clock, 0.1)
    assert len(head.sent("C")) == n  # sent once
    reply = ask(service, server, {"cmd": "config", "eyes_brightness": 0})  # the eyes off
    assert reply["eyes_brightness"] == 0 and head.eyes_brightness == 0
    assert service.status(clock.now)["eyes"]["brightness"] == 0


def test_asleep_the_eyes_are_the_sleepy_looks_and_come_back_with_a_brain(
    table: FaceTable,
) -> None:
    clock, head, server, service = make(table)
    service.step(0.0)
    ask(service, server, {"cmd": "eyes", "source": "llm", "lid_top": 0})
    ask(service, server, {"cmd": "gaze", "x": 0.5, "y": 0.0})
    ask(service, server, {"cmd": "lease", "name": "goal_server", "seconds": 1.0})
    run(service, clock, 1.1)
    assert head.expression == table.id_of("sleepy")
    assert frames(head)[-1] == EyesFrame(mode=EYE_RELEASE)
    ask(service, server, {"cmd": "lease", "name": "goal_server", "seconds": 6.0})
    assert frames(head)[-1].eyes == Eyes(lid_top=0, gaze_x=50, gaze_y=0)


def test_a_reboot_resends_the_eyes_and_the_status_names_them(table: FaceTable) -> None:
    clock, head, server, service = make(table, offset_us=900_000_000)
    run(service, clock, 1.0)
    ask(service, server, {"cmd": "eyes", "source": "llm", "brow_pos": -0.2})
    n = len(head.sent("Y"))
    head.offset_us = 300_000 - round(clock.now * 1e6)  # micros start over
    head.config_id = 0
    run(service, clock, 1.0)
    assert len(head.sent("Y")) == n + 1 and frames(head)[-1].eyes == Eyes(brow_pos=-20)
    head.status(eyes_ma=310, eyes_brightness=64, eyes_mode="timed", eye_layout=0)
    run(service, clock, 0.1)
    status = server.broadcasts[-1]
    assert status["esp"]["eyes_ma"] == 310 and status["esp"]["version"] == 4
    assert status["eyes"]["eyes"] == {"brow_pos": -0.2} and status["eyes"]["by"] == ["llm"]
    report = service.report(clock.now)
    assert "firmware 4" in report
    assert "eyes 310 mA at brightness 64 (mode timed, layout 0, set classic)" in report
    assert "eyes overridden by llm (brightness 64, set follow," in report


def test_the_eye_set_rides_every_config_and_a_config_switches_it(table: FaceTable) -> None:
    """config/face.json's eyes.eye_set (follow) is the 'C' frame's tenth byte from the first
    one on, a style's 'C' included; a config names another, live; the status line and the
    report name the setting and the set the head shows; an unknown one is refused with the
    choices."""
    assert table.eyes is not None and table.eyes.eye_set == "follow"  # as shipped
    clock, head, server, service = make(table)
    run(service, clock, 0.1)
    assert head.sent("C")[-1][9] == 0  # follow the face style
    ask(service, server, {"cmd": "style", "source": "llm", "name": "manga"})
    assert head.sent("C")[-1][7:] == bytes((2, 64, 0))  # the style's 'C' carries it too
    reply = ask(service, server, {"cmd": "config", "eye_set": "kawaii"})
    assert reply["eye_set"] == "kawaii" and reply["sent"] is True
    assert head.sent("C")[-1][9] == 2 and head.eye_set == "kawaii"
    head.status()
    run(service, clock, 0.1)
    eyes = server.broadcasts[-1]["eyes"]
    assert (eyes["set"], eyes["set_shown"]) == ("kawaii", "kawaii")
    assert server.broadcasts[-1]["esp"]["eye_set"] == "kawaii"
    report = service.report(clock.now)
    assert "layout 0, set kawaii)" in report and "(brightness 64, set kawaii," in report
    n = len(head.sent("C"))
    head.status()
    run(service, clock, 0.1)
    assert len(head.sent("C")) == n  # the head shows what it was sent: nothing more goes
    bad = ask(service, server, {"cmd": "config", "eye_set": "cute"})
    assert bad["type"] == "error" and "no eye set 'cute': follow, classic, kawaii" in bad["error"]
    assert service.status(clock.now)["eyes"]["set"] == "kawaii"
    ask(service, server, {"cmd": "config", "eye_set": "follow"})
    assert head.sent("C")[-1][9] == 0 and head.eye_set == "classic"
    kawaii = replace(table, eyes=replace(table.eyes, eye_set="kawaii"))
    clock, head, server, service = make(kawaii)
    run(service, clock, 0.1)
    assert head.sent("C")[-1][9] == 2  # face.json's setting from the first 'C'


def test_a_firmware_without_the_eye_sets_gets_a_config_of_nine_bytes(table: FaceTable) -> None:
    """A firmware 4 built before the sets drops the first 'C' (ten bytes) whole: its status
    (41 bytes, no set) gets one of nine at once; a config's eye set is kept for a reflash,
    whose first status gets the ten bytes."""
    clock, head, server, service = make(table)
    head.eye_sets = False
    run(service, clock, 0.1)
    assert len(head.sent("C")[-1]) == 10 and head.config_id == 0  # dropped
    head.status()
    run(service, clock, 0.1)
    assert len(head.sent("C")[-1]) == 9 and head.config_id == 1
    reply = ask(service, server, {"cmd": "config", "eye_set": "kawaii"})
    assert reply["eye_set"] == "kawaii" and len(head.sent("C")[-1]) == 9
    head.status()
    run(service, clock, 0.1)
    status = server.broadcasts[-1]
    assert (status["eyes"]["set"], status["eyes"]["set_shown"]) == ("kawaii", None)
    assert "layout 0)" in service.report(clock.now)  # no set shown
    n = len(head.sent("C"))
    head.status()
    run(service, clock, 0.1)
    assert len(head.sent("C")) == n  # nine bytes are what it takes: sent once
    head.eye_sets = True  # reflashed, within a second of the last micros
    head.status()
    run(service, clock, 0.1)
    assert head.sent("C")[-1][9] == 2 and head.eye_set == "kawaii"


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
        rose = [s.name for s in table.styles].index("rose")
        client.style("rose", 60_000)
        deadline = time.monotonic() + 3.0
        while head.sent("C")[-1][7] != rose and time.monotonic() < deadline:
            time.sleep(0.01)
        assert head.sent("C")[-1][7] == rose
        client.style(None)
        deadline = time.monotonic() + 3.0
        while head.sent("C")[-1][7] == rose and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ask({"cmd": "status"}, "127.0.0.1", server.port)["style"] == table.style
        reply = ask({"cmd": "event", "source": "goal", "name": "arrived"}, "127.0.0.1", server.port)
        assert reply["showing"] == "happy"
    finally:
        client.close()
        stop.set()
        thread.join(2.0)
        server.close()
