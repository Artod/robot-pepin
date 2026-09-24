"""The board-minus-laptop clock measurement: NTP arithmetic, packet checks and the verdict line."""

from __future__ import annotations

import ast
import struct
from pathlib import Path

import pytest

from pepin.timesync import (
    DEFAULT_TIME_SOURCE,
    EXIT_OVER,
    EXIT_UNMEASURED,
    EXIT_WITHIN,
    PACKET_BYTES,
    Reply,
    Sample,
    best,
    from_ntp,
    main,
    measure,
    parse,
    request,
    sample,
    time_source,
    to_ntp,
    verdict,
)

REPO = Path(__file__).resolve().parents[2]


def _server_reply(sent: bytes, t2: float, t3: float, stratum: int = 10, leap: int = 0) -> bytes:
    """What a server would answer to ``sent``: our transmit echoed, its receive and transmit."""
    first = (leap << 6) | (4 << 3) | 4
    originate = struct.unpack("!Q", sent[40:48])[0]
    return struct.pack("!BB22xQQQ", first, stratum, originate, to_ntp(t2), to_ntp(t3))


def test_ntp_timestamps_round_trip_below_a_microsecond() -> None:
    """32.32 fixed point since 1900 carries a Unix time with ~0.2 ns resolution."""
    for unix in (0.0, 1_790_198_351.536103, 1_790_200_000.999999):
        assert from_ntp(to_ntp(unix)) == pytest.approx(unix, abs=1e-6)


def test_a_request_is_a_client_packet_carrying_our_transmit_time() -> None:
    packet = request(1_790_198_351.5)
    assert len(packet) == PACKET_BYTES
    assert packet[0] & 0x7 == 3, "mode 3: a client"
    assert (packet[0] >> 3) & 0x7 == 4, "version 4"
    assert from_ntp(struct.unpack("!Q", packet[40:48])[0]) == pytest.approx(1_790_198_351.5)


def test_offset_and_delay_are_the_standard_four_timestamp_arithmetic() -> None:
    """Server 0.25 s ahead, 40 ms each way, 2 ms held: offset +0.25, delay 80 ms."""
    t1 = 1000.0
    reply = Reply(0, 10, t1, t1 + 0.040 + 0.25, t1 + 0.042 + 0.25)
    s = sample(t1, reply, t1 + 0.082)
    assert s.offset_s == pytest.approx(0.25)
    assert s.delay_s == pytest.approx(0.080)


def test_a_reply_to_another_query_or_from_a_client_is_refused() -> None:
    sent = request(1000.0)
    other = request(2000.0)
    with pytest.raises(ValueError, match="another query"):
        parse(_server_reply(other, 1000.1, 1000.1), sent)
    with pytest.raises(ValueError, match="not a server"):
        parse(sent, sent)
    with pytest.raises(ValueError, match="not an NTP packet"):
        parse(b"\x24" * 12, sent)


def test_the_shortest_round_trip_is_the_sample_kept() -> None:
    """The asymmetry of an exchange can hide up to half its delay, so the fastest one wins."""
    slow = Sample(0.300, 0.900, 10, 0)
    fast = Sample(0.010, 0.060, 10, 0)
    assert best([slow, fast, Sample(0.1, 0.4, 10, 0)]) == fast
    assert best([]) is None


def test_measure_skips_silence_and_garbage_and_keeps_the_rest() -> None:
    """A radio that loses a query or returns junk costs a sample, never the measurement."""
    clock_now = iter([1000.0, 1001.0, 1002.0])
    answers = iter(["silent", "garbage", "good"])

    def exchange(packet: bytes) -> tuple[bytes, float] | None:
        sent = from_ntp(struct.unpack("!Q", packet[40:48])[0])
        kind = next(answers)
        if kind == "silent":
            return None
        if kind == "garbage":
            return b"\x00" * 10, sent + 0.05
        return _server_reply(packet, sent + 0.52, sent + 0.52), sent + 0.04

    got = measure(exchange, 3, clock=lambda: next(clock_now))
    assert len(got) == 1
    assert got[0].offset_s == pytest.approx(0.50)
    assert got[0].delay_s == pytest.approx(0.04)


def test_the_verdict_speaks_as_the_board_and_warns_over_the_threshold() -> None:
    """The client is the board: a server (laptop) 30 ms behind is the board 30 ms ahead."""
    code, line = verdict(Sample(-0.030, 0.080, 10, 0))
    assert code == EXIT_WITHIN
    assert "board - laptop +30.0 ms (the board is ahead)" in line
    code, line = verdict(Sample(0.250, 0.080, 10, 0))
    assert code == EXIT_OVER and "-250.0 ms" in line and "over 100 ms" in line


def test_an_unsynchronised_server_is_not_a_measurement() -> None:
    for s in (Sample(0.0, 0.05, 16, 0), Sample(0.0, 0.05, 10, 3), Sample(0.0, 0.05, 0, 0)):
        assert verdict(s)[0] == EXIT_UNMEASURED


def test_no_answer_at_all_is_unmeasured_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence must never read as 'the clocks agree'."""
    monkeypatch.setattr("pepin.timesync.udp_exchange", lambda *a, **k: lambda packet: None)
    assert main(["10.0.0.167", "--samples", "2"]) == EXIT_UNMEASURED


def test_the_time_source_switch_defaults_to_the_old_clock_until_measured() -> None:
    """pool is what the board runs under systemd-timesyncd; laptop becomes the default only once
    the chrony deploy is measured on the robot (CLAUDE.md rule 19). A typo is refused."""
    assert time_source({}) == DEFAULT_TIME_SOURCE == "pool"
    assert time_source({"PEPIN_TIME_SOURCE": ""}) == "pool"
    assert time_source({"PEPIN_TIME_SOURCE": "laptop"}) == "laptop"
    with pytest.raises(ValueError):
        time_source({"PEPIN_TIME_SOURCE": "laptpo"})


def test_the_shell_and_the_module_default_to_the_same_clock() -> None:
    """ros/lib.sh decides whether the laptop starts its server, timesync.py names the default:
    one flip, in both places, or the two disagree about which clock the robot runs on."""
    lib = (REPO / "ros/lib.sh").read_text()
    assert f'PEPIN_TIME_SOURCE="${{PEPIN_TIME_SOURCE:-{DEFAULT_TIME_SOURCE}}}"' in lib


def _directives(path: str) -> list[str]:
    """The non-comment, non-empty lines of a chrony configuration in the repo."""
    text = (REPO / path).read_text()
    return [ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()]


def test_the_board_steps_only_in_chronyd_s_first_updates_and_never_trusts_the_laptop() -> None:
    """A step only in the first three updates after chronyd starts (normally the boot, before the
    stack's wait ends), a slew afterwards; and a laptop minutes off after a Mac wake must be
    outvoted by the pool, not followed."""
    lines = _directives("board/chrony/chrony.conf")
    assert "makestep 1 3" in lines, "step within the first three updates only"
    assert "sourcedir /etc/chrony/sources.d" in lines, "the switch is a reload, not a restart"
    assert any(ln.startswith("pool ") and "maxsources 3" in ln for ln in lines), "three voters"
    assert not any(ln.startswith("server ") for ln in lines), "the laptop lives in sources.d"
    installer = (REPO / "board/chrony.sh").read_text()
    server = next(ln for ln in installer.splitlines() if ln.startswith("server $2"))
    assert " prefer" in server and "trust" not in server, server


def test_the_board_installer_keeps_its_way_back_and_takes_all_of_ours_on_uninstall() -> None:
    installer = (REPO / "board/chrony.sh").read_text()
    keep = installer.index("apt-get download systemd-timesyncd")
    assert keep < installer.index("apt-get install -y --no-install-recommends chrony"), (
        "the timesyncd .deb is kept before apt removes the daemon"
    )
    uninstall = installer[installer.index("do_uninstall() {") :]
    assert uninstall.index('rm -f "$LAPTOP_SOURCES"') < uninstall.index("apt-get purge -y chrony")
    assert "systemctl enable --now systemd-timesyncd" in uninstall
    assert "default_put PEPIN_TIME_SOURCE" in uninstall, "the board's copy of the switch goes too"


def test_the_laptop_server_serves_the_vm_clock_and_never_sets_it() -> None:
    lines = _directives("ros/chrony/laptop.conf")
    assert "local stratum 10" in lines
    assert not any(ln.split()[0] in ("server", "pool", "peer") for ln in lines), (
        "no sources: the reference is the VM's clock itself"
    )
    dockerfile = (REPO / "ros/chrony/Dockerfile").read_text()
    assert '"-x"' in dockerfile, "chronyd must never adjust the clock every container shares"
    start = (REPO / "ros/lib.sh").read_text().split("pepin_timeserver_up() {")[1].split("\n}")[0]
    assert "--cap-add" not in start.replace("No --cap-add SYS_TIME", "")


def test_the_restart_check_asks_for_the_server_before_it_asks_the_board() -> None:
    """Check 1.15 runs on every restart: with no server here it answers at once, not after eight
    one-second timeouts on the board."""
    offset = (REPO / "ros/time.sh").read_text().split("    offset)")[1].split(";;")[0]
    assert offset.index('grep -qx "$PEPIN_TIMESERVER"') < offset.index('ssh "root@$BOARD"')
    unit = (REPO / "board/pepin-ros.service").read_text()
    assert "NTPSynchronized" in unit, "the boot wait the chrony design relies on"


def test_the_module_runs_on_the_board_with_nothing_installed() -> None:
    """ros/time.sh pipes this file into the board's system python3: standard library only, no
    import of this package, and a __main__ entry."""
    source = (REPO / "src/pepin/timesync.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "os",
        "socket",
        "struct",
        "sys",
        "time",
    }, imported
    assert 'if __name__ == "__main__":' in source
