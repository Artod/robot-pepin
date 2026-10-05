"""The audio server's logic with fakes: card lookup, channel split, stamps, deadman, player."""

import array
import subprocess
import threading
import time
from typing import Any

import pytest

from pepin.audio_link import AudioClient, AudioFrame, DoaReading
from pepin.audio_server import (
    AudioServer,
    AudioService,
    CaptureLoop,
    ChannelSplitter,
    DeadStreamPolicy,
    DoaPoller,
    Framer,
    Player,
    SampleClock,
    XrunCounter,
    XvfHost,
    find_array,
    reboot_array,
    resolve_device,
)

CARDS = """\
 0 [audiocodec     ]: audiocodec - audiocodec
                      audiocodec
 1 [respeaker      ]: USB-Audio - reSpeaker XVF3800 4-Mic Array
                      Seeed Studio reSpeaker XVF3800 4-Mic Array at usb-5101000.usb-1.3, high speed
 2 [Camera         ]: USB-Audio - USB Camera
                      XIFT USB Camera at usb-5101000.usb-1.2, high speed
"""


def stereo(left: list[int], right: list[int]) -> bytes:
    """Interleaved s16le from two channels."""
    return array.array("h", [s for pair in zip(left, right, strict=True) for s in pair]).tobytes()


def mono(samples: list[int]) -> bytes:
    return array.array("h", samples).tobytes()


# -- the card ----------------------------------------------------------------------------------


def test_the_array_is_found_by_name_whatever_its_card_id() -> None:
    card = find_array(CARDS)
    assert card is not None and card.index == 1 and card.id == "respeaker"
    assert "usb-5101000.usb-1.3" in card.name  # where it hangs: the day-one bus check
    assert find_array(CARDS.replace("[respeaker      ]", "[Array          ]")).id == "Array"
    assert find_array(CARDS.split(" 1 [")[0]) is None


def test_auto_resolves_to_plughw_by_card_id_and_an_absent_array_is_an_oserror(
    tmp_path: Any,
) -> None:
    cards = tmp_path / "cards"
    cards.write_text(CARDS)
    device, where = resolve_device("auto", str(cards))
    assert device == "plughw:CARD=respeaker,DEV=0" and "XVF3800" in where
    assert resolve_device("hw:3,0", str(cards))[0] == "hw:3,0"
    cards.write_text(CARDS.split(" 1 [")[0])
    with pytest.raises(OSError, match="no XVF3800"):
        resolve_device("auto", str(cards))


# -- samples, frames and stamps ----------------------------------------------------------------


def test_one_channel_comes_out_of_interleaved_chunks_cut_mid_sample() -> None:
    wire = stereo([1, 2, 3, 4], [-1, -2, -3, -4])
    for cut in range(len(wire) + 1):
        left, right = ChannelSplitter(2, 0), ChannelSplitter(2, 1)
        assert left.take(wire[:cut]) + left.take(wire[cut:]) == mono([1, 2, 3, 4])
        assert right.take(wire[:cut]) + right.take(wire[cut:]) == mono([-1, -2, -3, -4])
    with pytest.raises(ValueError):
        ChannelSplitter(2, 2)


def test_six_channel_firmware_keeps_the_chosen_channel() -> None:
    frames = [[10 * f + c for c in range(6)] for f in range(3)]
    wire = array.array("h", [s for f in frames for s in f]).tobytes()
    assert ChannelSplitter(6, 1).take(wire) == mono([1, 11, 21])


def test_frames_are_fixed_size_and_know_their_first_sample() -> None:
    framer = Framer(frame_samples=4)
    assert framer.push(mono([0, 1, 2])) == []
    out = framer.push(mono([3, 4, 5, 6, 7, 8]))
    assert out == [(0, mono([0, 1, 2, 3])), (4, mono([4, 5, 6, 7]))]
    assert framer.samples_in == 9


def test_stamps_ignore_bursty_reads_and_follow_the_least_delayed_one() -> None:
    clock = SampleClock(rate=1000)
    # The stream really starts at t=10.0; chunks of 100 samples arrive 5..80 ms late.
    for i, late in enumerate([0.08, 0.005, 0.03, 0.06, 0.02]):
        clock.observe(10.0 + (i + 1) * 0.1 + late, (i + 1) * 100)
    assert clock.time_of(0) == pytest.approx(10.005, abs=1e-3)
    assert clock.time_of(250) == pytest.approx(10.255, abs=1e-3)


def test_stamps_follow_a_sound_card_slower_than_the_board() -> None:
    clock = SampleClock(rate=1000)
    # 200 ppm slow: a second of board time carries only 999.8 samples; a fixed 5 ms latency.
    for second in range(1, 601):
        clock.observe(second + 0.005, round(second * 999.8))
    true_time_of_last = 600.0
    assert clock.time_of(round(600 * 999.8)) == pytest.approx(true_time_of_last, abs=0.01)


# -- the capture loop --------------------------------------------------------------------------


class FakeSource:
    """Scripted reads: bytes, b"" for a silent read, or an exception to raise."""

    def __init__(self, clock: "FakeClock") -> None:
        self.clock = clock
        self.script: list[Any] = []
        self.opens = self.closes = 0
        self.refuse = False

    def open(self) -> str:
        if self.refuse:
            raise OSError("no XVF3800 among the sound cards")
        self.opens += 1
        return "plughw:CARD=respeaker,DEV=0 (fake)"

    def read(self, timeout_s: float) -> bytes:
        item = self.script.pop(0) if self.script else b""
        if isinstance(item, Exception):
            raise item
        if not item:
            self.clock.now += timeout_s
        return bytes(item)

    def close(self) -> None:
        self.closes += 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_loop(**kwargs: Any) -> tuple[CaptureLoop, FakeSource, FakeClock, list[AudioFrame]]:
    clock = FakeClock()
    source = FakeSource(clock)
    frames: list[AudioFrame] = []
    loop = CaptureLoop(source, frames.append, frame_samples=4, rate=1000, clock=clock, **kwargs)
    return loop, source, clock, frames


def test_capture_sends_the_processed_channel_as_numbered_stamped_frames() -> None:
    loop, source, clock, frames = make_loop()
    source.script = [stereo([1, 2, 3], [9, 9, 9]), stereo([4, 5, 6, 7, 8], [9] * 5)]
    loop.step()  # open
    clock.now += 0.1
    loop.step()
    loop.step()
    assert [f.pcm for f in frames] == [mono([1, 2, 3, 4]), mono([5, 6, 7, 8])]
    assert [f.seq for f in frames] == [0, 1]
    assert frames[1].stamp_s - frames[0].stamp_s == pytest.approx(0.004)
    assert loop.alive(clock.now)


def test_two_silent_seconds_are_a_stall_and_the_device_is_reopened_at_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop, source, clock, frames = make_loop(deadman_s=2.0, read_timeout_s=0.5)
    source.script = [stereo([1, 2, 3, 4], [0] * 4)]
    loop.step()  # open
    loop.step()  # one frame
    assert loop.alive(clock.now)
    for _ in range(5):  # 2.5 s of nothing
        loop.step()
    assert loop.stalls == 1 and source.closes == 1 and not loop.alive(clock.now)
    assert "CAPTURE STALLED" in caplog.text and "13-pin" in caplog.text
    loop.step()  # reopened without waiting
    assert source.opens == 2 and loop.opens == 2
    source.script = [stereo([5, 6, 7, 8], [0] * 4)]
    loop.step()
    assert frames[-1].seq == 1  # seq runs on across the reopen


def test_an_ended_stream_is_reopened_after_the_retry_time() -> None:
    loop, source, clock, _ = make_loop(retry_s=2.0)
    source.script = [EOFError("arecord exited with code 1")]
    loop.step()  # open
    loop.step()  # ends
    assert loop.ends == 1 and "arecord exited" in str(loop.error)
    loop.step()
    assert source.opens == 1  # too early
    clock.now += 2.0
    loop.step()
    assert source.opens == 2


def test_an_absent_card_is_retried_and_reported() -> None:
    loop, source, clock, _ = make_loop(retry_s=2.0)
    source.refuse = True
    loop.step()
    assert "no XVF3800" in str(loop.error) and not loop.alive(clock.now)
    source.refuse = False
    clock.now += 2.0
    loop.step()
    assert loop.opens == 1 and loop.error is None


def test_frames_per_second_counts_the_last_five_seconds() -> None:
    loop, source, clock, _ = make_loop()
    loop.step()
    for _ in range(50):
        source.script = [stereo([0] * 4, [0] * 4)]
        clock.now += 0.1
        loop.step()
    assert loop.frames_per_s(clock.now) == pytest.approx(10.0, abs=0.3)


def test_xruns_are_counted_from_the_tools_stderr() -> None:
    counter = XrunCounter()
    counter.line("overrun!!! (at least 12.345 ms long)", "arecord")
    counter.line("Recording raw data 'stdin' : Signed 16 bit Little Endian", "arecord")
    counter.line("arecord: pcm_read:2221: read error: Input/output error", "arecord")
    assert counter.count == 1


# -- the cold-boot rescue ----------------------------------------------------------------------


def test_the_policy_reboots_once_after_two_dead_streams_in_a_row() -> None:
    policy = DeadStreamPolicy(dead_streams=2, max_reboots=1)
    assert not policy.closed(delivered=False)
    assert not policy.closed(delivered=True)  # a stream that delivered resets the count
    assert not policy.closed(delivered=False)
    assert policy.closed(delivered=False)
    assert policy.reboots == 1
    assert [policy.closed(delivered=False) for _ in range(10)] == [False] * 10  # once only


def dead_stream(loop: CaptureLoop, source: FakeSource, clock: FakeClock) -> None:
    """One open whose arecord ends without a byte (the cold-boot read error), then the retry."""
    source.script = [EOFError("arecord exited with code 1")]
    loop.step()  # open
    loop.step()  # ends
    clock.now += 2.0


def test_two_streams_without_a_byte_reboot_the_array_once_and_reopen_at_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reboots: list[float] = []
    loop, source, clock, frames = make_loop(retry_s=2.0, rescue=lambda: reboots.append(1.0))
    dead_stream(loop, source, clock)
    assert reboots == []
    source.script = [EOFError("arecord exited with code 1")]
    loop.step()
    loop.step()  # second dead stream: reboot, reopen without the retry wait
    assert reboots == [1.0] and loop.reboots == 1 and "ARRAY CAPTURE DEAD" in caplog.text
    source.script = [stereo([1, 2, 3, 4], [0] * 4)]
    loop.step()  # open
    loop.step()
    assert len(frames) == 1 and "capture ALIVE after the array's reboot" in caplog.text
    for _ in range(4):
        dead_stream(loop, source, clock)
    assert reboots == [1.0]  # never again in this process


def test_a_reboot_that_does_not_cure_is_said_once_and_not_repeated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reboots: list[int] = []
    loop, source, clock, _ = make_loop(retry_s=2.0, rescue=lambda: reboots.append(1))
    for _ in range(8):
        dead_stream(loop, source, clock)
    assert reboots == [1] and caplog.text.count("STILL DEAD") == 1


def test_a_stream_that_delivered_and_ended_is_not_the_cold_boot_state() -> None:
    reboots: list[int] = []
    loop, source, clock, _ = make_loop(retry_s=2.0, rescue=lambda: reboots.append(1))
    for _ in range(5):
        source.script = [stereo([1, 2, 3, 4], [0] * 4), EOFError("read error")]
        loop.step()
        loop.step()
        loop.step()
        clock.now += 2.0
    assert loop.ends == 5 and reboots == []


def test_without_a_rescue_dead_streams_are_only_retried() -> None:
    loop, source, clock, _ = make_loop(retry_s=2.0)
    for _ in range(5):
        dead_stream(loop, source, clock)
    assert loop.reboots == 0 and source.opens == 5


def completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, stdout, "")


FOUND = "Device (USB)::device_init() -- Found device VID: 10374 PID: 26 interface: 3\n"


def test_xvf_host_answers_are_read_off_its_last_line_and_a_refusal_is_an_oserror() -> None:
    answers = {
        "GPO_READ_VALUES": completed(FOUND + "GPO_READ_VALUES 0 0 0 1 0 \n"),
        "GPI_READ_VALUES": completed(FOUND + "GPI_READ_VALUES 1 0 0 \n"),
        "PLL_LOCK_STATUS": completed(FOUND + "PLL_LOCK_STATUS -1 \n"),
        "USB_D2H_BUFFER_STABLE": completed("No device found\n", 1),
    }
    xvf = XvfHost("xvf_host", run=lambda command: answers[command[1]])
    assert xvf.read("GPO_READ_VALUES") == "GPO_READ_VALUES 0 0 0 1 0"
    with pytest.raises(OSError, match="No device found"):
        xvf.read("USB_D2H_BUFFER_STABLE")
    assert xvf.state() == (
        "GPO_READ_VALUES 0 0 0 1 0; GPI_READ_VALUES 1 0 0; PLL_LOCK_STATUS -1; "
        "USB_D2H_BUFFER_STABLE ? (xvf_host USB_D2H_BUFFER_STABLE: No device found)"
    )


def test_a_reboot_without_its_argument_or_without_the_tool_is_an_oserror() -> None:
    usage = "Command: REBOOT is write-only and expects 1 argument(s), \n0 are given.\n"
    with pytest.raises(OSError, match="expects"):
        XvfHost(run=lambda command: completed(usage)).reboot()

    def missing(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    with pytest.raises(OSError, match="REBOOT 1"):
        XvfHost("/nope/xvf_host", run=missing).reboot()


def test_the_rescue_logs_the_state_reboots_and_waits_for_the_chip_to_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []
    clock = FakeClock()
    down = {"left": 2}  # VERSION fails twice while the chip re-enumerates

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(" ".join(command[1:]))
        if command[1] == "VERSION" and down["left"]:
            down["left"] -= 1
            return completed("No device found\n", 1)
        return completed(f"{command[1]} 0 \n")

    def sleep(s: float) -> None:
        clock.now += s

    reboot_array(XvfHost(run=run), settle_s=2.0, wait_s=10.0, sleep=sleep, clock=clock)
    assert calls.index("REBOOT 1") == len(XvfHost.STATE)  # the state first, then the reboot
    assert calls.count("VERSION") == 3 and calls[-1] == XvfHost.STATE[-1]
    assert "before its reboot: GPO_READ_VALUES 0" in caplog.text
    assert "after its reboot: GPO_READ_VALUES 0" in caplog.text
    assert clock.now == pytest.approx(103.0)


# -- direction ---------------------------------------------------------------------------------


class FakeControl:
    def __init__(self, version: tuple[int, int, int] = (2, 1, 1)) -> None:
        self.fw = version
        self.answers: list[Any] = []
        self.closed = False

    def version(self) -> tuple[int, int, int]:
        return self.fw

    def doa(self) -> tuple[int, bool]:
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def test_directions_are_read_passed_on_and_a_lost_array_is_reopened() -> None:
    clock = FakeClock()
    controls = [FakeControl(), FakeControl()]
    controls[0].answers = [(135, True), OSError("[Errno 19] No such device")]
    controls[1].answers = [(140, False)]
    readings: list[DoaReading] = []
    opened = iter(controls)
    poller = DoaPoller(lambda: next(opened), readings.append, clock=clock)
    poller.step()  # open
    assert poller.firmware == "2.1.1"
    poller.step()
    assert readings == [DoaReading(100.0, 135, True)] and poller.age_s(100.5) == 0.5
    poller.step()  # fails
    assert controls[0].closed and "No such device" in str(poller.error)
    clock.now += 2.0
    poller.step()  # reopen
    poller.step()
    assert readings[-1].deg == 140 and poller.error is None and poller.failures == 1


def test_firmware_without_doa_value_is_reported_not_polled() -> None:
    control = FakeControl(version=(2, 0, 5))
    poller = DoaPoller(lambda: control, lambda _: None)
    poller.step()
    poller.step()  # would pop from an empty script if it polled
    assert "DOA_VALUE" in str(poller.error) and poller.latest is None


def test_no_pyusb_is_an_error_in_the_status_not_a_crash() -> None:
    def no_pyusb() -> Any:
        raise ImportError("No module named 'usb'")

    poller = DoaPoller(no_pyusb, lambda _: None)
    poller.step()
    assert "No module named 'usb'" in str(poller.error)


# -- speech ------------------------------------------------------------------------------------


class FakeSink:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.written = b""

    def write(self, pcm: bytes) -> None:
        self.written += pcm
        self.log.append(f"write {len(pcm)}")

    def finish(self) -> None:
        self.log.append("finish")

    def abort(self) -> None:
        self.log.append("abort")


def make_player(**kwargs: Any) -> tuple[Player, list[FakeSink], list[str], FakeClock]:
    log: list[str] = []
    sinks: list[FakeSink] = []
    clock = FakeClock()

    def open_sink() -> FakeSink:
        log.append("open")
        sinks.append(FakeSink(log))
        return sinks[-1]

    return Player(open_sink, rate=1000, clock=clock, **kwargs), sinks, log, clock


def test_an_utterance_opens_the_device_plays_and_closes_it_when_ended() -> None:
    player, sinks, log, _ = make_player()
    player.feed(b"\x01\x00" * 10)
    player.feed(b"\x02\x00" * 10)
    player.end()
    for _ in range(4):
        player.step(timeout_s=0.0)
    assert log == ["open", "write 20", "write 20", "finish"]
    assert player.played_s == pytest.approx(0.02) and not player.playing
    player.end()  # a stray end with nothing open must not spin or open anything
    player.step(timeout_s=0.0)
    assert log[-1] == "finish" and len(sinks) == 1


def test_flush_drops_the_queue_and_silences_the_device_at_once() -> None:
    player, _, log, _ = make_player()
    player.feed(b"\x01\x00" * 10)
    player.step(timeout_s=0.0)
    player.feed(b"\x02\x00" * 10)
    player.flush()
    assert log == ["open", "write 20", "abort"] and player.queued_s == 0.0
    player.step(timeout_s=0.0)
    assert log[-1] == "abort" and not player.playing


def test_a_laptop_that_vanishes_mid_sentence_releases_the_device_after_the_idle_time() -> None:
    player, _, log, clock = make_player(idle_close_s=2.0)
    player.feed(b"\x01\x00" * 10)
    player.step(timeout_s=0.0)
    player.step(timeout_s=0.0)
    assert log == ["open", "write 20"]
    clock.now += 2.5
    player.step(timeout_s=0.0)
    assert log[-1] == "finish"


def test_more_than_the_queue_limit_is_refused_and_counted() -> None:
    player, _, _, _ = make_player(max_queue_s=0.05)  # 50 samples at 1 kHz
    player.feed(b"\x00\x00" * 40)
    player.feed(b"\x00\x00" * 20)
    assert player.overflows == 1 and player.queued_s == pytest.approx(0.04)


def test_a_device_that_will_not_open_costs_one_error_per_utterance() -> None:
    clock = FakeClock()
    attempts: list[int] = []

    def broken() -> FakeSink:
        attempts.append(1)
        raise OSError("aplay: main:850: audio open error: Device or resource busy")

    player = Player(broken, rate=1000, clock=clock, idle_close_s=2.0)
    for _ in range(20):  # one utterance, chunk after chunk
        player.feed(b"\x00\x00" * 10)
        player.step(timeout_s=0.0)
        clock.now += 0.02
    assert attempts == [1] and player.errors == 1 and player.queued_s == 0.0
    player.end()  # the next utterance tries again
    player.feed(b"\x00\x00" * 10)
    player.step(timeout_s=0.0)
    assert len(attempts) == 2
    clock.now += 3.0  # or a pause does
    player.feed(b"\x00\x00" * 10)
    player.step(timeout_s=0.0)
    assert len(attempts) == 3


# -- the whole server over localhost -----------------------------------------------------------


class RealtimeSource:
    """Delivers 5 ms stereo chunks in about real time: positive on the left, -1 on the right."""

    def __init__(self) -> None:
        self.n = 0
        self.open_now = False

    def open(self) -> str:
        self.open_now = True
        return "fake"

    def read(self, timeout_s: float) -> bytes:
        time.sleep(0.005)
        left = [1 + (v % 1000) for v in range(self.n, self.n + 80)]
        self.n += 80
        return stereo(left, [-1] * 80)

    def close(self) -> None:
        self.open_now = False


@pytest.mark.slow
def test_frames_directions_status_and_playback_travel_the_socket() -> None:
    sinks: list[FakeSink] = []
    log: list[str] = []
    server = AudioServer(0, lambda: {"type": "hello", "rate": 16000, "play_rate": 16000})
    capture = CaptureLoop(RealtimeSource(), lambda f: server.broadcast(f.encode()))
    control = FakeControl()
    control.answers = [(45, True)] * 1000
    doa = DoaPoller(lambda: control, lambda r: server.broadcast(r.encode()), hz=50.0)
    player = Player(lambda: sinks.append(FakeSink(log)) or sinks[-1])
    counters = XrunCounter(), XrunCounter()
    service = AudioService(
        server.start(), capture, player, doa, capture_xruns=counters[0], play_xruns=counters[1]
    )
    stop = threading.Event()
    runner = threading.Thread(target=service.run, args=(stop,), daemon=True)
    runner.start()
    client = AudioClient("127.0.0.1", server.port).start()
    try:
        frames = [client.next_frame(2.0) for _ in range(10)]
        assert all(f is not None for f in frames)
        seqs = [f.seq for f in frames if f is not None]
        assert seqs == list(range(seqs[0], seqs[0] + 10))
        samples = array.array("h")
        samples.frombytes(frames[0].pcm)
        assert all(s > 0 for s in samples)  # the left channel, the processed one
        deadline = time.monotonic() + 2.0
        while client.latest_doa() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.latest_doa() is not None and client.latest_doa().deg == 45
        status = client.status()
        assert status is not None and status["capture_alive"] and status["firmware"] == "2.1.1"
        assert status["listeners"] == 1 and status["dropouts"] == 0
        client.play(b"\x01\x00" * 320)
        client.play_end()
        deadline = time.monotonic() + 2.0
        while "finish" not in log and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sinks and sinks[0].written == b"\x01\x00" * 320 and "finish" in log
    finally:
        client.close()
        stop.set()
        runner.join(timeout=5.0)
    assert not runner.is_alive()


@pytest.mark.slow
def test_a_status_probe_that_never_listens_gets_clean_json() -> None:
    import socket

    server = AudioServer(0, lambda: {"type": "hello"}).start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2.0) as probe:
            probe.sendall(b'{"cmd":"status"}\n')
            (client, header, _), *_ = server.commands(2.0)
            server.broadcast(AudioFrame(0, 0.0, b"\x00\x00" * 320).encode())  # not a listener
            server.reply(client, {"type": "status", "ok": True})
            received = b""
            while received.count(b"\n") < 2:
                received += probe.recv(4096)
        assert received == b'{"type":"hello"}\n{"type":"status","ok":true}\n'
        assert header == {"cmd": "status"}
    finally:
        server.close()
