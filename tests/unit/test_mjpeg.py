"""The MJPEG reader keeps each frame's headers, so a picture carries the moment it was taken."""

import io

import pytest

from pepin.mjpeg import capture_time, parts


def stream(frames: list[tuple[float, bytes]]) -> io.BytesIO:
    out = b""
    for stamp, body in frames:
        out += (
            b"--boundarydonotcross\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(body)}\r\nX-Timestamp: {stamp:.6f}\r\n\r\n".encode()
            + body
            + b"\r\n"
        )
    return io.BytesIO(out)


def test_every_part_comes_with_its_headers_and_whole_body() -> None:
    got = list(parts(stream([(1.5, b"\xff\xd8abc\xff\xd9"), (1.6, b"\xff\xd8" + b"x" * 100000)])))
    assert len(got) == 2
    assert got[0][1] == b"\xff\xd8abc\xff\xd9" and capture_time(got[0][0]) == 1.5
    assert len(got[1][1]) == 100002 and capture_time(got[1][0]) == 1.6
    assert got[0][0]["content-type"] == "image/jpeg"


def test_a_stream_cut_mid_frame_ends_cleanly_and_a_missing_stamp_is_none() -> None:
    whole = stream([(2.0, b"1234567890")]).getvalue()
    assert list(parts(io.BytesIO(whole[:-6]))) == []  # the body never completes
    assert capture_time({"content-type": "image/jpeg"}) is None
    assert capture_time({"x-timestamp": "soon"}) is None


GRAB = {
    "x-timestamp": "1700000000.250",  # realtime at the send
    "x-ustreamer-send-time": "5000.250",  # monotonic at the send
    "x-ustreamer-grab-time": "5000.200",  # monotonic at the V4L2 capture: 50 ms earlier
}


def test_the_grab_mode_moves_the_capture_stamp_onto_the_realtime_clock() -> None:
    """ustreamer's X-Timestamp is the SEND time; the capture is grab + (X-Timestamp - send)."""
    from pepin.mjpeg import has_grab, send_lag_s

    assert capture_time(GRAB) == 1700000000.250, "send stays the default"
    assert capture_time(GRAB, mode="grab") == pytest.approx(1700000000.200, abs=1e-6)
    renamed = dict(GRAB)
    renamed["x-ustreamer-grab-begin-time"] = renamed.pop("x-ustreamer-grab-time")  # 6.x
    assert capture_time(renamed, mode="grab") == pytest.approx(1700000000.200, abs=1e-6)
    assert send_lag_s(GRAB) == pytest.approx(0.050, abs=1e-6)
    assert has_grab(GRAB) and has_grab(renamed)


def test_without_the_extra_headers_grab_falls_back_to_the_send_time() -> None:
    from pepin.mjpeg import has_grab, send_lag_s

    plain = {"x-timestamp": "12.5"}
    assert capture_time(plain, mode="grab") == 12.5
    assert send_lag_s(plain) is None and not has_grab(plain)
    assert capture_time({}, mode="grab") is None
    with pytest.raises(ValueError, match="stamp mode"):
        capture_time(plain, mode="exposure")
