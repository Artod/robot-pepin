"""The MJPEG reader keeps each frame's headers, so a picture carries the moment it was taken."""

import io

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
