"""whisper.cpp as a transcriber: the request it sends, the text it hands on, the server process
it starts and stops (a stand-in server, no model)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import io
import json
import sys
import textwrap
import wave
from pathlib import Path

import pytest

from pepin.voice_live.wake import clean_transcript
from pepin.voice_live.whisper_cpp import (
    WhisperCppServer,
    WhisperCppTranscriber,
    coreml_encoder,
    free_port,
    multipart,
    server_command,
    wav_bytes,
)


class FakePost:
    """whisper-server's /inference: records each request, answers with ``text``."""

    def __init__(self, text: str = " Пепин, где ты?\n", status: int = 200) -> None:
        self.text, self.status = text, status
        self.bodies: list[bytes] = []

    def __call__(self, body: bytes, content_type: str) -> tuple[int, bytes]:
        assert content_type.startswith("multipart/form-data; boundary=")
        self.bodies.append(body)
        return self.status, json.dumps({"text": self.text}).encode()


def test_every_engine_s_text_is_one_line_of_single_spaces() -> None:
    assert clean_transcript(" Пепин, где ты?\n Езжай к принтеру.\n") == (
        "Пепин, где ты? Езжай к принтеру."
    )
    assert clean_transcript("  \n") == ""


def test_pcm_goes_out_as_a_16_khz_mono_wav() -> None:
    pcm = b"\x01\x00" * 1600
    with wave.open(io.BytesIO(wav_bytes(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == pcm


def test_the_form_carries_the_fields_and_the_file() -> None:
    body, content_type = multipart({"language": "ru"}, b"RIFF..", boundary="b0")
    assert content_type == "multipart/form-data; boundary=b0"
    assert b'--b0\r\nContent-Disposition: form-data; name="language"\r\n\r\nru\r\n' in body
    assert (
        b'name="file"; filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\nRIFF..\r\n'
        in body
    )
    assert body.endswith(b"--b0--\r\n")


def test_the_transcriber_forces_the_language_and_cleans_the_text() -> None:
    post = FakePost()
    closed: list[bool] = []
    t = WhisperCppTranscriber(post, language="ru", initial_prompt="Пепин", name="whispercpp:x",
                              on_close=lambda: closed.append(True))  # fmt: skip
    assert t.transcribe(b"\x00\x00" * 160) == "Пепин, где ты?"
    (body,) = post.bodies
    assert b'name="language"\r\n\r\nru\r\n' in body
    assert 'name="prompt"\r\n\r\nПепин\r\n'.encode() in body
    assert b'name="response_format"\r\n\r\njson\r\n' in body
    t.close()
    t.close()
    assert closed == [True] and t.name == "whispercpp:x"


def test_a_failed_request_raises_with_the_server_s_answer() -> None:
    t = WhisperCppTranscriber(FakePost(status=500))
    with pytest.raises(RuntimeError, match="answered 500"):
        t.transcribe(b"\x00\x00")


def test_the_coreml_encoder_sits_beside_the_model_quantised_or_not() -> None:
    models = Path("/m")
    assert coreml_encoder(models / "ggml-large-v3-turbo.bin") == (
        models / "ggml-large-v3-turbo-encoder.mlmodelc"
    )
    assert coreml_encoder(models / "ggml-large-v3-q5_0.bin") == (
        models / "ggml-large-v3-encoder.mlmodelc"
    )


def test_the_server_command_keeps_the_decoder_off_the_gpu_unless_asked() -> None:
    cpu = server_command(Path("/w"), Path("/m.bin"), port=9, language="ru", threads=4, gpu=False)
    assert cpu[:3] == ["/w", "--model", "/m.bin"] and "--no-gpu" in cpu
    assert cpu[cpu.index("--language") + 1] == "ru" and "--suppress-nst" in cpu
    metal = server_command(Path("/w"), Path("/m.bin"), port=9, language="ru", threads=4, gpu=True)
    assert "--no-gpu" not in metal


def test_a_missing_build_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="whisper-server"):
        WhisperCppTranscriber.start(tmp_path / "whisper-server", tmp_path / "ggml-x.bin")


STAND_IN = textwrap.dedent(
    """
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            text = " heard %d bytes\\n" % len(body)
            data = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


@pytest.mark.slow
def test_the_server_starts_answers_restarts_once_dead_and_stops(tmp_path: Path) -> None:
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    port = free_port()
    server = WhisperCppServer(
        [sys.executable, str(script), str(port)], port=port, log=tmp_path / "server.log",
        ready_timeout_s=10.0,
    )  # fmt: skip
    t = WhisperCppTranscriber(server.post, on_close=server.close)
    assert t.transcribe(b"\x00\x00" * 100).startswith("heard ")
    assert server.process is not None
    server.process.kill()
    server.process.wait()
    assert t.transcribe(b"\x00\x00" * 100).startswith("heard ")  # started again
    assert server.restarts == 1
    process = server.process
    t.close()
    assert process is not None and process.poll() is not None and not server.alive()
