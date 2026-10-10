"""Whisper through whisper.cpp, its encoder on the Mac's Neural Engine (CoreML).

Every Whisper call pays for the encoder's full 30 s window, whatever the utterance's length; on
the GPU (:class:`pepin.voice_live.wake.MlxWhisper`) that competes with the voice (Qwen3-TTS,
also on the GPU) and stretches when the ROS stack loads the CPU. whisper.cpp built with CoreML
runs the encoder as a CoreML model (``ggml-<model>-encoder.mlmodelc`` beside the ggml file) on
the Neural Engine, which nothing else here uses, and the decoder's few tokens on the CPU (or on
Metal with ``gpu``). The model lives in whisper.cpp's own server, a child process on a free
local port loaded once (:class:`WhisperCppServer`); :class:`WhisperCppTranscriber` posts each
utterance to it as a WAV and cleans the text as every engine does
(:func:`pepin.voice_live.wake.clean_transcript`).

The build (whisper.cpp v1.9.4, macOS 26, Xcode command-line tools; cmake from PyPI)::

    curl -L https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v1.9.4.tar.gz | tar xz
    cd whisper.cpp-1.9.4
    # optional: the encoder's CoreML compute units from WHISPER_COREML_UNITS (ane, all, gpu,
    # cpu); upstream hard-codes MLComputeUnitsAll in src/coreml/whisper-encoder.mm
    uvx --from cmake cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_COREML=1 \\
        -DBUILD_SHARED_LIBS=OFF
    uvx --from cmake cmake --build build -j --target whisper-server
    sh models/download-ggml-model.sh large-v3-turbo
    # the CoreML encoder: models/generate-coreml-model.sh's steps with --quantize True (the
    # script converts at float32, which the Neural Engine does not run), in a Python 3.11
    # venv holding coremltools 9.0, torch 2.7.0, openai-whisper and ane_transformers
    python models/convert-whisper-to-coreml.py --model large-v3-turbo --encoder-only True \\
        --optimize-ane True --quantize True
    xcrun coremlc compile models/coreml-encoder-large-v3-turbo.mlpackage models/
    mv models/coreml-encoder-large-v3-turbo.mlmodelc models/ggml-large-v3-turbo-encoder.mlmodelc

The first load of an encoder compiles it for the Neural Engine (once per model and macOS
version; macOS caches the result).
"""

from __future__ import annotations

import atexit
import http.client
import io
import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

from pepin.audio_link import RATE
from pepin.voice_live.wake import clean_transcript

READY_TIMEOUT_S = 900.0  # the first load compiles the encoder for the Neural Engine
REQUEST_TIMEOUT_S = 30.0
WARM_PCM = b"\x00\x00" * (RATE // 2)  # half a second of silence: the first call pays the setup
_QUANT_SUFFIX = re.compile(r"-q\d_\d$")

Post = Callable[[bytes, str], tuple[int, bytes]]  # (body, content type) -> (status, body)


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    """16-bit mono PCM as a WAV file in memory."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def multipart(
    fields: Mapping[str, str], wav: bytes, boundary: str | None = None
) -> tuple[bytes, str]:
    """A multipart/form-data body of ``fields`` and the WAV as ``file``, and its content type."""
    mark = boundary or uuid.uuid4().hex
    parts = [
        f'--{mark}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{mark}\r\nContent-Disposition: form-data; name="file"; filename="utterance.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + wav
        + b"\r\n"
    )
    parts.append(f"--{mark}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={mark}"


def coreml_encoder(model: Path) -> Path:
    """Where whisper.cpp looks for a ggml model's CoreML encoder: ``ggml-<name>.bin`` (or a
    quantised ``ggml-<name>-q5_0.bin``) -> ``ggml-<name>-encoder.mlmodelc`` beside it."""
    return model.with_name(_QUANT_SUFFIX.sub("", model.stem) + "-encoder.mlmodelc")


def server_command(
    binary: Path, model: Path, *, port: int, language: str, threads: int, gpu: bool
) -> list[str]:
    """whisper-server on 127.0.0.1:``port``: the language forced, no timestamps, the non-speech
    tokens suppressed (MLX's default), the decoder on the CPU unless ``gpu``."""
    command = [
        str(binary),
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--language", language,
        "--threads", str(threads),
        "--no-timestamps",
        "--suppress-nst",
    ]  # fmt: skip
    if not gpu:
        command.append("--no-gpu")
    return command


def free_port() -> int:
    """A local TCP port no one listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class WhisperCppServer:
    """whisper.cpp's server as a child process, the model loaded once: :meth:`post` sends one
    utterance (restarting the server once if it has died), :meth:`close` stops it."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        port: int,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
        ready_timeout_s: float = READY_TIMEOUT_S,
        request_timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        """``command`` serves HTTP on 127.0.0.1:``port``; its output is appended to ``log``
        (None: dropped)."""
        self.command, self.port, self.env, self.log = list(command), port, env, log
        self.ready_timeout_s, self.request_timeout_s = ready_timeout_s, request_timeout_s
        self.process: subprocess.Popen[bytes] | None = None
        self.load_s = 0.0  # the last start, until the server answered
        self.restarts = 0
        self._out: IO[bytes] | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)
        self._start()

    def _start(self) -> None:
        if self.log is not None and self._out is None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self._out = self.log.open("ab")
        t0 = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._out if self._out is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=dict(self.env) if self.env is not None else None,
        )
        try:
            self._wait_ready(t0 + self.ready_timeout_s)
        except BaseException:
            self.close()
            raise
        self.load_s = time.monotonic() - t0

    def _wait_ready(self, deadline: float) -> None:
        process = self.process
        assert process is not None
        while True:
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"whisper-server exited with {code} while loading{self._hint()}")
            try:
                status, _ = self._request("GET", "/health", None, "", timeout_s=1.0)
                if status == 200:
                    return
            except OSError:
                pass  # not listening yet: the model is still loading
            if time.monotonic() > deadline:
                raise TimeoutError(f"whisper-server not ready in {self.ready_timeout_s:.0f} s")
            time.sleep(0.05)

    def _hint(self) -> str:
        return f" (its output: {self.log})" if self.log is not None else ""

    def alive(self) -> bool:
        """Whether the server process is running."""
        return self.process is not None and self.process.poll() is None

    def post(self, body: bytes, content_type: str) -> tuple[int, bytes]:
        """POST /inference; a server found dead is started again once."""
        try:
            return self._request("POST", "/inference", body, content_type)
        except OSError:
            with self._lock:
                if self.alive() or self.process is None:
                    raise
                self.restarts += 1
                self._start()
            return self._request("POST", "/inference", body, content_type)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        content_type: str,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes]:
        timeout = self.request_timeout_s if timeout_s is None else timeout_s
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            headers = {"Content-Type": content_type} if body is not None else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def close(self) -> None:
        """Stop the server (SIGTERM, then SIGKILL after 3 s)."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._out is not None:
            self._out.close()
            self._out = None


class WhisperCppTranscriber:
    """Whisper through whisper.cpp's server, Russian forced like :class:`MlxWhisper` (greedy,
    falling back to higher temperatures on a doubtful reading, as both engines do by default).
    :meth:`start` loads and warms the model; ``post`` is any callable with the server's
    answer, so a test needs no process."""

    def __init__(
        self,
        post: Post,
        *,
        language: str = "ru",
        initial_prompt: str | None = None,
        name: str = "whispercpp",
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """``post`` sends a multipart body to /inference: (status, body)."""
        self.name = name
        self._post = post
        self._fields = {"language": language, "response_format": "json", "temperature": "0.0"}
        if initial_prompt:
            self._fields["prompt"] = initial_prompt
        self._on_close = on_close

    @classmethod
    def start(
        cls,
        binary: Path,
        model: Path,
        *,
        language: str = "ru",
        initial_prompt: str | None = None,
        threads: int = 4,
        gpu: bool = False,
        units: str = "ane",
        log: Path | None = None,
        ready_timeout_s: float = READY_TIMEOUT_S,
    ) -> WhisperCppTranscriber:
        """whisper-server ``binary`` on ``model`` (``ggml-<name>.bin``, its CoreML encoder
        beside it), the encoder on CoreML's ``units`` (the patched build's
        ``WHISPER_COREML_UNITS``), loaded and warmed."""
        for need in (binary, model, coreml_encoder(model)):
            if not need.exists():
                raise FileNotFoundError(f"{need}: see pepin.voice_live.whisper_cpp for the build")
        port = free_port()
        server = WhisperCppServer(
            server_command(binary, model, port=port, language=language, threads=threads, gpu=gpu),
            port=port,
            env={**os.environ, "WHISPER_COREML_UNITS": units},
            log=log,
            ready_timeout_s=ready_timeout_s,
        )
        name = model.stem.removeprefix("ggml-")
        transcriber = cls(
            server.post,
            language=language,
            initial_prompt=initial_prompt,
            name=f"whispercpp:{name}",
            on_close=server.close,
        )
        try:
            transcriber.transcribe(WARM_PCM)
        except BaseException:
            transcriber.close()
            raise
        return transcriber

    def transcribe(self, pcm: bytes) -> str:
        """The words in 16 kHz s16le mono ``pcm``."""
        body, content_type = multipart(self._fields, wav_bytes(pcm))
        status, answer = self._post(body, content_type)
        if status != 200:
            raise RuntimeError(f"whisper-server answered {status}: {answer[:200]!r}")
        return clean_transcript(str(json.loads(answer).get("text", "")))

    def close(self) -> None:
        """Stop the server (once)."""
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            on_close()
