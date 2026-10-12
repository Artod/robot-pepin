"""The cascade's other speech-to-text engines (pepin.voice_cascade.stt) on fake runtimes: what
each passes to its runtime, the words it hands on, the memory it frees, and the switch that
builds one by name (no model, no onnxruntime session, no GPU)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pepin.voice_cascade.config import DEFAULT_STT, TRANSCRIBER_ENGINES, CascadeConfig
from pepin.voice_cascade.stages import Transcriber
from pepin.voice_cascade.stt import build, runtime
from pepin.voice_cascade.transcriber import (
    Choice,
    SwitchingTranscriber,
    build_transcriber,
    choice_of,
    configured,
)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"
PCM = b"\x00\x40" * 1600  # 0.1 s at half scale


class OnnxModel:
    """onnx-asr's recogniser: remembers each call's samples and options."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int, dict[str, Any]]] = []

    def recognize(self, samples: Any, *, sample_rate: int, **options: Any) -> str:
        self.calls.append((len(samples), float(samples.max(initial=0.0)), sample_rate, options))
        return "  Сколько ты жрёшь\n токенов? "


class Onnx:
    """onnx-asr's ``load_model`` and onnxruntime's ``SessionOptions``, recorded."""

    def __init__(self) -> None:
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.models: list[OnnxModel] = []

    def load_model(self, model: str, **kwargs: Any) -> OnnxModel:
        self.loads.append((model, kwargs))
        self.models.append(OnnxModel())
        return self.models[-1]

    def runtime(self) -> tuple[Any, Any]:
        return self.load_model, SimpleNamespace(SessionOptions=SimpleNamespace)


class MlxModel:
    """An mlx-audio model: its dtype, its generate calls."""

    def __init__(self) -> None:
        self.dtype: str | None = None
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def set_dtype(self, dtype: str) -> None:
        self.dtype = dtype

    def parameters(self) -> dict[str, int]:
        return {"weight": 1}

    def generate(self, audio: Any, **options: Any) -> SimpleNamespace:
        self.calls.append((len(audio), options))
        return SimpleNamespace(text=" Алло, привет!  Пипин, привет! ")


class Mlx:
    """``mlx.core`` and mlx-audio's ``load_model``, recorded."""

    bfloat16 = "bf16"

    def __init__(self) -> None:
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.models: list[MlxModel] = []
        self.evaluated: list[Any] = []
        self.cleared = 0

    def array(self, samples: Any) -> Any:
        return samples

    def eval(self, parameters: Any) -> None:
        self.evaluated.append(parameters)

    def clear_cache(self) -> None:
        self.cleared += 1

    def load_model(self, repo: str, **kwargs: Any) -> MlxModel:
        self.loads.append((repo, kwargs))
        self.models.append(MlxModel())
        return self.models[-1]

    def runtime(self) -> tuple[Any, Any]:
        return self, self.load_model


@pytest.fixture
def onnx(monkeypatch: pytest.MonkeyPatch) -> Onnx:
    fake = Onnx()
    monkeypatch.setattr(runtime, "onnx_runtime", fake.runtime)
    return fake


@pytest.fixture
def mlx(monkeypatch: pytest.MonkeyPatch) -> Mlx:
    fake = Mlx()
    monkeypatch.setattr(runtime, "mlx_runtime", fake.runtime)
    return fake


class Named:
    """An engine that answers with its own name; remembers being closed."""

    def __init__(self, name: str) -> None:
        self.name, self.closed = name, False

    def transcribe(self, pcm: bytes) -> str:
        return self.name

    def close(self) -> None:
        self.closed = True


def now(work: Any) -> None:
    work()


def test_the_file_names_every_engine_with_its_device_and_picks_gigaam() -> None:
    config = CascadeConfig.load(CONFIG)
    assert configured(config).engine == "gigaam"
    assert set(config.stt) == set(DEFAULT_STT) and set(TRANSCRIBER_ENGINES) > set(config.stt)
    assert config.stt == DEFAULT_STT  # the file's blocks say what the defaults say
    gigaam = CascadeConfig.load(CONFIG, ["transcriber.engine=gigaam"])
    assert configured(gigaam) == Choice("gigaam", "gigaam-v3-e2e-rnnt")
    assert configured(gigaam).label == "gigaam:gigaam-v3-e2e-rnnt"
    small = CascadeConfig.load(CONFIG, ["transcriber.gigaam.quantization=int8"])
    assert small.stt["gigaam"].quantization == "int8"
    full = CascadeConfig.load(CONFIG, ["transcriber.parakeet.dtype=null"])
    assert full.stt["parakeet"].dtype is None
    for bad in (
        "transcriber.gigaam.device=gpu",
        "transcriber.voxtral.device=cpu",
        "transcriber.canary.quantization=fp16",
        "transcriber.parakeet.dtype=int4",
        "transcriber.voxtral.delay_ms=0",
        "transcriber.gigaam.threads=0",
    ):
        with pytest.raises(ValueError, match="transcriber"):
            CascadeConfig.load(CONFIG, [bad])


def test_a_control_value_names_a_new_engine_and_maybe_its_model() -> None:
    config = CascadeConfig.load(CONFIG)
    assert choice_of("parakeet", config) == Choice("parakeet", "mlx-community/parakeet-tdt-0.6b-v3")
    assert choice_of("gigaam:gigaam-v3-ctc", config) == Choice("gigaam", "gigaam-v3-ctc")
    assert choice_of({"engine": "voxtral"}, config) == Choice(
        "voxtral", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    )
    assert choice_of("canary", config) == Choice("canary", "nemo-canary-1b-v2")


def test_gigaam_runs_on_the_cpu_with_its_threads_and_hands_on_one_clean_line(onnx: Onnx) -> None:
    config = CascadeConfig.load(CONFIG, ["transcriber.gigaam.threads=2"])
    engine = build("gigaam", config.stt["gigaam"], "ru")
    ((model, kwargs),) = onnx.loads
    assert model == "gigaam-v3-e2e-rnnt" and kwargs["quantization"] is None
    assert kwargs["providers"] == ["CPUExecutionProvider"]
    assert kwargs["sess_options"].intra_op_num_threads == 2
    assert engine.transcribe(PCM) == "Сколько ты жрёшь токенов?"
    warm, call = onnx.models[0].calls
    assert warm[:2] == (8000, 0.0)  # warmed on 0.5 s of silence at the load
    assert call == (1600, 0.5, 16000, {})  # s16le as float samples; no language: Russian only


def test_canary_forces_the_language_both_ways(onnx: Onnx) -> None:
    config = CascadeConfig.load(CONFIG)
    engine = build("canary", config.stt["canary"], "ru")
    assert onnx.loads[0][0] == "nemo-canary-1b-v2" and onnx.loads[0][1]["quantization"] == "int8"
    engine.transcribe(PCM)
    assert onnx.models[0].calls[-1][3] == {"language": "ru", "target_language": "ru"}


def test_an_onnx_engine_closed_drops_its_model(onnx: Onnx) -> None:
    engine = build("gigaam", DEFAULT_STT["gigaam"], "ru")
    engine.close()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="closed"):
        engine.transcribe(PCM)


def test_parakeet_loads_lazily_casts_then_evaluates_and_frees_on_close(mlx: Mlx) -> None:
    engine = build("parakeet", DEFAULT_STT["parakeet"], "ru")
    ((repo, kwargs),) = mlx.loads
    assert repo == "mlx-community/parakeet-tdt-0.6b-v3" and kwargs == {"lazy": True}
    assert mlx.models[0].dtype == "bf16"  # cast before the weights are evaluated
    assert mlx.evaluated == [{"weight": 1}] and mlx.cleared == 1
    assert engine.transcribe(PCM) == "Алло, привет! Пипин, привет!"
    assert mlx.models[0].calls[-1] == (1600, {})
    engine.close()  # type: ignore[attr-defined]
    assert mlx.cleared == 2
    with pytest.raises(RuntimeError, match="closed"):
        engine.transcribe(PCM)


def test_voxtral_keeps_its_stored_dtype_and_passes_its_delay(mlx: Mlx) -> None:
    config = CascadeConfig.load(CONFIG, ["transcriber.voxtral.delay_ms=240"])
    engine = build("voxtral", config.stt["voxtral"], "ru")
    assert mlx.models[0].dtype is None
    engine.transcribe(PCM)
    assert mlx.models[0].calls[-1][1] == {"transcription_delay_ms": 240}
    build("voxtral", DEFAULT_STT["voxtral"], "ru").transcribe(PCM)
    assert mlx.models[1].calls[-1][1] == {}  # the model's own delay


def test_an_unknown_engine_is_refused() -> None:
    with pytest.raises(ValueError, match="vosk"):
        build("vosk", DEFAULT_STT["gigaam"], "ru")


def test_a_switch_builds_the_engine_by_name_and_frees_the_old_one(onnx: Onnx, mlx: Mlx) -> None:
    config = CascadeConfig.load(CONFIG)
    whisper = Named("whisper")
    wanted: list[Choice | None] = [choice_of("gigaam", config)]
    lines: list[str] = []
    sizes = iter([3072.0, 4096.0, 1536.0, 1536.0, 3072.0, 2048.0])
    built: list[Transcriber] = []

    def load(choice: Choice) -> Transcriber:
        built.append(build_transcriber(config, choice, Path(".")))
        return built[-1]

    switch = SwitchingTranscriber(
        whisper, configured(config), load, lambda: wanted[0],
        report=lines.append, spawn=now, footprint=lambda: next(sizes), settle_s=0,
    )  # fmt: skip
    switch.poll()
    assert switch.name == "gigaam:gigaam-v3-e2e-rnnt" and whisper.closed
    assert switch.transcribe(PCM) == "Сколько ты жрёшь токенов?"
    assert lines[0].startswith("transcriber: mlx:whisper-large-v3-turbo -> gigaam:")
    assert "footprint 3.00 -> 4.00 with both -> 1.50 GB" in lines[0]
    wanted[0] = choice_of("parakeet", config)
    switch.poll()
    assert switch.transcribe(PCM) == "Алло, привет! Пипин, привет!"
    with pytest.raises(RuntimeError, match="closed"):  # GigaAM's sessions were dropped
        built[0].transcribe(PCM)
    assert mlx.cleared == 1  # Parakeet's load; nothing freed on the GPU side yet
    assert "gigaam:gigaam-v3-e2e-rnnt -> parakeet:parakeet-tdt-0.6b-v3" in lines[1]
    assert "1.50 -> 3.00 with both -> 2.00 GB" in lines[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS's")
def test_the_footprint_is_this_process_in_megabytes() -> None:
    mb = runtime.footprint_mb()
    assert math.isfinite(mb) and 5 < mb < 64 * 1024
