"""A cloud voice: Gemini TTS streamed sentence by sentence, a local voice standing in for it.

Each sentence is one ``generate_content_stream`` call (response modality AUDIO, a prebuilt
voice) run on a helper thread that hands its PCM over a queue, so the voice waits at most
``timeout_s`` for each piece of audio. A call that fails or stays silent that long before its
first audio is left behind and the local fallback says the whole sentence; the cloud is then
left alone for ``RETRY_S``, so a dead network costs one wait, not one per sentence. A style
("шёпотом") travels in the part's ``speech_metadata.style``, which the model follows and does
not read aloud: an instruction written into the text itself is read out (2026-10-09, both
3.8 models: Whisper heard "Скажи грустно и устало." in front of the sentence; with the style
field it heard the sentence alone, whispered 11-13 dB quieter and 1.2-1.4x longer). The audio
is 24 kHz s16le mono, streamed in 50-100 pieces a sentence (first audio 0.4-0.7 s). Short
lines ("Да?", "Секунду.") are kept on disk by model, voice, style and text, so a line said
once is never paid for again. Each sentence
leaves a :class:`pepin.voice_cascade.stages.VoiceBill`: the model or the fallback that spoke,
the first audio after the call, the tokens and their price.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pepin.voice_cascade.config import ModelPrice
from pepin.voice_cascade.stages import Voice, VoiceBill
from pepin.voice_live.audio import Resampler

logger = logging.getLogger(__name__)

GEMINI_RATE = 24_000  # Hz: Gemini TTS's PCM
AUDIO_TOKENS_PER_S = 25  # Google's rate for audio output
RETRY_S = 30.0  # after a failed call the fallback speaks this long before the cloud is tried
CACHE_MAX_CHARS = 40  # lines this short are kept on disk
HTTP_TIMEOUT_S = 30.0  # the client's own limit on a whole call (a long sentence streams ~10 s)
_RATE_IN_MIME = re.compile(r"rate=(\d+)")


@dataclass(frozen=True)
class _Done:
    """The call ended; its usage metadata (None when it sent none)."""

    usage: Any


_Item = bytes | BaseException | _Done


def default_cache() -> Path:
    """``~/.cache/pepin/tts`` (beside the barge-in's VAD model)."""
    return Path.home() / ".cache" / "pepin" / "tts"


def audio_parts(chunk: Any) -> Iterator[tuple[bytes, str]]:
    """The inline audio of a streamed response chunk: (bytes, MIME type) per part."""
    for candidate in getattr(chunk, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            data = getattr(part, "inline_data", None)
            if data is not None and getattr(data, "data", None):
                yield bytes(data.data), str(getattr(data, "mime_type", "") or "")


def rate_of(mime: str) -> int:
    """The sample rate an ``audio/L16;codec=pcm;rate=24000`` MIME type names (24 kHz when
    none)."""
    match = _RATE_IN_MIME.search(mime)
    return int(match.group(1)) if match else GEMINI_RATE


class GeminiVoice:
    """:class:`pepin.voice_cascade.stages.Voice` (and ``Billed``) over Gemini TTS; the key from
    ``GEMINI_API_KEY``."""

    def __init__(
        self,
        model: str,
        *,
        voice: str = "Kore",
        price: ModelPrice,
        fallback: Voice | None,
        fallback_name: str = "local",
        timeout_s: float = 2.0,
        max_calls: int | None = None,
        cache_dir: Path | None = None,
        client: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``price``: USD per million text tokens in (``input``) and audio tokens out
        (``output``); ``fallback``: the local voice that speaks when the call fails or is
        slower than ``timeout_s`` (None: the sentence is lost); ``max_calls``: calls this voice
        may make in its life (None: no cap); ``cache_dir``: where short lines are kept (None:
        nowhere); ``client``: a ``genai.Client`` (built from the environment at load when
        None)."""
        self.model, self.voice, self.price = model, voice, price
        self.fallback, self.fallback_name = fallback, fallback_name
        self.timeout_s, self.max_calls = timeout_s, max_calls
        self.cache_dir = cache_dir / model / voice if cache_dir is not None else None
        self.client: Any = client
        self.clock = clock
        self.rate = GEMINI_RATE
        self.calls = 0
        self._failed_at: float | None = None
        self._bill: VoiceBill | None = None
        self._types: Any = None

    # -- the Voice ---------------------------------------------------------------------------

    def load(self) -> None:
        """The client (and its connection, opened by a free metadata call) and the fallback."""
        from google.genai import types

        self._types = types
        if self.client is None:
            from google import genai

            self.client = genai.Client(
                http_options=types.HttpOptions(timeout=int(HTTP_TIMEOUT_S * 1000))
            )
            try:  # the TLS handshake now, not in the first sentence
                self.client.models.get(model=self.model)
            except Exception as error:
                logger.warning("gemini voice: %s not described: %s", self.model, error)
        if self.fallback is not None:
            self.fallback.load()

    def bill(self) -> VoiceBill | None:
        """The last sentence's bill, taken once."""
        bill, self._bill = self._bill, None
        return bill

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        """``text`` in ``style`` from the cloud as it arrives; from the disk for a short line
        said before; from the fallback when the cloud fails, is slow or is resting."""
        cached = self._cached(text, style)
        if cached is not None:
            self._bill = VoiceBill(self.model, first_audio_ms=0)
            yield cached
            return
        reason = self._rest_reason()
        if reason is not None:
            yield from self._local(text, style, reason, calls=0)
            return
        self.calls += 1
        start = self.clock()
        items: queue.Queue[_Item] = queue.Queue()
        cancel = threading.Event()
        threading.Thread(
            target=self._call, args=(text, style, items, cancel), name="gemini-tts", daemon=True
        ).start()
        first_ms: int | None = None
        audio = bytearray()
        usage: Any = None
        error: BaseException | None = None
        try:
            while True:
                try:
                    item = items.get(timeout=self.timeout_s)
                except queue.Empty:
                    item = TimeoutError(f"no audio for {self.timeout_s:.1f} s")
                if isinstance(item, _Done):
                    usage = item.usage
                    break
                if isinstance(item, BaseException):
                    error = item
                    break
                if first_ms is None:
                    first_ms = round((self.clock() - start) * 1000)
                audio += item
                yield item
        finally:
            cancel.set()
            self._bill = self._priced(text, style, usage, len(audio), first_ms)
        if error is not None:
            self._failed_at = self.clock()
            why = f"{type(error).__name__}: {error}"[:200]
            if first_ms is None:  # nothing said yet: the local voice says it all
                logger.warning("gemini voice: %s; the fallback speaks", why)
                yield from self._local(text, style, why, calls=1)
            else:  # half said: the rest is lost (repeating the start would be worse)
                logger.warning("gemini voice: %s after %d bytes; sentence ended", why, len(audio))
            return
        self._keep(text, style, bytes(audio))

    # -- the call ----------------------------------------------------------------------------

    def request(self, text: str, style: str) -> dict[str, Any]:
        """The ``generate_content_stream`` arguments: the text as the transcript, the style in
        its speech metadata, audio out in the configured voice."""
        t = self._types
        metadata = t.SpeechMetadata(style=style) if style else None
        part = t.Part(text=text, speech_metadata=metadata)
        voice = t.VoiceConfig(prebuilt_voice_config=t.PrebuiltVoiceConfig(voice_name=self.voice))
        return {
            "model": self.model,
            "contents": [t.Content(role="user", parts=[part])],
            "config": t.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=t.SpeechConfig(voice_config=voice),
                automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            ),
        }

    def _call(
        self, text: str, style: str, items: queue.Queue[_Item], cancel: threading.Event
    ) -> None:
        """On the helper thread: the call's PCM into ``items`` (whole samples at 24 kHz), then
        its end or its error."""
        usage: Any = None
        carry = b""
        resampler: Resampler | None = None
        try:
            stream = self.client.models.generate_content_stream(**self.request(text, style))
            for chunk in stream:
                if cancel.is_set():
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
                    return
                usage = getattr(chunk, "usage_metadata", None) or usage
                for data, mime in audio_parts(chunk):
                    rate = rate_of(mime)
                    if rate != GEMINI_RATE and resampler is None:
                        resampler = Resampler(rate, GEMINI_RATE)
                    data = carry + data
                    whole = len(data) - len(data) % 2
                    carry = data[whole:]
                    pcm = data[:whole]
                    if resampler is not None:
                        pcm = resampler.process(pcm)
                    if pcm:
                        items.put(pcm)
            items.put(_Done(usage))
        except BaseException as error:
            items.put(error)

    def _priced(
        self, text: str, style: str, usage: Any, audio_bytes: int, first_ms: int | None
    ) -> VoiceBill:
        """A call's bill: the API's token counts, or (when it sent none) the audio's length at
        25 tokens a second and the text at ~4 characters a token."""
        text_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        audio_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        if not text_tokens:
            text_tokens = max(1, (len(text) + len(style)) // 4)
        if not audio_tokens:
            audio_tokens = round(audio_bytes / 2 / GEMINI_RATE * AUDIO_TOKENS_PER_S)
        usd = (text_tokens * self.price.input + audio_tokens * self.price.output) / 1e6
        return VoiceBill(self.model, first_ms, usd, text_tokens, audio_tokens, calls=1)

    # -- the fallback and the rest ------------------------------------------------------------

    def _rest_reason(self) -> str | None:
        """Why the cloud is not called now (None: it is)."""
        if self.max_calls is not None and self.calls >= self.max_calls:
            return f"the cap of {self.max_calls} calls"
        if self._failed_at is not None and self.clock() - self._failed_at < RETRY_S:
            return f"resting after a failure ({RETRY_S:.0f} s)"
        return None

    def _local(self, text: str, style: str, reason: str, *, calls: int) -> Iterator[bytes]:
        """The fallback's audio at this voice's rate; the bill keeps the cloud call's cost."""
        spent = self._bill if calls else None
        engine = self.fallback_name if self.fallback is not None else "none"
        self._bill = VoiceBill(
            engine,
            usd=spent.usd if spent else 0.0,
            text_tokens=spent.text_tokens if spent else 0,
            audio_tokens=spent.audio_tokens if spent else 0,
            calls=calls,
            fallback=reason,
        )
        if self.fallback is None:
            return
        resampler = Resampler(self.fallback.rate, self.rate)
        for pcm in self.fallback.stream(text, style):
            out = resampler.process(pcm)
            if out:
                yield out

    def _path(self, text: str, style: str) -> Path | None:
        if self.cache_dir is None or len(text) > CACHE_MAX_CHARS:
            return None
        key = hashlib.sha1(f"{style}\0{text}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}.pcm"

    def _cached(self, text: str, style: str) -> bytes | None:
        path = self._path(text, style)
        if path is None or not path.exists():
            return None
        return path.read_bytes() or None

    def _keep(self, text: str, style: str, pcm: bytes) -> None:
        path = self._path(text, style)
        if path is None or not pcm:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pcm)
            tmp.replace(path)
        except OSError as error:  # a full disk must not silence the robot
            logger.warning("gemini voice: the line was not kept: %s", error)
