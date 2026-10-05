"""The Live API as the conversation sees it: a session that takes audio, images and tool results
and yields :class:`LiveMessage` — and the one adapter that speaks google-genai.

Everything Gemini-specific is here (:func:`live_config`, :func:`convert`,
:class:`GeminiLive`); the conversation loop is written against :class:`LiveConnector` and is
tested with a scripted fake (:mod:`pepin.voice_live.fake`) that never opens a socket.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pepin.voice_live.budget import Usage
from pepin.voice_live.config import LiveConfig


@dataclass(frozen=True)
class ToolCall:
    """The model asks for a tool."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    """A tool's result going back; ``scheduling`` for a non-blocking call (WHEN_IDLE,
    INTERRUPT, SILENT), None for a blocking one."""

    id: str
    name: str
    response: dict[str, Any]
    scheduling: str | None = None


@dataclass(frozen=True)
class LiveMessage:
    """One server message, flattened to what the loop acts on."""

    audio: bytes = b""  # 24 kHz s16le mono
    input_text: str = ""  # a fragment of the transcription of what the user said
    output_text: str = ""  # a fragment of the transcription of what the model says
    tool_calls: tuple[ToolCall, ...] = ()
    cancelled: tuple[str, ...] = ()  # tool call ids the model no longer wants
    interrupted: bool = False  # the user spoke over the model: stop playing
    turn_complete: bool = False
    handle: str | None = None  # a new resumable session handle
    go_away_s: float | None = None  # the server will close the connection in this many seconds
    activity: str | None = None  # the server's VAD: "start" or "end" of the person's speech
    usage: Usage | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)  # for the log, audio left out


class LiveSession(Protocol):
    """An open Live session."""

    async def send_audio(self, pcm: bytes) -> None:
        """16 kHz s16le mono from the mic."""
        ...

    async def send_image(self, jpeg: bytes) -> None:
        """One picture (a tool's), as a video frame."""
        ...

    async def send_tool_responses(self, responses: Sequence[ToolResponse]) -> None:
        """Results of tool calls."""
        ...

    async def end_audio(self) -> None:
        """The mic pauses: the server ends the person's turn on what it has heard."""
        ...

    async def send_text(self, text: str) -> None:
        """A text turn (the robot's own report, not the person's words); the model answers it."""
        ...

    def messages(self) -> AsyncIterator[LiveMessage]:
        """Server messages until the connection closes."""
        ...


class LiveConnector(Protocol):
    """Opens sessions: ``handle`` resumes an earlier one; ``system`` is the session's system
    instruction (:func:`system_instruction`)."""

    def connect(
        self, handle: str | None, system: str
    ) -> contextlib.AbstractAsyncContextManager[LiveSession]:
        """An async context manager yielding the open session."""
        ...


SYSTEM = (
    "You are Pepin (Пепин), a small home robot: a wheeled cart with a camera head, living in"
    " Artem's flat. Someone just called you by name. Answer briefly, in one or two short spoken"
    " sentences, in the language the person speaks (usually Russian). Facts about yourself and"
    " the flat (where you are, what you see, your state) come ONLY from a tool call made now;"
    " never guess them. If you are only called by name, answer with a short 'Да?' or 'Слушаю'."
    " {places} {drives} Start a drive only when a person asks for one, never on your own."
    " When the person says goodbye (пока, всё, спасибо), say a short goodbye and call"
    " end_conversation. Speech that is clearly not meant for you, stay silent."
)
PLACES_KNOWN = (
    "The places go_to takes are {names} (exactly these names): you know them already, so when"
    " asked to go to one, call go_to at once; list_places is only for their distances."
)
PLACES_UNKNOWN = "list_places gives the names go_to accepts; never invent a place."
ROBOT_REPORT = "[robot]"  # a text turn that is the robot's own report, not the person
DRIVES_REPORTED_TWICE = (
    "A drive (go_to, go_to_pose) answers when the robot sets off (status driving): then say a"
    " short acknowledgement{acknowledge}. When it ends, a message beginning "
    + ROBOT_REPORT
    + " (your own body's report, not the person's words) says how: if it arrived, say a short"
    " line that you are there{arrived}; if not, say in one short sentence that you could not get"
    " there and why."
)
DRIVES_REPORTED_ONCE = (
    "A drive (go_to, go_to_pose) answers when it ends: if it arrived, say a short line that you"
    " are there{arrived}; if not, say in one short sentence that you could not get there and why."
)


def system_instruction(config: LiveConfig, places: Sequence[str] | None) -> str:
    """The session's instructions: who the robot is, the places it knows (``places``, the
    registry's book at the session's start; None when it could not be read), how a drive is
    reported and the persona's lines for it (config/voice_live.json ``persona``), in the
    model's own words."""

    def style(line: str) -> str:
        return f", in your own words, in the style of '{line}'" if line else ""

    known = PLACES_KNOWN.format(names=", ".join(places)) if places else PLACES_UNKNOWN
    drives = DRIVES_REPORTED_TWICE if config.report_drive_start else DRIVES_REPORTED_ONCE
    return SYSTEM.format(
        places=known,
        drives=drives.format(
            acknowledge=style(config.persona_acknowledge), arrived=style(config.persona_arrived)
        ),
    )


END_CONVERSATION = "end_conversation"


def declarations(registry_declarations: list[dict[str, Any]], moving: set[str]) -> list[Any]:
    """The tools as Live function declarations, plus ``end_conversation``; a tool that sets the
    robot in motion is NON_BLOCKING (the conversation goes on while it drives), the rest are
    BLOCKING (an answer the model waits for, a fraction of a second)."""
    from google.genai import types

    out = []
    for d in registry_declarations:
        behavior = types.Behavior.NON_BLOCKING if d["name"] in moving else types.Behavior.BLOCKING
        out.append(
            types.FunctionDeclaration(
                name=d["name"],
                description=d["description"],
                parameters=d.get("parameters"),  # pydantic makes the Schema, as in voice.py
                behavior=behavior,
            )
        )
    out.append(
        types.FunctionDeclaration(
            name=END_CONVERSATION,
            description="End the conversation after saying goodbye: the robot stops listening"
            " until it is called by name again.",
            behavior=types.Behavior.BLOCKING,
        )
    )
    return out


def live_config(config: LiveConfig, tools: list[Any], handle: str | None, system: str) -> Any:
    """The session's setup: audio out in ``config.voice``, both transcriptions (the person's
    hinted with ``config.transcription_languages``), automatic server VAD (its defaults), context
    compression, resumption (``handle`` resumes), the tools, ``system`` as the instruction."""
    from google.genai import types

    languages = list(config.transcription_languages) or None
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system,
        tools=[types.Tool(function_declarations=tools)],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.voice)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=languages),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=False)
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=config.compression_trigger_tokens,
            sliding_window=types.SlidingWindow(target_tokens=config.compression_target_tokens),
        ),
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )


def _seconds(text: str | None) -> float | None:
    """A protobuf duration ("12.5s") in seconds."""
    if not text:
        return None
    try:
        return float(str(text).rstrip("s"))
    except ValueError:
        return None


def usage_of(metadata: Any) -> Usage:
    """A usage report by modality: prompt (and tool-use prompt) tokens in, response and thought
    tokens out; a token of unknown modality is priced as text."""
    usage = Usage(reports=1)

    def split(details: Any, total: int | None) -> tuple[int, int]:
        audio = sum(
            int(d.token_count or 0)
            for d in details or []
            if str(getattr(d.modality, "value", d.modality)) == "AUDIO"
        )
        return audio, max(0, int(total or 0) - audio)

    usage.audio_in, usage.text_in = split(
        metadata.prompt_tokens_details, metadata.prompt_token_count
    )
    usage.text_in += int(metadata.tool_use_prompt_token_count or 0)
    usage.audio_out, usage.text_out = split(
        metadata.response_tokens_details, metadata.response_token_count
    )
    usage.text_out += int(metadata.thoughts_token_count or 0)
    return usage


def convert(message: Any) -> LiveMessage:
    """A google-genai ``LiveServerMessage`` as a :class:`LiveMessage`."""
    content = message.server_content
    audio = b""
    input_text = output_text = ""
    interrupted = turn_complete = False
    if content is not None:
        for part in (content.model_turn.parts or []) if content.model_turn else []:
            if part.inline_data is not None and part.inline_data.data:
                audio += part.inline_data.data
        input_text = content.input_transcription.text or "" if content.input_transcription else ""
        output_text = (
            content.output_transcription.text or "" if content.output_transcription else ""
        )
        interrupted = bool(content.interrupted)
        turn_complete = bool(content.turn_complete)
    calls: tuple[ToolCall, ...] = ()
    if message.tool_call is not None:
        calls = tuple(
            ToolCall(str(c.id or ""), str(c.name or ""), dict(c.args or {}))
            for c in message.tool_call.function_calls or []
        )
    cancelled: tuple[str, ...] = ()
    if message.tool_call_cancellation is not None:
        cancelled = tuple(message.tool_call_cancellation.ids or ())
    activity: str | None = None
    if message.voice_activity is not None:
        kind = str(getattr(message.voice_activity.voice_activity_type, "value", ""))
        activity = {"ACTIVITY_START": "start", "ACTIVITY_END": "end"}.get(kind)
    update = message.session_resumption_update
    handle = update.new_handle if update is not None and update.resumable else None
    raw = message.model_dump(mode="json", exclude_none=True)
    with contextlib.suppress(KeyError, TypeError, AttributeError):
        for part in raw["server_content"]["model_turn"]["parts"]:
            if "inline_data" in part:
                part["inline_data"] = {"bytes": len(audio)}
    return LiveMessage(
        audio=audio,
        input_text=input_text,
        output_text=output_text,
        tool_calls=calls,
        cancelled=cancelled,
        interrupted=interrupted,
        turn_complete=turn_complete,
        handle=handle or None,
        go_away_s=_seconds(message.go_away.time_left) if message.go_away is not None else None,
        activity=activity,
        usage=usage_of(message.usage_metadata) if message.usage_metadata is not None else None,
        raw=raw,
    )


class _GeminiSession:
    """:class:`LiveSession` over google-genai's ``AsyncSession``."""

    def __init__(self, session: Any) -> None:
        from google.genai import types

        self._session = session
        self._types = types

    async def send_audio(self, pcm: bytes) -> None:
        blob = self._types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
        await self._session.send_realtime_input(audio=blob)

    async def send_image(self, jpeg: bytes) -> None:
        blob = self._types.Blob(data=jpeg, mime_type="image/jpeg")
        await self._session.send_realtime_input(video=blob)

    async def send_tool_responses(self, responses: Sequence[ToolResponse]) -> None:
        t = self._types
        await self._session.send_tool_response(
            function_responses=[
                t.FunctionResponse(
                    id=r.id,
                    name=r.name,
                    response=r.response,
                    scheduling=t.FunctionResponseScheduling(r.scheduling) if r.scheduling else None,
                )
                for r in responses
            ]
        )

    async def end_audio(self) -> None:
        await self._session.send_realtime_input(audio_stream_end=True)

    async def send_text(self, text: str) -> None:
        t = self._types
        turn = t.Content(role="user", parts=[t.Part(text=text)])
        await self._session.send_client_content(turns=turn, turn_complete=True)

    async def messages(self) -> AsyncIterator[LiveMessage]:
        while True:  # receive() ends at each turn's end; the session goes on
            async for message in self._session.receive():
                yield convert(message)


class GeminiLive:
    """:class:`LiveConnector` for the Gemini API (the key from ``GEMINI_API_KEY``)."""

    def __init__(self, config: LiveConfig, tools: list[Any]) -> None:
        """``tools``: function declarations (:func:`declarations`)."""
        from google import genai

        self.config = config
        self.tools = tools
        self._client = genai.Client()

    @contextlib.asynccontextmanager
    async def connect(self, handle: str | None, system: str) -> AsyncIterator[LiveSession]:
        """Open (or resume) a session."""
        setup = live_config(self.config, self.tools, handle, system)
        async with self._client.aio.live.connect(model=self.config.model, config=setup) as s:
            yield _GeminiSession(s)
