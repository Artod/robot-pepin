"""The talker on an OpenAI-compatible Chat Completions endpoint (Groq): the same turn, tools
and history rules as the Claude brain (:mod:`pepin.voice_cascade.brain`).

The tools are the registry's in the function format (:func:`pepin.tools.schemas.openai_tools`,
narrowed to ``brain.tools``: the free tier limits tokens per minute, so the prompt stays
compact). A round streams text deltas into sentences and tool-call deltas into calls; the calls
run in parallel and their results go back as ``tool`` messages; a picture a tool returns follows
in a user message as a ``data:`` URL, and a model that refuses pictures gets the call answered in
words instead (once noticed, for the rest of the run). The client never retries: a 429 is
printed as one line and fails the turn (the loop says its error line). Some models emit
reasoning; the preset's ``extra_body`` asks the endpoint to leave it out.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

from pepin.tools.registry import Image, Result, render
from pepin.voice_cascade.brain import (
    END_TOOL,
    NOT_AWAITED,
    PLACES_KNOWN,
    PLACES_UNKNOWN,
)
from pepin.voice_cascade.config import BrainConfig, ModelPrice
from pepin.voice_cascade.stages import (
    FOLLOW_UP,
    Emit,
    FirstToken,
    Played,
    Sentence,
    ToolDone,
    ToolExecutor,
    ToolUse,
    Usage,
)
from pepin.voice_cascade.text import SentenceSplitter

Message = dict[str, Any]
END_FUNCTION = {
    "type": "function",
    "function": {
        "name": END_TOOL["name"],
        "description": END_TOOL["description"],
        "parameters": {"type": "object", "properties": {}},
    },
}
PICTURE_GONE = "(the picture the tool took was shown here; it is not kept)"
NO_PICTURES = "the picture could not be shown: this model takes no images"
FIRST_SENTENCE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.DOTALL)


def first_sentence(text: str, limit: int) -> str:
    """The first sentence of ``text`` (at most ``limit`` characters, whole words)."""
    match = FIRST_SENTENCE.match(" ".join(text.split()))
    sentence = match.group(1) if match else " ".join(text.split())
    if len(sentence) <= limit:
        return sentence
    return sentence[:limit].rsplit(" ", 1)[0] + "..."


def compact(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Function-format tools with each description cut to its first sentence and each
    argument's to its first clause: the same tools in about half the tokens."""
    out = []
    for tool in tools:
        function = dict(tool["function"])
        function["description"] = first_sentence(function.get("description", ""), 200)
        parameters = dict(function.get("parameters") or {})
        properties = {}
        for name, schema in (parameters.get("properties") or {}).items():
            schema = dict(schema)
            if "description" in schema:
                clause = re.split(r"[;:(]", schema["description"], maxsplit=1)[0]
                schema["description"] = first_sentence(clause, 90)
            properties[name] = schema
        if parameters:
            parameters["properties"] = properties
            function["parameters"] = parameters
        out.append({**tool, "function": function})
    return out


def chat_usage(usage: Any, price: ModelPrice, stop: str | None, ms: int) -> Usage:
    """An OpenAI-style usage report as :class:`Usage`, priced (cached tokens, when reported, at
    the cache rate)."""
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    out = int(getattr(usage, "completion_tokens", 0) or 0)
    fresh = max(0, prompt - cached)
    usd = (fresh * price.input + cached * price.cache_read + out * price.output) / 1e6
    return Usage(fresh, cached, 0, out, usd, stop, ms)


class ChatBrain:
    """:class:`pepin.voice_cascade.stages.Brain` on Chat Completions (``openai`` SDK, async,
    streaming); the endpoint and the key's variable come from the preset."""

    def __init__(
        self,
        config: BrainConfig,
        *,
        persona: str,
        tools: list[dict[str, Any]],
        price: ModelPrice,
        client: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``tools``: function-format tool definitions (``end_conversation`` is added);
        ``client``: an ``AsyncOpenAI`` (built from the preset when None)."""
        if client is None:
            import openai

            client = openai.AsyncOpenAI(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_env or "", ""),
                max_retries=0,
                timeout=config.timeout_s,
            )
        self.config, self.persona, self.price = config, persona, price
        self.client: Any = client
        self.clock = clock
        offered = compact(tools) if config.compact_tools else tools
        self.tools = [*offered, END_FUNCTION]
        self.pictures = True  # the model takes images (until it refuses one)
        self.places_line = PLACES_UNKNOWN
        self.history: list[Message] = []
        self._user: Message | None = None
        self._rounds: list[list[Message]] = []  # each: the assistant message, then its results
        self._said: list[Sentence] = []
        self._acted = False

    # -- the request -----------------------------------------------------------------------

    def messages(self) -> list[Message]:
        """The system message, the history, the open turn."""
        system = {"role": "system", "content": f"{self.persona}\n\n{self.places_line}"}
        turn = [self._user] if self._user is not None else []
        return [system, *self.history, *turn, *(m for r in self._rounds for m in r)]

    def request(self, *, first: bool, final: bool) -> dict[str, Any]:
        """The keyword arguments of one streamed call."""
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": self.messages(),
            "tools": self.tools,
            "tool_choice": "none" if final else cfg.tool_choice if first else "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": cfg.max_tokens,
        }
        if cfg.reasoning_effort is not None:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.extra_body:
            kwargs["extra_body"] = dict(cfg.extra_body)
        return kwargs

    # -- the conversation ------------------------------------------------------------------

    def begin(self, places: list[str] | None) -> None:
        """A new conversation (the history forgotten) knowing ``places``."""
        names = ", ".join(places) if places else ""
        self.places_line = PLACES_KNOWN.format(names=names) if names else PLACES_UNKNOWN
        self.history = []
        self._reset()

    def _reset(self) -> None:
        self._user, self._rounds, self._said, self._acted = None, [], [], False

    @property
    def acted(self) -> bool:
        """The open turn has run a tool."""
        return self._acted

    async def prewarm(self) -> Usage | None:
        """Open the connection with a free request (the model list): a token-billed warm-up
        would eat the free tier's tokens per minute. (Not the model's own card: Groq answers
        404 to the escaped slash of "openai/gpt-oss-120b".)"""
        if not self.config.prewarm:
            return None
        t0 = self.clock()
        await self.client.models.list()
        return Usage(ms=round((self.clock() - t0) * 1000))

    async def turn(self, text: str, tools: ToolExecutor, emit: Emit) -> None:
        """One turn: calls until one ends without tool calls (at most ``max_tool_rounds``)."""
        self._reset()
        self._user = {"role": "user", "content": text}
        first = True
        for round_index in range(self.config.max_tool_rounds + 1):
            final = round_index == self.config.max_tool_rounds
            splitter = SentenceSplitter()
            words: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            usage: Any = None
            stop: str | None = None
            t0 = self.clock()

            def say(sentence: str, index: int = round_index) -> None:
                said = Sentence(sentence, index)
                self._said.append(said)
                emit(said)

            stream = await self._create(self.request(first=round_index == 0, final=final))
            try:
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None) or usage
                    groq = getattr(chunk, "x_groq", None)
                    if groq is not None and getattr(groq, "usage", None) is not None:
                        usage = groq.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    stop = choice.finish_reason or stop
                    delta = choice.delta
                    if delta.content:
                        if first:
                            first = False
                            emit(FirstToken())
                        words.append(delta.content)
                        for sentence in splitter.feed(delta.content):
                            say(sentence)
                    for part in delta.tool_calls or []:
                        if not calls:
                            rest = splitter.flush()  # the words before a call end here
                            if rest:
                                say(rest)
                        slot = calls.setdefault(part.index, {"id": "", "name": "", "args": ""})
                        if part.id:
                            slot["id"] = part.id
                        if part.function is not None:
                            slot["name"] += part.function.name or ""
                            slot["args"] += part.function.arguments or ""
            finally:
                await stream.close()
            rest = splitter.flush()
            if rest:
                say(rest)
            emit(chat_usage(usage, self.price, stop, round((self.clock() - t0) * 1000)))
            spoken = "".join(words).strip()
            if not calls:
                if spoken:
                    self._rounds.append([{"role": "assistant", "content": spoken}])
                return
            ordered = [calls[i] for i in sorted(calls)]
            message: Message = {
                "role": "assistant",
                "content": spoken or None,
                "tool_calls": [
                    {
                        "id": c["id"] or f"call_{round_index}_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                    }
                    for i, c in enumerate(ordered)
                ],
            }
            round_messages = [message]
            self._rounds.append(round_messages)
            self._acted = True
            for call in message["tool_calls"]:
                emit(ToolUse(call["id"], call["function"]["name"], self._args(call)))
            done = await asyncio.gather(
                *(self._run(tools, call, emit) for call in message["tool_calls"])
            )
            round_messages.extend(answer for answer, _, _ in done)
            pictures = [image for _, images, _ in done for image in images]
            if pictures:
                round_messages.append(self._picture_message(pictures))
            if not any(follow for _, _, follow in done):
                return

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """One streamed call: a 429 is printed; a request refused while it carries pictures is
        sent again without them (the calls answered in words), for the rest of the run."""
        import openai

        try:
            return await self.client.chat.completions.create(**kwargs)
        except openai.RateLimitError as error:
            after = getattr(getattr(error, "response", None), "headers", {}).get("retry-after")
            print(f"  !! {self.config.model}: 429 rate limit (retry after {after} s)", flush=True)
            raise
        except openai.BadRequestError:
            if not self._has_pictures():
                raise
            # gpt-oss-120b on Groq: "messages[4].content must be a string" for a picture
            self.pictures = False
            self._drop_pictures()
            kwargs["messages"] = self.messages()
            return await self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _args(call: Message) -> dict[str, Any]:
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except ValueError:
            return {}
        return args if isinstance(args, dict) else {}

    async def _run(
        self, tools: ToolExecutor, call: Message, emit: Emit
    ) -> tuple[Message, list[Image], bool]:
        """One call: its ``tool`` message, its pictures, and whether the model should speak
        after it."""
        name = call["function"]["name"]
        t0 = self.clock()
        try:
            json.loads(call["function"]["arguments"] or "{}")
            result: Result = dict(await tools(name, self._args(call)))
        except ValueError:
            result = {"ok": False, "why": "the arguments were not valid JSON"}
        follow = bool(result.pop(FOLLOW_UP, True))
        text, images = render(result)
        if images and not self.pictures:
            text, images = f"{text} ({NO_PICTURES})", []
        emit(ToolDone(call["id"], name, text, round((self.clock() - t0) * 1000)))
        return {"role": "tool", "tool_call_id": call["id"], "content": text}, images, follow

    @staticmethod
    def _picture_message(images: list[Image]) -> Message:
        parts: list[dict[str, Any]] = [{"type": "text", "text": "The picture the tool took:"}]
        for image in images:
            data = base64.standard_b64encode(image.data).decode("ascii")
            url = f"data:{image.mime};base64,{data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": parts}

    def _has_pictures(self) -> bool:
        return any(
            m["role"] == "user" and isinstance(m.get("content"), list)
            for r in self._rounds
            for m in r
        )

    def _drop_pictures(self) -> None:
        for round_messages in self._rounds:
            kept = []
            for m in round_messages:
                if m["role"] == "user" and isinstance(m.get("content"), list):
                    continue
                if m["role"] == "tool" and "attached" in str(m.get("content")):
                    m = {**m, "content": f"{m['content']} ({NO_PICTURES})"}
                kept.append(m)
            round_messages[:] = kept

    @staticmethod
    def _without_pictures(messages: list[Message]) -> list[Message]:
        """The messages with every picture replaced by a line of text: a picture is for the
        turn that asked for it; kept in the history it costs input every turn and some models
        (qwen3.8-27b on Groq, 2026-10-09) answer in two tokens once one sits behind them."""
        out: list[Message] = []
        for m in messages:
            if m["role"] == "user" and isinstance(m.get("content"), list):
                out.append({"role": "user", "content": PICTURE_GONE})
            else:
                out.append(m)
        return out

    def commit(self) -> None:
        """The turn ran to its end: it joins the history, its pictures as a line of text."""
        if self._user is not None:
            rounds = [m for r in self._rounds for m in r]
            self.history += [self._user, *self._without_pictures(rounds)]
        self._reset()

    def abandon(self) -> None:
        """Drop the open turn."""
        self._reset()

    def interrupt(self, played: Played) -> None:
        """As the Claude brain: rounds with tool calls stay (answered "interrupted" when their
        results never came); the round being said keeps only the words heard."""
        if self._user is None:
            return
        kept: list[Message] = [self._user]
        tool_rounds = 0
        for round_messages in self._rounds:
            head = round_messages[0]
            if not head.get("tool_calls"):
                break
            kept.append(head)
            answered = {m["tool_call_id"] for m in round_messages if m["role"] == "tool"}
            kept += round_messages[1:]
            kept += [
                {"role": "tool", "tool_call_id": c["id"], "content": NOT_AWAITED}
                for c in head["tool_calls"]
                if c["id"] not in answered
            ]
            tool_rounds += 1
        heard = [s.text for s in self._said[: played.sentences] if s.round == tool_rounds]
        cut = self._said[played.sentences] if played.sentences < len(self._said) else None
        if played.partial and cut is not None and cut.round == tool_rounds:
            heard.append(played.partial)
        if heard:
            # an ellipsis, not the bracketed marker: chat models copy a marker into replies
            kept.append({"role": "assistant", "content": " ".join(heard) + " …"})
        self.history += self._without_pictures(kept)
        self._reset()
