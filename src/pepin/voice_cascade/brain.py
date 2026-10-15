"""The talker on the Claude API: one streamed model call per round, sentences out as they end,
the robot's tools in between, the prompt cache and the price of every call.

The request is laid out for the cache: the tools (the registry's, in its fixed order), then the
persona as the first system block with a cache breakpoint (shared by every conversation), then
the places book as a second system block, then the conversation, cached as it grows (the
request's top-level breakpoint). A conversation's history keeps the model's own content blocks,
so a later round reads exactly what it wrote; a turn the person cut short keeps only what they
heard (:meth:`ClaudeBrain.interrupt`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from pepin.tools.registry import Registry, Result, render
from pepin.tools.schemas import anthropic_tool_result, anthropic_tools
from pepin.voice_cascade.config import BrainConfig, ModelPrice
from pepin.voice_cascade.stages import (
    CUT,
    FOLLOW_UP,
    Ahead,
    Emit,
    FirstToken,
    Played,
    Sentence,
    ToolDone,
    ToolExecutor,
    ToolUse,
    Usage,
    text_tokens,
)
from pepin.voice_cascade.text import SentenceSplitter

logger = logging.getLogger(__name__)

END_CONVERSATION = "end_conversation"
END_TOOL = {
    "name": END_CONVERSATION,
    "description": "End the conversation after saying goodbye: the robot stops listening until"
    " it is called by name again.",
    "input_schema": {"type": "object", "properties": {}},
}
INTERRUPTED = " …"  # an ellipsis: a bracketed marker got copied into replies by chat models
NOT_AWAITED = "interrupted by the person before the result came back"
PLACES_KNOWN = (
    "The places go_to takes are: {names} (exactly these names). You know them already: when"
    " asked to go to one, call go_to at once."
)
PLACES_UNKNOWN = "The places book could not be read: list_places gives the names go_to accepts."
WARM_UP = "(warm-up)"

Message = dict[str, Any]


def tool_definitions(registry: Registry, allowed: set[str]) -> list[dict[str, Any]]:
    """The registry's tools the brain may call (in registration order, so the cached prefix
    never changes between runs), plus ``end_conversation``."""
    return [t for t in anthropic_tools(registry) if t["name"] in allowed] + [END_TOOL]


def usage_of(usage: Any, price: ModelPrice, stop: str | None, ms: int) -> Usage:
    """An API usage report as :class:`Usage`, priced."""
    fresh = int(getattr(usage, "input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    usd = (
        fresh * price.input
        + read * price.cache_read
        + write * price.cache_write
        + out * price.output
    ) / 1e6
    return Usage(fresh, read, write, out, usd, stop, ms)


def cut_usage(started: Any, chars: int, last_prompt: int, price: ModelPrice, ms: int) -> Usage:
    """A call cut mid-stream, priced as an estimate: its prompt as ``message_start`` reported it
    (or, cut before that, the last call's prompt read from the cache), its output as the larger
    of what the stream reported and the text it streamed."""
    if started is None:
        return usage_of(
            SimpleNamespace(cache_read_input_tokens=last_prompt, output_tokens=text_tokens(chars)),
            price,
            CUT,
            ms,
        )
    out = max(int(getattr(started, "output_tokens", 0) or 0), text_tokens(chars))
    return usage_of(
        SimpleNamespace(
            input_tokens=getattr(started, "input_tokens", 0),
            cache_read_input_tokens=getattr(started, "cache_read_input_tokens", 0),
            cache_creation_input_tokens=getattr(started, "cache_creation_input_tokens", 0),
            output_tokens=out,
        ),
        price,
        CUT,
        ms,
    )


def prompt_tokens(usage: Usage) -> int:
    """Every input token of a call (fresh, read and written)."""
    return usage.input + usage.cache_read + usage.cache_write


class ClaudeBrain:
    """:class:`pepin.voice_cascade.stages.Brain` on the Messages API (``anthropic`` SDK, async,
    streaming). The key comes from ``ANTHROPIC_API_KEY`` in the environment."""

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
        """``tools``: Messages API tool definitions (:func:`tool_definitions`); ``client``: an
        ``AsyncAnthropic`` (built from the environment when None)."""
        if client is None:
            import anthropic  # lazy: the unit tier runs the brain over a fake client

            client = anthropic.AsyncAnthropic(timeout=config.timeout_s, max_retries=1)
        self.config = config
        self.persona = persona
        self.tools = tools
        self.price = price
        self.client: Any = client
        self.clock = clock
        self.places_line = PLACES_UNKNOWN
        self.history: list[Message] = []
        self._turn: list[Message] = []
        self._said: list[Sentence] = []
        self._acted = False
        self.last_prompt = 0  # the last call's input tokens (a call cut early is billed so)

    # -- the request -----------------------------------------------------------------------

    def system(self) -> list[dict[str, Any]]:
        """The persona (cached across conversations), then the places book."""
        persona: dict[str, Any] = {"type": "text", "text": self.persona}
        if self.config.cache:
            persona["cache_control"] = {"type": "ephemeral"}
        return [persona, {"type": "text", "text": self.places_line}]

    def request(self, messages: list[Message], *, final: bool = False) -> dict[str, Any]:
        """The keyword arguments of one call; ``final``: tools may not be called (the round
        limit is reached)."""
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": self.system(),
            "tools": self.tools,
            "messages": messages,
        }
        if cfg.thinking is not None:
            kwargs["thinking"] = {"type": cfg.thinking}
        if cfg.effort is not None:
            kwargs["output_config"] = {"effort": cfg.effort}
        if cfg.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        if final:
            kwargs["tool_choice"] = {"type": "none"}
        return kwargs

    # -- the conversation ------------------------------------------------------------------

    def begin(self, places: list[str] | None) -> None:
        """A new conversation (the history forgotten) knowing ``places``."""
        names = ", ".join(places) if places else ""
        self.places_line = PLACES_KNOWN.format(names=names) if names else PLACES_UNKNOWN
        self.history = []
        self._turn, self._said, self._acted = [], [], False

    @property
    def acted(self) -> bool:
        """The open turn has run a tool."""
        return self._acted

    async def prewarm(self) -> Usage | None:
        """A ``max_tokens: 0`` call with the conversation's tools, system and thinking: it
        writes the cache up to the persona's breakpoint, so the first turn reads it."""
        if not (self.config.cache and self.config.prewarm):
            return None
        kwargs = self.request([{"role": "user", "content": WARM_UP}])
        kwargs["max_tokens"] = 0
        kwargs.pop("cache_control", None)  # the warm-up's own message is not worth a write
        t0 = self.clock()
        message = await self.client.messages.create(**kwargs)
        ms = round((self.clock() - t0) * 1000)
        usage = usage_of(message.usage, self.price, getattr(message, "stop_reason", None), ms)
        self.last_prompt = prompt_tokens(usage)
        return usage

    async def turn(self, text: str, tools: ToolExecutor, emit: Emit) -> None:
        """One turn: model calls until one ends without tools (at most ``max_tool_rounds``
        rounds of tools)."""
        self._turn = [{"role": "user", "content": text}]
        self._said, self._acted = [], False
        first = True
        for round_index in range(self.config.max_tool_rounds + 1):
            final = round_index == self.config.max_tool_rounds
            kwargs = self.request(self.history + self._turn, final=final)
            splitter = SentenceSplitter()
            t0 = self.clock()

            def say(sentence: str, index: int = round_index) -> None:
                said = Sentence(sentence, index)
                self._said.append(said)
                emit(said)

            started: Any = None  # the call's usage so far, as the stream reports it
            chars = 0
            try:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        kind = getattr(event, "type", "")
                        if kind == "message_start":
                            started = event.message.usage
                        elif kind == "text":
                            chars += len(event.text)
                            if first:
                                first = False
                                emit(FirstToken())
                            for sentence in splitter.feed(event.text):
                                say(sentence)
                            for early in splitter.ahead():
                                emit(Ahead(early, round_index))
                        elif kind == "content_block_stop":
                            rest = splitter.flush()  # a text block ends before a tool call
                            if rest:
                                say(rest)
                    message: Any = await stream.get_final_message()
            except asyncio.CancelledError:
                ms = round((self.clock() - t0) * 1000)
                emit(cut_usage(started, chars, self.last_prompt, self.price, ms))
                raise
            rest = splitter.flush()
            if rest:
                say(rest)
            ms = round((self.clock() - t0) * 1000)
            usage = usage_of(message.usage, self.price, message.stop_reason, ms)
            self.last_prompt = prompt_tokens(usage)
            emit(usage)
            uses = [block for block in message.content if block.type == "tool_use"]
            if message.stop_reason != "tool_use" or not uses:
                # a call cut by max_tokens leaves no tool call to answer; an empty reply (speech
                # not meant for the robot) leaves no message: the API refuses an empty one
                kept = [block for block in message.content if block.type != "tool_use"]
                if kept:
                    self._turn.append({"role": "assistant", "content": kept})
                return
            self._turn.append({"role": "assistant", "content": message.content})
            self._acted = True
            for use in uses:
                emit(ToolUse(use.id, use.name, dict(use.input or {})))
            done = await asyncio.gather(
                *(self._run(tools, use.id, use.name, dict(use.input or {}), emit) for use in uses)
            )
            self._turn.append({"role": "user", "content": [block for block, _ in done]})
            if not any(follow for _, follow in done):
                return  # every result said there is nothing to add: no call to hear it

    async def _run(
        self, tools: ToolExecutor, use_id: str, name: str, args: dict[str, Any], emit: Emit
    ) -> tuple[dict[str, Any], bool]:
        """One tool: its result block, and whether the model should speak after it (a result
        with ``follow_up: false`` needs no answer: the robot said its line already)."""
        t0 = self.clock()
        result: Result = dict(await tools(name, args))
        follow = bool(result.pop(FOLLOW_UP, True))
        text, _ = render(result)
        emit(ToolDone(use_id, name, text, round((self.clock() - t0) * 1000)))
        return anthropic_tool_result(use_id, result), follow

    def commit(self) -> None:
        """The turn ran to its end: it joins the history."""
        self.history += self._turn
        self._turn, self._said, self._acted = [], [], False

    def abandon(self) -> None:
        """Drop the open turn."""
        self._turn, self._said, self._acted = [], [], False

    def interrupt(self, played: Played) -> None:
        """The person cut in. Rounds that called tools stay as they were, with their results
        (or "interrupted" for results that never came back): the tools did act. The round being
        said becomes a plain text reply holding only what was heard, marked as cut off, and is
        left out when nothing of it was heard."""
        if not self._turn:
            return
        kept: list[Message] = [self._turn[0]]
        rounds = self._turn[1:]
        tool_rounds = 0
        for index in range(0, len(rounds), 2):
            message = rounds[index]
            if not any(getattr(b, "type", None) == "tool_use" for b in message["content"]):
                break  # the text-only round being said: replaced below
            kept.append(message)
            has_results = index + 1 < len(rounds)
            kept.append(rounds[index + 1] if has_results else self._not_awaited(message))
            tool_rounds += 1
        heard = [s.text for s in self._said[: played.sentences] if s.round == tool_rounds]
        cut = self._said[played.sentences] if played.sentences < len(self._said) else None
        if played.partial and cut is not None and cut.round == tool_rounds:
            heard.append(played.partial)
        if heard:
            text = " ".join(heard) + INTERRUPTED
            kept.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        self.history += kept
        self._turn, self._said, self._acted = [], [], False

    @staticmethod
    def _not_awaited(message: Message) -> Message:
        """Results for the tool calls of ``message`` that never came back."""
        uses = [b for b in message["content"] if getattr(b, "type", None) == "tool_use"]
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": u.id,
                    "content": NOT_AWAITED,
                    "is_error": True,
                }
                for u in uses
            ],
        }


def said_text(sentences: Sequence[Sentence]) -> str:
    """The sentences of a reply as one line."""
    return " ".join(s.text for s in sentences)
