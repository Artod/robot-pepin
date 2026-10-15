"""The talker on the Gemini API (google-genai): the same turn, tools and history rules as the
Claude brain (:mod:`pepin.voice_cascade.brain`), on ``generate_content_stream``.

The persona and the places book are the system instruction; the tools are the registry's
function declarations (as the Live loop renders them); thinking is the lowest each model takes
(``thinking_budget`` or ``thinking_level`` from the preset). A round's streamed parts are kept
as they came (Gemini 3 binds a thought signature to a function call part and wants it back);
its function calls run in parallel; their results go back as function-response parts, and a
picture a tool returns travels as an inline image part in the same turn.

The static prefix (system instruction and tools, places included) lives in an explicit context
cache (``brain.cache``): created at the prewarm with a one-hour TTL, the TTL renewed by a turn
that finds it under ten minutes, a new cache made when the places change. Gemini's implicit
cache needs 4096 tokens and the prefix is ~3.3k, so it hit once in a live session; the explicit
one is read on every call (2026-10-09, Flash-Lite: 3251 of 3258 prompt tokens cached, function
calls unchanged). A call with a cache may not set the system, the tools or the tool config, so
the last round of a turn (tools forbidden) goes without it. The usage's
``cached_content_token_count`` is the cache read; storing it costs ~1 USD per million tokens an
hour (~0.3 US cents an hour here), and the cache is deleted at the end of the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

from pepin.tools.registry import Image, Result, render
from pepin.voice_cascade.brain import (
    END_TOOL,
    INTERRUPTED,
    NOT_AWAITED,
    PLACES_KNOWN,
    PLACES_UNKNOWN,
    WARM_UP,
)
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

# Gemini at its lowest thinking says "Сейчас гляну." and ends the reply without the call it
# announced (2026-10-09: 4 of 4 on "посмотри на меня", Flash-Lite); asked to call first, it calls
# (4 of 4), and the loop says the filler itself (pepin.voice_cascade.loop.SLOW_TOOLS).
CACHE_TTL_S = 3600
CACHE_RENEW_S = 600  # a turn renews the TTL when less than this is left
CACHE_STORAGE_USD_PER_MTOK_HOUR = 1.0
CALL_FIRST = (
    "# Tools on this model\n\nWhen a request needs a tool, call the tool at once, as the first"
    " thing in your reply, without any words before it: the robot says a short filler by itself"
    " while the tool works. Speak after the result comes back."
)


def gemini_usage(metadata: Any, price: ModelPrice, stop: str | None, ms: int) -> Usage:
    """A ``usage_metadata`` as :class:`Usage`: the prompt split into cached and fresh, the
    answer and the thoughts as output, priced."""
    prompt = int(getattr(metadata, "prompt_token_count", 0) or 0)
    cached = int(getattr(metadata, "cached_content_token_count", 0) or 0)
    tools = int(getattr(metadata, "tool_use_prompt_token_count", 0) or 0)
    out = int(getattr(metadata, "candidates_token_count", 0) or 0)
    out += int(getattr(metadata, "thoughts_token_count", 0) or 0)
    fresh = max(0, prompt - cached) + tools
    usd = (fresh * price.input + cached * price.cache_read + out * price.output) / 1e6
    return Usage(fresh, cached, 0, out, usd, stop, ms)


def gemini_cut(metadata: Any, chars: int, last_prompt: int, price: ModelPrice, ms: int) -> Usage:
    """A call cut mid-stream, priced as an estimate: its latest ``usage_metadata`` when a chunk
    carried one (the output at least the text streamed), else the last call's prompt as fresh
    input and the streamed text as output."""
    out = text_tokens(chars)
    if metadata is not None:
        usage = gemini_usage(metadata, price, CUT, ms)
        extra = max(0, out - usage.output)
        return Usage(usage.input, usage.cache_read, 0, usage.output + extra,
                     usage.usd + extra * price.output / 1e6, CUT, ms)  # fmt: skip
    usd = (last_prompt * price.input + out * price.output) / 1e6
    return Usage(last_prompt, 0, 0, out, usd, CUT, ms)


class GeminiBrain:
    """:class:`pepin.voice_cascade.stages.Brain` on Gemini; the key from ``GEMINI_API_KEY``."""

    def __init__(
        self,
        config: BrainConfig,
        *,
        persona: str,
        declarations: list[dict[str, Any]],
        price: ModelPrice,
        client: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``declarations``: the registry's Gemini function declarations the brain may call
        (:func:`pepin.tools.schemas.gemini_function_declarations`), ``end_conversation`` is
        added; ``client``: a ``genai.Client`` (built from the environment when None)."""
        from google.genai import types  # lazy: the unit tier runs it over a fake client

        if client is None:
            from google import genai

            client = genai.Client(
                http_options=types.HttpOptions(timeout=int(config.timeout_s * 1000))
            )
        self.config, self.persona, self.price = config, persona, price
        self.client: Any = client
        self.clock = clock
        self.types: Any = types
        end = {"name": END_TOOL["name"], "description": END_TOOL["description"]}
        self.declarations = [*declarations, end]
        self.places_line = PLACES_UNKNOWN
        self.history: list[Any] = []
        self._turn: list[Any] = []
        self._said: list[Sentence] = []
        self._acted = False
        self.last_prompt = 0  # the last call's input tokens (a call cut early is billed so)
        self._cache: str | None = None  # the cache's name
        self._cache_key = ""  # the system instruction it holds
        self._cache_until = 0.0  # when its TTL runs out (monotonic)
        self._cache_task: asyncio.Task[None] | None = None
        self.cache_tokens = 0

    # -- the request -----------------------------------------------------------------------

    def system(self) -> str:
        """The system instruction: the persona, the call-first rule, the places book."""
        return f"{self.persona}\n\n{CALL_FIRST}\n\n{self.places_line}"

    def thinking(self) -> Any:
        """The preset's thinking setting, None when it sets none."""
        cfg = self.config
        if cfg.thinking_budget is None and cfg.thinking_level is None:
            return None
        return self.types.ThinkingConfig(
            thinking_budget=cfg.thinking_budget, thinking_level=cfg.thinking_level
        )

    def tools(self) -> list[Any]:
        """The function declarations as one Gemini tool."""
        t = self.types
        functions = [
            t.FunctionDeclaration(
                name=d["name"], description=d["description"], parameters=d.get("parameters")
            )
            for d in self.declarations
        ]
        return [t.Tool(function_declarations=functions)]

    def request(
        self, *, final: bool = False, max_tokens: int | None = None, cache: str | None = None
    ) -> Any:
        """The call's ``GenerateContentConfig``; ``final``: no more tool calls; ``cache``: the
        context cache holding the system and the tools (then not sent again)."""
        t, cfg = self.types, self.config
        if cache is not None and not final:
            return t.GenerateContentConfig(
                cached_content=cache,
                thinking_config=self.thinking(),
                max_output_tokens=max_tokens or cfg.max_tokens,
                automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            )
        return t.GenerateContentConfig(
            system_instruction=self.system(),
            tools=self.tools(),
            thinking_config=self.thinking(),
            max_output_tokens=max_tokens or cfg.max_tokens,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(mode="NONE"))
            if final
            else None,
        )

    def contents(self) -> list[Any]:
        """The history and the open turn, adjacent turns of one role joined (Gemini wants
        them alternating)."""
        joined: list[Any] = []
        for content in [*self.history, *self._turn]:
            if joined and joined[-1].role == content.role:
                parts = [*(joined[-1].parts or []), *(content.parts or [])]
                joined[-1] = self.types.Content(role=content.role, parts=parts)
            else:
                joined.append(content)
        return joined

    def user(self, *parts: Any) -> Any:
        """A user turn of ``parts``."""
        return self.types.Content(role="user", parts=list(parts))

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
        """Make the context cache now (the connection opens with it); without a cache, one
        short call with the system and the tools opens the connection instead."""
        if not self.config.prewarm:
            return None
        t0 = self.clock()
        if self.config.cache:
            await self._make_cache()
            storage = self.cache_tokens * CACHE_STORAGE_USD_PER_MTOK_HOUR * CACHE_TTL_S / 3600e6
            ms = round((self.clock() - t0) * 1000)
            return Usage(cache_write=self.cache_tokens, usd=storage, ms=ms)
        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=[self.user(self.types.Part(text=WARM_UP))],
            config=self.request(max_tokens=16),
        )
        ms = round((self.clock() - t0) * 1000)
        return gemini_usage(response.usage_metadata, self.price, None, ms)

    async def _make_cache(self) -> None:
        """A new context cache for the current system instruction (the old one deleted)."""
        t = self.types
        key = self.system()
        created = await self.client.aio.caches.create(
            model=self.config.model,
            config=t.CreateCachedContentConfig(
                system_instruction=key,
                tools=self.tools(),
                ttl=f"{CACHE_TTL_S}s",
                display_name="pepin-voice-cascade",
            ),
        )
        old = self._cache
        self._cache, self._cache_key = created.name, key
        self._cache_until = self.clock() + CACHE_TTL_S
        meta = getattr(created, "usage_metadata", None)
        self.cache_tokens = int(getattr(meta, "total_token_count", 0) or 0)
        if old is not None:
            await self._delete(old)

    async def _renew_cache(self) -> None:
        name = self._cache
        if name is None:
            return
        await self.client.aio.caches.update(
            name=name, config=self.types.UpdateCachedContentConfig(ttl=f"{CACHE_TTL_S}s")
        )
        self._cache_until = self.clock() + CACHE_TTL_S

    async def _delete(self, name: str) -> None:
        with contextlib.suppress(Exception):  # it expires by itself anyway
            await self.client.aio.caches.delete(name=name)

    def cache(self) -> str | None:
        """The context cache to use now, or None; a cache that is stale (the places changed),
        expired or close to it is remade or renewed in the background meanwhile."""
        if not self.config.cache:
            return None
        busy = self._cache_task is not None and not self._cache_task.done()
        now = self.clock()
        fresh = self._cache is not None and self._cache_key == self.system()
        if fresh and now < self._cache_until - CACHE_RENEW_S:
            return self._cache
        if not busy:
            job = self._renew_cache() if fresh and now < self._cache_until else self._make_cache()
            self._cache_task = asyncio.ensure_future(self._quietly(job))
        return self._cache if fresh and now < self._cache_until else None

    @staticmethod
    async def _quietly(job: Any) -> None:
        with contextlib.suppress(Exception):  # the turn goes on uncached
            await job

    async def aclose(self) -> None:
        """Delete the context cache (the run ends)."""
        if self._cache is not None:
            await self._delete(self._cache)
            self._cache = None

    async def turn(self, text: str, tools: ToolExecutor, emit: Emit) -> None:
        """One turn: calls until one ends without function calls (at most ``max_tool_rounds``
        rounds of tools)."""
        t = self.types
        self._turn = [self.user(t.Part(text=text))]
        self._said, self._acted = [], False
        first = True
        for round_index in range(self.config.max_tool_rounds + 1):
            splitter = SentenceSplitter()
            parts: list[Any] = []
            calls: list[Any] = []
            metadata: Any = None
            stop: str | None = None
            t0 = self.clock()

            def say(sentence: str, index: int = round_index) -> None:
                said = Sentence(sentence, index)
                self._said.append(said)
                emit(said)

            stream = await self.client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=self.contents(),
                config=self.request(
                    final=round_index == self.config.max_tool_rounds, cache=self.cache()
                ),
            )
            chars = 0
            try:
                async with contextlib.aclosing(stream):
                    async for chunk in stream:
                        metadata = chunk.usage_metadata or metadata
                        candidate = chunk.candidates[0] if chunk.candidates else None
                        if candidate is None:
                            continue
                        if candidate.finish_reason is not None:
                            stop = str(
                                getattr(candidate.finish_reason, "value", candidate.finish_reason)
                            )
                        for part in (candidate.content.parts if candidate.content else None) or []:
                            if (
                                part.text
                                or part.function_call is not None
                                or part.thought_signature
                            ):
                                parts.append(part)  # an empty part carries nothing to send back
                            if part.function_call is not None:
                                calls.append(part.function_call)
                                rest = splitter.flush()  # the words before a call end here
                                if rest:
                                    say(rest)
                            elif part.text and not part.thought:
                                if first:
                                    first = False
                                    emit(FirstToken())
                                chars += len(part.text)
                                for sentence in splitter.feed(part.text):
                                    say(sentence)
                                for early in splitter.ahead():
                                    emit(Ahead(early, round_index))
            except asyncio.CancelledError:
                ms = round((self.clock() - t0) * 1000)
                emit(gemini_cut(metadata, chars, self.last_prompt, self.price, ms))
                raise
            rest = splitter.flush()
            if rest:
                say(rest)
            priced = gemini_usage(metadata, self.price, stop, round((self.clock() - t0) * 1000))
            self.last_prompt = priced.input + priced.cache_read
            emit(priced)
            if parts:
                self._turn.append(t.Content(role="model", parts=parts))
            if not calls:
                return
            self._acted = True
            for call in calls:
                emit(ToolUse(call.id or call.name, call.name, dict(call.args or {})))
            done = await asyncio.gather(*(self._run(tools, call, emit) for call in calls))
            answers = [answer for answer, _, _ in done]
            pictures = [part for _, images, _ in done for part in images]
            self._turn.append(self.user(*answers, *pictures))
            if not any(follow for _, _, follow in done):
                return

    async def _run(self, tools: ToolExecutor, call: Any, emit: Emit) -> tuple[Any, list[Any], bool]:
        """One function call: its response part, its pictures as inline parts, and whether the
        model should speak after it."""
        t = self.types
        t0 = self.clock()
        result: Result = dict(await tools(call.name, dict(call.args or {})))
        follow = bool(result.pop(FOLLOW_UP, True))
        text, images = render(result)
        emit(ToolDone(call.id or call.name, call.name, text, round((self.clock() - t0) * 1000)))
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"text": text}
        answer = t.Part(
            function_response=t.FunctionResponse(id=call.id, name=call.name, response=payload)
        )
        return answer, [self._picture(image) for image in images], follow

    def _picture(self, image: Image) -> Any:
        return self.types.Part.from_bytes(data=image.data, mime_type=image.mime)

    def commit(self) -> None:
        """The turn ran to its end: it joins the history."""
        self.history += self._turn
        self._turn, self._said, self._acted = [], [], False

    def abandon(self) -> None:
        """Drop the open turn."""
        self._turn, self._said, self._acted = [], [], False

    def interrupt(self, played: Played) -> None:
        """As the Claude brain: rounds with function calls stay (answered "interrupted" when
        their results never came); the round being said keeps only the words heard."""
        if not self._turn:
            return
        t = self.types
        kept: list[Any] = [self._turn[0]]
        rounds = self._turn[1:]
        tool_rounds = 0
        for index in range(0, len(rounds), 2):
            content = rounds[index]
            calls = [p.function_call for p in content.parts or [] if p.function_call is not None]
            if not calls:
                break
            kept.append(content)
            if index + 1 < len(rounds):
                kept.append(rounds[index + 1])
            else:
                kept.append(
                    self.user(
                        *(
                            t.Part(
                                function_response=t.FunctionResponse(
                                    id=c.id, name=c.name, response={"error": NOT_AWAITED}
                                )
                            )
                            for c in calls
                        )
                    )
                )
            tool_rounds += 1
        heard = [s.text for s in self._said[: played.sentences] if s.round == tool_rounds]
        cut = self._said[played.sentences] if played.sentences < len(self._said) else None
        if played.partial and cut is not None and cut.round == tool_rounds:
            heard.append(played.partial)
        if heard:
            words = " ".join(heard) + INTERRUPTED
            kept.append(t.Content(role="model", parts=[t.Part(text=words)]))
        self.history += kept
        self._turn, self._said, self._acted = [], [], False
