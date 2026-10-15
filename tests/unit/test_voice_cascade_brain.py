"""The Claude brain over a fake Messages API: the request laid out for the cache, sentences out
as they end, tools between rounds, the price of each call (one cut mid-stream too), a turn cut
to what was heard, the persona's first-sentence line, and the server-side web search (its tool,
its blocks in the reply and the history, its count and price)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pepin.tools.registry import Result
from pepin.voice_cascade.brain import (
    INTERRUPTED,
    NOT_AWAITED,
    PLACES_UNKNOWN,
    ClaudeBrain,
    cut_usage,
    tool_definitions,
    usage_of,
)
from pepin.voice_cascade.config import DEFAULT_PRICES, BrainConfig, CascadeConfig, ModelPrice
from pepin.voice_cascade.engines import persona_text
from pepin.voice_cascade.stages import (
    CUT,
    BrainEvent,
    FirstToken,
    Played,
    Sentence,
    ToolDone,
    ToolUse,
    Usage,
    text_tokens,
)

HAIKU = DEFAULT_PRICES["claude-haiku-5-5"]
SEARCHED = replace(HAIKU, web_search_per_1000=10.0)  # the file's price of a search
CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(id: str, name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=args)


def usage(read: int = 100, write: int = 10, fresh: int = 5, out: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=fresh,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write,
        output_tokens=out,
    )


class FakeStream:
    """One streamed call: its start (the prompt's usage), text deltas, the text block's stop and
    each tool call's start, the final message. ``opened`` holds the start back, ``gate`` each
    delta."""

    def __init__(
        self,
        deltas: list[str],
        tools: list[SimpleNamespace],
        gate: asyncio.Event | None,
        opened: asyncio.Event | None = None,
    ):
        self.deltas, self.tools, self.gate, self.opened = deltas, tools, gate, opened
        text = "".join(deltas)
        content = ([text_block(text)] if text else []) + tools
        self.message = SimpleNamespace(
            content=content, stop_reason="tool_use" if tools else "end_turn", usage=usage()
        )

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._events()

    async def _events(self) -> AsyncIterator[SimpleNamespace]:
        if self.opened is not None:
            await self.opened.wait()
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage(out=1)))
        for delta in self.deltas:
            yield SimpleNamespace(type="text", text=delta)
            if self.gate is not None:
                await self.gate.wait()
        yield SimpleNamespace(type="content_block_stop")
        for tool in self.tools:
            yield SimpleNamespace(type="content_block_start", content_block=tool)

    async def get_final_message(self) -> SimpleNamespace:
        return self.message


PAGE = "https://www.gismeteo.ru/weather-montreal-3920/"


def search_use(id: str = "s1") -> SimpleNamespace:
    """The model's search as the API reports it (it ran on the server)."""
    query = {"query": "погода Монреаль завтра"}
    return SimpleNamespace(type="server_tool_use", id=id, name="web_search", input=query)


def search_result(use_id: str = "s1") -> SimpleNamespace:
    page = SimpleNamespace(
        type="web_search_result",
        url=PAGE,
        title="Погода в Монреале на завтра",
        encrypted_content="EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5LTQ1NGUtODgxNC0xNjNjNjZmN2YzZTI",
        page_age="14 октября 2026",
    )
    return SimpleNamespace(type="web_search_tool_result", tool_use_id=use_id, content=[page])


def cited(text: str) -> SimpleNamespace:
    """A text block quoting a search result (the URL in its citation, never in its text)."""
    citation = SimpleNamespace(
        type="web_search_result_location",
        url=PAGE,
        title="Погода в Монреале на завтра",
        cited_text="Завтра днём +5, дождь",
        encrypted_index="Eo8BCioIAhgBIiQyYjE0ZDE5My04OTI3",
    )
    return SimpleNamespace(type="text", text=text, citations=[citation])


class BlockStream:
    """One streamed call as the API sends a reply with server tool blocks: each block's start,
    a text block's text as one delta, its stop; the final message with ``stop`` and the
    searches in its usage. ``hold``: the block type whose start the stream stops at
    (``reached`` set there)."""

    def __init__(
        self,
        blocks: list[SimpleNamespace],
        stop: str = "end_turn",
        searches: int = 0,
        hold: str | None = None,
    ) -> None:
        self.blocks, self.hold = blocks, hold
        self.log: list[str] = []  # the block starts sent, in order
        self.reached = asyncio.Event()
        server = SimpleNamespace(web_search_requests=searches, web_fetch_requests=0)
        billed = SimpleNamespace(**vars(usage()), server_tool_use=server)
        self.message = SimpleNamespace(content=blocks, stop_reason=stop, usage=billed)

    async def __aenter__(self) -> "BlockStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._events()

    async def _events(self) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage(out=1)))
        for block in self.blocks:
            self.log.append(block.type)
            yield SimpleNamespace(type="content_block_start", content_block=block)
            if block.type == self.hold:
                self.reached.set()
                await asyncio.Event().wait()  # the search never comes back: the turn is cut
            if block.type == "text":
                yield SimpleNamespace(type="text", text=block.text)
            yield SimpleNamespace(type="content_block_stop", content_block=block)

    async def get_final_message(self) -> SimpleNamespace:
        return self.message


Round = tuple[list[str], list[SimpleNamespace]] | BlockStream


class FakeMessages:
    """Answers call n with ``rounds[n]``: (text deltas, tool calls), or a :class:`BlockStream`
    as it is; keeps every request."""

    def __init__(self, rounds: Sequence[Round]) -> None:
        self.rounds = rounds
        self.requests: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None
        self.opened: asyncio.Event | None = None

    def stream(self, **kwargs: Any) -> FakeStream | BlockStream:
        self.requests.append(kwargs)
        entry = self.rounds[len(self.requests) - 1]
        if isinstance(entry, BlockStream):
            return entry
        deltas, tools = entry
        return FakeStream(deltas, tools, self.gate, self.opened)

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.created.append(kwargs)
        return SimpleNamespace(content=[], stop_reason="max_tokens", usage=usage(0, 4000, 5, 0))


def brain_over(
    rounds: Sequence[Round], price: ModelPrice = HAIKU, **config: Any
) -> tuple[ClaudeBrain, FakeMessages]:
    messages = FakeMessages(rounds)
    client = SimpleNamespace(messages=messages)
    tools = [{"name": "see", "description": "look", "input_schema": {"type": "object"}}]
    brain = ClaudeBrain(
        BrainConfig(**config), persona="You are Pepin.", tools=tools, price=price, client=client
    )
    return brain, messages


def run_turn(
    brain: ClaudeBrain, text: str, results: dict[str, Result] | None = None
) -> list[BrainEvent]:
    events: list[BrainEvent] = []

    async def tools(name: str, args: dict[str, Any]) -> Result:
        return (results or {}).get(name, {"ok": True})

    asyncio.run(brain.turn(text, tools, events.append))
    return events


def test_the_request_puts_the_persona_in_the_cache_and_the_places_after_it() -> None:
    brain, _ = brain_over([])
    brain.begin(["home", "printer"])
    request = brain.request([{"role": "user", "content": "hi"}])
    persona, places = request["system"]
    assert persona == {
        "type": "text",
        "text": "You are Pepin.",
        "cache_control": {"type": "ephemeral"},
    }
    assert "home, printer" in places["text"] and "cache_control" not in places
    assert request["thinking"] == {"type": "disabled"}
    assert request["output_config"] == {"effort": "low"}
    assert request["cache_control"] == {"type": "ephemeral"}  # the conversation, as it grows
    assert "tool_choice" not in request
    assert brain.request([], final=True)["tool_choice"] == {"type": "none"}


def test_switches_leave_their_parameters_out() -> None:
    brain, _ = brain_over([], thinking=None, effort=None, cache=False)
    brain.begin(None)
    request = brain.request([])
    assert "thinking" not in request and "output_config" not in request
    assert "cache_control" not in request and "cache_control" not in request["system"][0]
    assert request["system"][1]["text"] == PLACES_UNKNOWN


def test_sonnet_off_is_between_tools() -> None:
    brain, _ = brain_over([], model="claude-sonnet-5-5", thinking="between_tools")
    assert brain.request([])["thinking"] == {"type": "between_tools"}


def test_sentences_go_out_as_they_end_and_the_turn_is_kept_on_commit() -> None:
    brain, messages = brain_over([(["Привет", "! Я до", "ма. Всё"], [])])
    brain.begin(None)
    events = run_turn(brain, "Пепин, где ты?")
    said = [e for e in events if isinstance(e, Sentence)]
    assert isinstance(events[0], FirstToken)
    assert [s.text for s in said] == ["Привет!", "Я дома.", "Всё"]
    usage_events = [e for e in events if isinstance(e, Usage)]
    assert len(usage_events) == 1 and usage_events[0].cache_read == 100
    assert brain.history == []  # nothing kept before the commit
    brain.commit()
    assert [m["role"] for m in brain.history] == ["user", "assistant"]
    assert messages.requests[0]["messages"] == [{"role": "user", "content": "Пепин, где ты?"}]


def test_a_tool_round_says_its_announcement_first_then_answers_with_the_result() -> None:
    rounds = [
        (["Сейчас гляну."], [tool_block("t1", "see", {})]),
        (["Вижу стол."], []),
    ]
    brain, messages = brain_over(rounds)
    brain.begin(None)
    events = run_turn(brain, "что видишь?", {"see": {"ok": True, "objects": ["стол"]}})
    kinds = [type(e).__name__ for e in events]
    assert kinds == [  # each round's first word is told ahead of its sentence (text_ahead)
        "FirstToken", "Ahead", "Sentence", "Usage", "ToolUse", "ToolDone", "Ahead", "Sentence",
        "Usage",
    ]  # fmt: skip
    assert [e.round for e in events if isinstance(e, Sentence)] == [0, 1]
    assert isinstance(events[4], ToolUse) and events[4].name == "see"
    assert isinstance(events[5], ToolDone) and "стол" in events[5].result
    second = messages.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"][0]["tool_use_id"] == "t1"
    assert brain.acted
    brain.commit()
    assert len(brain.history) == 4 and not brain.acted


def test_an_empty_reply_leaves_no_empty_message() -> None:
    brain, messages = brain_over([([], []), (["Да?"], [])])
    brain.begin(None)
    run_turn(brain, "(разговор не с роботом)")
    brain.commit()
    run_turn(brain, "Пепин")
    roles = [m["role"] for m in messages.requests[1]["messages"]]
    assert roles == ["user", "user"]  # the API joins them into one turn


def test_a_result_that_needs_no_answer_ends_the_turn_without_another_call() -> None:
    rounds = [(["Еду."], [tool_block("t1", "go_to", {"place": "printer"})])]
    brain, messages = brain_over(rounds)
    brain.begin(None)
    events = run_turn(brain, "к принтеру", {"go_to": {"status": "driving", "follow_up": False}})
    assert len(messages.requests) == 1
    done = [e for e in events if isinstance(e, ToolDone)]
    assert "follow_up" not in done[0].result
    brain.commit()
    assert [m["role"] for m in brain.history] == ["user", "assistant", "user"]


def test_the_round_limit_forbids_more_tools() -> None:
    rounds = [([], [tool_block("t1", "see", {})]), (["Готово."], [])]
    brain, messages = brain_over(rounds, max_tool_rounds=1)
    brain.begin(None)
    run_turn(brain, "смотри")
    assert "tool_choice" not in messages.requests[0]
    assert messages.requests[1]["tool_choice"] == {"type": "none"}


def test_an_interrupt_keeps_the_tool_round_and_only_the_words_heard() -> None:
    rounds = [
        (["Сейчас гляну."], [tool_block("t1", "see", {})]),
        (["Вижу стол и чашку. На столе ещё книга."], []),
    ]
    brain, _ = brain_over(rounds)
    brain.begin(None)
    run_turn(brain, "что видишь?")
    brain.interrupt(Played(sentences=2, partial="На столе"))
    roles = [m["role"] for m in brain.history]
    assert roles == ["user", "assistant", "user", "assistant"]
    last = brain.history[-1]["content"][0]["text"]
    assert last == "Вижу стол и чашку. На столе" + INTERRUPTED


def test_an_interrupt_before_anything_of_the_answer_was_heard_leaves_no_reply() -> None:
    brain, _ = brain_over([(["Длинный ответ."], [])])
    brain.begin(None)
    run_turn(brain, "расскажи")
    brain.interrupt(Played())
    assert [m["role"] for m in brain.history] == ["user"]


def test_an_interrupt_while_a_tool_runs_answers_it_as_not_awaited() -> None:
    rounds = [(["Сейчас."], [tool_block("t1", "see", {})])]
    brain, _ = brain_over(rounds)
    brain.begin(None)
    started = asyncio.Event()

    async def scenario() -> None:
        async def slow_tool(name: str, args: dict[str, Any]) -> Result:
            started.set()
            await asyncio.sleep(10)
            return {"ok": True}

        task = asyncio.create_task(brain.turn("смотри", slow_tool, lambda e: None))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    brain.interrupt(Played(sentences=1))
    assert [m["role"] for m in brain.history] == ["user", "assistant", "user"]
    result = brain.history[-1]["content"][0]
    assert result["tool_use_id"] == "t1" and result["content"] == NOT_AWAITED


def test_abandon_forgets_the_turn() -> None:
    brain, _ = brain_over([(["Да?"], [])])
    brain.begin(None)
    run_turn(brain, "Пепин")
    brain.abandon()
    assert brain.history == []


def test_prewarm_is_a_zero_token_call_without_the_conversation_breakpoint() -> None:
    brain, messages = brain_over([])
    brain.begin(None)
    warm = asyncio.run(brain.prewarm())
    assert warm is not None and warm.cache_write == 4000
    (kwargs,) = messages.created
    assert kwargs["max_tokens"] == 0 and "cache_control" not in kwargs
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    brain_off, messages_off = brain_over([], prewarm=False)
    assert asyncio.run(brain_off.prewarm()) is None and messages_off.created == []


def test_usage_is_priced_by_kind() -> None:
    priced = usage_of(usage(read=1_000_000, write=0, fresh=0, out=0), HAIKU, "end_turn", 5)
    assert priced.usd == pytest.approx(HAIKU.cache_read)
    priced = usage_of(
        usage(read=0, write=1_000_000, fresh=1_000_000, out=1_000_000), HAIKU, None, 0
    )
    assert priced.usd == pytest.approx(HAIKU.cache_write + HAIKU.input + HAIKU.output)


def cut_turn(brain: ClaudeBrain, messages: FakeMessages, started: bool) -> list[BrainEvent]:
    """A turn cancelled in its first text delta (``started``) or before its stream began."""
    events: list[BrainEvent] = []

    async def scenario() -> None:
        first = asyncio.Event()

        def emit(event: BrainEvent) -> None:
            events.append(event)
            if isinstance(event, FirstToken):
                first.set()

        async def tools(name: str, args: dict[str, Any]) -> Result:
            return {"ok": True}

        messages.gate, messages.opened = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(brain.turn("расскажи", tools, emit))
        if started:
            messages.opened.set()
            await first.wait()
        else:
            await asyncio.sleep(0.01)  # the call is out, its start not back yet
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    return events


def test_a_call_cut_mid_stream_still_reports_its_usage() -> None:
    brain, messages = brain_over([(["Сейчас расскажу", " всё по порядку."], [])])
    brain.begin(None)
    events = cut_turn(brain, messages, started=True)
    (cut,) = [e for e in events if isinstance(e, Usage)]
    assert cut.stop == CUT
    assert (cut.cache_read, cut.cache_write, cut.input) == (100, 10, 5)  # its message_start's
    assert cut.output == text_tokens(len("Сейчас расскажу")) and cut.usd > 0
    brain.abandon()
    assert brain.history == []


def test_a_call_cut_before_its_start_is_billed_as_the_last_prompt_read() -> None:
    brain, messages = brain_over([([], [])])
    brain.begin(None)
    asyncio.run(brain.prewarm())  # 4000 + 5 input tokens written
    events = cut_turn(brain, messages, started=False)
    (cut,) = [e for e in events if isinstance(e, Usage)]
    assert cut.stop == CUT and (cut.cache_read, cut.output) == (4005, 0)
    assert cut.usd == pytest.approx(4005 * HAIKU.cache_read / 1e6)


def test_the_first_sentence_line_is_one_fixed_line_of_the_cached_persona() -> None:
    persona = "You are Pepin.\n"
    assert persona_text(persona, 0) is persona  # off: today's cached prefix, byte for byte
    six = persona_text(persona, 6)
    assert six.startswith("You are Pepin.\n\n") and "at most 6 words" in six
    assert six == persona_text(persona, 6) != persona_text(persona, 5)  # one prefix per config
    assert "at most 1 word " in persona_text(persona, 1)
    tools = [{"name": "see", "description": "look", "input_schema": {"type": "object"}}]
    client = SimpleNamespace(messages=FakeMessages([]))
    brain = ClaudeBrain(BrainConfig(), persona=six, tools=tools, price=HAIKU, client=client)
    block = brain.system()[0]
    assert block["text"] == six and block["cache_control"] == {"type": "ephemeral"}


def test_the_tools_keep_the_registry_order_and_end_the_conversation() -> None:
    from pepin.tools import TOOLS

    allowed = {"go_to", "where_am_i"}
    names = [t["name"] for t in tool_definitions(TOOLS, allowed)]
    order = [t.name for t in TOOLS if t.name in allowed]
    assert names == [*order, "end_conversation"]


# -- the server-side web search ---------------------------------------------------------------


def searched_reply(searches: int = 1) -> BlockStream:
    """A reply that searched once: its announcement, the search, and a sentence quoting a result
    across three text blocks (as the API splits cited text)."""
    return BlockStream(
        [
            text_block("Сейчас поищу."),
            search_use(),
            search_result(),
            text_block("По данным сайта Гисметео, завтра "),
            cited("плюс пять и дождь"),
            text_block(". Зонт пригодится."),
        ],
        searches=searches,
    )


def test_the_web_search_follows_the_tools_and_is_in_the_warm_up_too() -> None:
    brain, messages = brain_over([])
    brain.begin(None)
    tools = brain.request([])["tools"]
    assert [t["name"] for t in tools] == ["see", "web_search"]
    assert tools[-1] == {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    asyncio.run(brain.prewarm())
    assert messages.created[0]["tools"] == tools  # one cached prefix for both
    narrow, _ = brain_over(
        [],
        web_search_tool="web_search_20260209",
        web_search_max_uses=2,
        web_search_allowed_domains=("gismeteo.ru", "ru.wikipedia.org"),
    )
    assert narrow.request([])["tools"][-1] == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 2,
        "allowed_domains": ["gismeteo.ru", "ru.wikipedia.org"],
    }


def test_the_switch_off_sends_no_search_tool() -> None:
    brain, messages = brain_over([(["Не знаю."], [])], web_search=False)
    brain.begin(None)
    run_turn(brain, "какая завтра погода?")
    assert [t["name"] for t in messages.requests[0]["tools"]] == ["see"]


def test_a_searched_reply_is_said_whole_and_its_search_counted_and_priced() -> None:
    reply = searched_reply()
    brain, messages = brain_over([reply, (["Пожалуйста."], [])], price=SEARCHED)
    brain.begin(None)
    order: list[str] = reply.log
    events: list[BrainEvent] = []

    def emit(event: BrainEvent) -> None:
        events.append(event)
        if isinstance(event, Sentence):
            order.append(event.text)

    async def tools(name: str, args: dict[str, Any]) -> Result:
        return {"ok": True}

    asyncio.run(brain.turn("какая завтра погода?", tools, emit))
    said = [e.text for e in events if isinstance(e, Sentence)]
    assert said == [  # the cited sentence whole, though it spans three text blocks
        "Сейчас поищу.",
        "По данным сайта Гисметео, завтра плюс пять и дождь.",
        "Зонт пригодится.",
    ]
    assert not any("http" in s for s in said)
    # the announcement goes to the voice as the search starts, before its result comes
    assert order.index("Сейчас поищу.") < order.index("web_search_tool_result")
    assert not any(isinstance(e, ToolUse | ToolDone) for e in events) and not brain.acted
    (priced,) = [e for e in events if isinstance(e, Usage)]
    tokens = usage_of(usage(), SEARCHED, "end_turn", 0).usd
    assert priced.web_searches == 1 and priced.usd == pytest.approx(tokens + 10.0 / 1000)
    brain.commit()
    kept = brain.history[-1]["content"]
    assert [b.type for b in kept] == [
        "text", "server_tool_use", "web_search_tool_result", "text", "text", "text",
    ]  # fmt: skip
    run_turn(brain, "спасибо")
    assert messages.requests[1]["messages"][1]["content"] == kept  # the next call reads it all


def test_a_paused_search_is_sent_again_and_its_reply_kept_as_one() -> None:
    paused = BlockStream([text_block("Сейчас поищу."), search_use()], stop="pause_turn")
    resumed = BlockStream([search_result(), text_block("Нашёл, плюс пять.")], searches=1)
    brain, messages = brain_over([paused, resumed])
    brain.begin(None)
    events = run_turn(brain, "погода на завтра?")
    second = messages.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]  # no "continue" message
    assert second[1]["content"] == paused.blocks
    said = [(e.text, e.round) for e in events if isinstance(e, Sentence)]
    assert said == [("Сейчас поищу.", 0), ("Нашёл, плюс пять.", 0)]  # one round of speech
    assert sum(e.web_searches for e in events if isinstance(e, Usage)) == 1
    brain.commit()
    assert [m["role"] for m in brain.history] == ["user", "assistant"]
    assert [b.type for b in brain.history[1]["content"]] == [
        "text", "server_tool_use", "web_search_tool_result", "text",
    ]  # fmt: skip


def test_an_interrupt_of_a_searched_reply_keeps_only_the_words_heard() -> None:
    brain, _ = brain_over([searched_reply()])
    brain.begin(None)
    run_turn(brain, "какая завтра погода?")
    brain.interrupt(Played(sentences=1))
    assert [m["role"] for m in brain.history] == ["user", "assistant"]
    assert brain.history[1]["content"] == [{"type": "text", "text": "Сейчас поищу." + INTERRUPTED}]


def test_a_search_cut_by_max_tokens_leaves_no_unanswered_call() -> None:
    cut = BlockStream([text_block("Сейчас поищу."), search_use()], stop="max_tokens")
    brain, _ = brain_over([cut])
    brain.begin(None)
    run_turn(brain, "погода?")
    brain.commit()
    assert [b.type for b in brain.history[1]["content"]] == ["text"]


def test_a_call_cut_during_its_search_is_billed_for_it() -> None:
    reply = BlockStream(
        [text_block("Сейчас поищу."), search_use(), search_result()],
        hold="web_search_tool_result",
    )
    brain, _ = brain_over([reply], price=SEARCHED)
    brain.begin(None)
    events: list[BrainEvent] = []

    async def scenario() -> None:
        async def tools(name: str, args: dict[str, Any]) -> Result:
            return {"ok": True}

        task = asyncio.create_task(brain.turn("погода?", tools, events.append))
        await reply.reached.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    (cut,) = [e for e in events if isinstance(e, Usage)]
    assert cut.stop == CUT and cut.web_searches == 1
    tokens = cut_usage(usage(out=1), len("Сейчас поищу."), 0, SEARCHED, 0).usd
    assert cut.usd == pytest.approx(tokens + 10.0 / 1000)


def test_usage_prices_and_sums_the_searches() -> None:
    three = SimpleNamespace(server_tool_use=SimpleNamespace(web_search_requests=3))
    priced = usage_of(three, SEARCHED, "end_turn", 0)
    assert priced.web_searches == 3 and priced.usd == pytest.approx(3 * 10.0 / 1000)
    assert usage_of(usage(), SEARCHED, None, 0).web_searches == 0  # no server tool reported
    assert priced.plus(priced).web_searches == 6


def test_the_switch_and_knobs_read_from_the_file() -> None:
    config = CascadeConfig.load(CONFIG)
    b = config.brain
    assert b.web_search and b.web_search_tool == "web_search_20250305"
    assert b.web_search_max_uses == 1 and b.web_search_allowed_domains == ()
    assert config.price.web_search_per_1000 == pytest.approx(10.0)
    assert config.price.input == pytest.approx(4.00)  # the model's own prices kept
    off = CascadeConfig.load(
        CONFIG, ["brain.web_search=false", "brain.web_search_allowed_domains=gismeteo.ru"]
    )
    assert not off.brain.web_search and off.brain.web_search_allowed_domains == ("gismeteo.ru",)
    with pytest.raises(ValueError, match="web_search_tool"):
        CascadeConfig.load(CONFIG, ["brain.web_search_tool=web_search_2099"])
    with pytest.raises(ValueError, match="web_search_max_uses"):
        CascadeConfig.load(CONFIG, ["brain.web_search_max_uses=0"])
