"""The Claude brain over a fake Messages API: the request laid out for the cache, sentences out
as they end, tools between rounds, the price of each call, and a turn cut to what was heard."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from pepin.tools.registry import Result
from pepin.voice_cascade.brain import (
    INTERRUPTED,
    NOT_AWAITED,
    PLACES_UNKNOWN,
    ClaudeBrain,
    tool_definitions,
    usage_of,
)
from pepin.voice_cascade.config import DEFAULT_PRICES, BrainConfig
from pepin.voice_cascade.stages import (
    BrainEvent,
    FirstToken,
    Played,
    Sentence,
    ToolDone,
    ToolUse,
    Usage,
)

HAIKU = DEFAULT_PRICES["claude-haiku-5-5"]


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
    """One streamed call: text deltas, a block stop before each tool call, the final message."""

    def __init__(self, deltas: list[str], tools: list[SimpleNamespace], gate: asyncio.Event | None):
        self.deltas, self.tools, self.gate = deltas, tools, gate
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
        for delta in self.deltas:
            yield SimpleNamespace(type="text", text=delta)
            if self.gate is not None:
                await self.gate.wait()
        yield SimpleNamespace(type="content_block_stop")

    async def get_final_message(self) -> SimpleNamespace:
        return self.message


class FakeMessages:
    """Answers call n with ``rounds[n]``: (text deltas, tool calls); keeps every request."""

    def __init__(self, rounds: list[tuple[list[str], list[SimpleNamespace]]]) -> None:
        self.rounds = rounds
        self.requests: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None

    def stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        deltas, tools = self.rounds[len(self.requests) - 1]
        return FakeStream(deltas, tools, self.gate)

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.created.append(kwargs)
        return SimpleNamespace(content=[], stop_reason="max_tokens", usage=usage(0, 4000, 5, 0))


def brain_over(
    rounds: list[tuple[list[str], list[SimpleNamespace]]], **config: Any
) -> tuple[ClaudeBrain, FakeMessages]:
    messages = FakeMessages(rounds)
    client = SimpleNamespace(messages=messages)
    tools = [{"name": "see", "description": "look", "input_schema": {"type": "object"}}]
    brain = ClaudeBrain(
        BrainConfig(**config), persona="You are Pepin.", tools=tools, price=HAIKU, client=client
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
    assert kinds == [
        "FirstToken", "Sentence", "Usage", "ToolUse", "ToolDone", "Sentence", "Usage",
    ]  # fmt: skip
    assert [e.round for e in events if isinstance(e, Sentence)] == [0, 1]
    assert isinstance(events[3], ToolUse) and events[3].name == "see"
    assert isinstance(events[4], ToolDone) and "стол" in events[4].result
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


def test_the_tools_keep_the_registry_order_and_end_the_conversation() -> None:
    from pepin.tools import TOOLS

    allowed = {"go_to", "where_am_i"}
    names = [t["name"] for t in tool_definitions(TOOLS, allowed)]
    order = [t.name for t in TOOLS if t.name in allowed]
    assert names == [*order, "end_conversation"]
