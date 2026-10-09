"""The Gemini brain over a fake google-genai client: the request, sentences out as they stream,
function calls answered with JSON and pictures in one turn, the implicit cache priced, and a
turn cut to what was heard."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

types = pytest.importorskip("google.genai.types")

from pepin.tools.registry import Image, Result  # noqa: E402
from pepin.voice_cascade.brain import INTERRUPTED  # noqa: E402
from pepin.voice_cascade.config import DEFAULT_PRICES, BrainConfig  # noqa: E402
from pepin.voice_cascade.gemini_brain import CALL_FIRST, GeminiBrain, gemini_usage  # noqa: E402
from pepin.voice_cascade.stages import (  # noqa: E402
    BrainEvent,
    FirstToken,
    Played,
    Sentence,
    ToolDone,
    ToolUse,
    Usage,
)

LITE = DEFAULT_PRICES["gemini-3.5-flash-lite"]
JPEG = b"\xff\xd8\xff\xe0picture\xff\xd9"


def metadata(prompt: int = 5000, cached: int = 0, out: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        cached_content_token_count=cached,
        candidates_token_count=out,
        thoughts_token_count=None,
        tool_use_prompt_token_count=None,
    )


def chunk(*parts: Any, finish: str | None = None, usage: Any = None) -> SimpleNamespace:
    candidate = SimpleNamespace(finish_reason=finish, content=SimpleNamespace(parts=list(parts)))
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def call(name: str, args: dict[str, Any] | None = None) -> Any:
    return types.Part(function_call=types.FunctionCall(id=f"id-{name}", name=name, args=args))


class FakeModels:
    """Answers call n with the chunks of ``rounds[n]``; keeps every request."""

    def __init__(self, rounds: list[list[SimpleNamespace]]) -> None:
        self.rounds = rounds
        self.requests: list[dict[str, Any]] = []

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.requests.append(kwargs)
        chunks = self.rounds[len(self.requests) - 1]

        async def stream() -> AsyncIterator[SimpleNamespace]:
            for item in chunks:
                yield item

        return stream()

    async def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(usage_metadata=metadata(5000, 0, 1))


def brain_over(
    rounds: list[list[SimpleNamespace]], **config: Any
) -> tuple[GeminiBrain, FakeModels]:
    models = FakeModels(rounds)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    settings = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "thinking": None,
                "effort": None, "thinking_level": "minimal", **config}  # fmt: skip
    see = {"name": "see", "description": "Take a picture."}
    brain = GeminiBrain(
        BrainConfig(**settings), persona="You are Pepin.", declarations=[see], price=LITE,
        client=client,
    )  # fmt: skip
    brain.begin(["home", "printer"])
    return brain, models


def run_turn(brain: GeminiBrain, text: str, results: dict[str, Result]) -> list[BrainEvent]:
    events: list[BrainEvent] = []

    async def tools(name: str, args: dict[str, Any]) -> Result:
        return results.get(name, {"ok": True})

    asyncio.run(brain.turn(text, tools, events.append))
    return events


def test_the_request_carries_persona_places_tools_and_the_lowest_thinking() -> None:
    brain, _ = brain_over([])
    config = brain.request()
    assert config.system_instruction.startswith("You are Pepin.")
    assert CALL_FIRST in config.system_instruction and "home, printer" in config.system_instruction
    names = [f.name for f in config.tools[0].function_declarations]
    assert names == ["see", "end_conversation"]
    level = config.thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == "minimal"
    assert config.thinking_config.thinking_budget is None
    assert config.automatic_function_calling.disable is True
    assert config.tool_config is None
    assert brain.request(final=True).tool_config.function_calling_config.mode == "NONE"
    budget, _ = brain_over([], thinking_level=None, thinking_budget=0)
    assert budget.request().thinking_config.thinking_budget == 0


def test_a_call_is_answered_with_its_json_and_its_picture_in_one_turn() -> None:
    rounds = [
        [chunk(types.Part(text="Сейчас ")), chunk(types.Part(text="гляну. "), call("see"))],
        [chunk(types.Part(text="Вижу красный круг.")),
         chunk(types.Part(text=""), finish="STOP", usage=metadata(6000, 4096, 8))],
    ]  # fmt: skip
    brain, models = brain_over(rounds)
    image = {"ok": True, "image": Image(JPEG, "image/jpeg", 320, 240)}
    events = run_turn(brain, "что видишь?", {"see": image})
    said = [e for e in events if isinstance(e, Sentence)]
    assert [s.text for s in said] == ["Сейчас гляну.", "Вижу красный круг."]
    assert [s.round for s in said] == [0, 1]
    assert isinstance(events[0], FirstToken)
    use = next(e for e in events if isinstance(e, ToolUse))
    done = next(e for e in events if isinstance(e, ToolDone))
    assert use.name == "see" and "attached" in done.result
    second = models.requests[1]["contents"]
    assert [c.role for c in second] == ["user", "model", "user"]
    answer, picture = second[2].parts
    assert answer.function_response.name == "see" and answer.function_response.id == "id-see"
    assert answer.function_response.response["ok"] is True
    assert picture.inline_data.data == JPEG and picture.inline_data.mime_type == "image/jpeg"
    usage = [e for e in events if isinstance(e, Usage)][-1]
    assert usage.cache_read == 4096 and usage.input == 6000 - 4096
    brain.commit()
    assert len(brain.history) == 4


def test_usage_prices_the_cached_prefix_at_the_cached_rate() -> None:
    usage = gemini_usage(metadata(1_000_000, 1_000_000, 0), LITE, "STOP", 5)
    assert usage.usd == pytest.approx(LITE.cache_read)
    usage = gemini_usage(metadata(1_000_000, 0, 1_000_000), LITE, None, 0)
    assert usage.usd == pytest.approx(LITE.input + LITE.output)
    assert gemini_usage(None, LITE, None, 0).usd == 0


def test_a_result_that_needs_no_answer_ends_the_turn() -> None:
    brain, models = brain_over([[chunk(types.Part(text="Еду."), call("go_to", {"place": "x"}))]])
    run_turn(brain, "к принтеру", {"go_to": {"status": "driving", "follow_up": False}})
    assert len(models.requests) == 1


def test_an_interrupt_keeps_the_call_and_the_words_heard() -> None:
    rounds = [
        [chunk(call("see"))],
        [chunk(types.Part(text="Вижу стол. На столе книга."))],
    ]
    brain, _ = brain_over(rounds)
    run_turn(brain, "что видишь?", {})
    brain.interrupt(Played(sentences=1, partial="На"))
    roles = [c.role for c in brain.history]
    assert roles == ["user", "model", "user", "model"]
    assert brain.history[-1].parts[0].text == "Вижу стол. На" + INTERRUPTED


def test_an_interrupt_while_a_call_runs_answers_it_as_not_awaited() -> None:
    brain, _ = brain_over([[chunk(call("see"))]])
    started = asyncio.Event()

    async def scenario() -> None:
        async def slow(name: str, args: dict[str, Any]) -> Result:
            started.set()
            await asyncio.sleep(10)
            return {"ok": True}

        task = asyncio.create_task(brain.turn("смотри", slow, lambda e: None))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    brain.interrupt(Played())
    assert [c.role for c in brain.history] == ["user", "model", "user"]
    assert "error" in brain.history[-1].parts[0].function_response.response


def test_adjacent_user_turns_are_joined_for_gemini() -> None:
    brain, models = brain_over([[chunk(types.Part(text=""))], [chunk(types.Part(text="Да?"))]])
    run_turn(brain, "(не роботу)", {})
    brain.commit()
    run_turn(brain, "Пепин", {})
    contents = models.requests[1]["contents"]
    assert [c.role for c in contents] == ["user"]
    assert [p.text for p in contents[0].parts] == ["(не роботу)", "Пепин"]


def test_prewarm_is_one_short_call() -> None:
    brain, models = brain_over([])
    warm = asyncio.run(brain.prewarm())
    assert warm is not None and warm.input == 5000
    assert models.requests[0]["config"].max_output_tokens == 16
