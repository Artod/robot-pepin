"""The OpenAI-compatible brain (Groq) over a fake client: the request, sentences out as they
stream, tool calls assembled from deltas and answered with tool messages and a picture, a model
that refuses pictures, a 429 surfaced, compact tool descriptions, and a turn cut to what was
heard."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

openai = pytest.importorskip("openai")

from pepin.tools.registry import Image, Result  # noqa: E402
from pepin.voice_cascade.brain import INTERRUPTED, NOT_AWAITED  # noqa: E402
from pepin.voice_cascade.chat_brain import NO_PICTURES, ChatBrain, compact  # noqa: E402
from pepin.voice_cascade.config import DEFAULT_PRICES, BrainConfig  # noqa: E402
from pepin.voice_cascade.stages import BrainEvent, Played, Sentence, ToolUse, Usage  # noqa: E402

OSS = DEFAULT_PRICES["openai/gpt-oss-120b"]
JPEG = b"\xff\xd8\xff\xe0picture\xff\xd9"
SEE = {
    "type": "function",
    "function": {
        "name": "see",
        "description": "One picture from the head camera. Call it alone for what you see.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def text(delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content=delta,
                                                                            tool_calls=None))],
        usage=None,
    )  # fmt: skip


def call(index: int, id: str | None, name: str | None, args: str | None) -> SimpleNamespace:
    part = SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=args))
    delta = SimpleNamespace(content=None, tool_calls=[part])
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=delta)], usage=None)


def done(prompt: int = 900, out: int = 10) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=None)
    return SimpleNamespace(choices=[], usage=usage)


class FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._run()

    async def _run(self) -> AsyncIterator[SimpleNamespace]:
        for chunk in self.chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


class FakeCompletions:
    """Answers call n with ``rounds[n]``: chunks, or an exception to raise."""

    def __init__(self, rounds: list[Any]) -> None:
        self.rounds = rounds
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        answer = self.rounds[len(self.requests) - 1]
        if isinstance(answer, Exception):
            raise answer
        return FakeStream(answer)


def brain_over(rounds: list[Any], **config: Any) -> tuple[ChatBrain, FakeCompletions]:
    completions = FakeCompletions(rounds)
    models = SimpleNamespace(list=lambda: _none())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), models=models)
    settings = {"provider": "openai", "model": "openai/gpt-oss-120b", "thinking": None,
                "effort": None, "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY", "reasoning_effort": "low",
                "extra_body": {"include_reasoning": False}, **config}  # fmt: skip
    brain = ChatBrain(
        BrainConfig(**settings), persona="You are Pepin.", tools=[SEE], price=OSS, client=client
    )
    brain.begin(["home"])
    return brain, completions


async def _none() -> None:
    return None


def run_turn(brain: ChatBrain, said: str, results: dict[str, Result]) -> list[BrainEvent]:
    events: list[BrainEvent] = []

    async def tools(name: str, args: dict[str, Any]) -> Result:
        return results.get(name, {"ok": True})

    asyncio.run(brain.turn(said, tools, events.append))
    return events


def http_error(kind: Any, status: int, message: str) -> Any:
    import httpx2

    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx2.Response(status, request=request, headers={"retry-after": "7"})
    return kind(message, response=response, body=None)


def test_the_request_streams_with_the_preset_s_reasoning_and_no_retries() -> None:
    brain, completions = brain_over([[text("Да?"), done()]])
    run_turn(brain, "Пепин", {})
    (request,) = completions.requests
    assert request["model"] == "openai/gpt-oss-120b" and request["stream"] is True
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"include_reasoning": False}
    assert request["tool_choice"] == "auto"
    assert request["messages"][0]["role"] == "system"
    assert "home" in request["messages"][0]["content"]
    assert [t["function"]["name"] for t in request["tools"]] == ["see", "end_conversation"]


def test_a_call_from_deltas_is_answered_with_a_tool_message_and_its_picture() -> None:
    rounds = [
        [text("Сейчас гляну. "), call(0, "c1", "se", None), call(0, None, "e", "{}"), done()],
        [text("Вижу красный круг."), done(1200, 6)],
    ]
    brain, completions = brain_over(rounds)
    image = {"ok": True, "image": Image(JPEG, "image/jpeg", 320, 240)}
    events = run_turn(brain, "что видишь?", {"see": image})
    assert [e.text for e in events if isinstance(e, Sentence)] == [
        "Сейчас гляну.",
        "Вижу красный круг.",
    ]
    use = next(e for e in events if isinstance(e, ToolUse))
    assert use.name == "see" and use.id == "c1"
    second = completions.requests[1]["messages"]
    assistant, tool, picture = second[-3:]
    assert assistant["tool_calls"][0]["function"] == {"name": "see", "arguments": "{}"}
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": tool["content"]}
    assert "attached" in tool["content"]
    assert picture["role"] == "user"
    assert picture["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    usage = [e for e in events if isinstance(e, Usage)]
    assert usage[-1].input == 1200 and usage[-1].usd == pytest.approx(
        (1200 * OSS.input + 6 * OSS.output) / 1e6
    )


def test_a_model_that_refuses_pictures_gets_the_call_answered_in_words() -> None:
    refused = http_error(openai.BadRequestError, 400, "messages[4].content must be a string")
    rounds = [
        [call(0, "c1", "see", "{}"), done()],
        refused,
        [text("Картинку не вижу."), done()],
    ]
    brain, completions = brain_over(rounds)
    run_turn(brain, "что видишь?", {"see": {"ok": True, "image": Image(JPEG)}})
    retried = completions.requests[2]["messages"]
    assert not any(isinstance(m.get("content"), list) for m in retried)
    assert NO_PICTURES in retried[-1]["content"]
    assert brain.pictures is False


def test_a_rate_limit_is_printed_and_fails_the_turn(capsys: pytest.CaptureFixture[str]) -> None:
    brain, _ = brain_over([http_error(openai.RateLimitError, 429, "rate limit")])
    with pytest.raises(openai.RateLimitError):
        run_turn(brain, "Пепин", {})
    assert "429 rate limit (retry after 7 s)" in capsys.readouterr().out


def test_required_forces_a_call_in_the_first_round_only() -> None:
    rounds = [[call(0, "c1", "see", "{}"), done()], [text("Вижу."), done()]]
    brain, completions = brain_over(rounds, tool_choice="required", max_tool_rounds=2)
    run_turn(brain, "посмотри на меня", {})
    assert [r["tool_choice"] for r in completions.requests] == ["required", "auto"]


def test_an_interrupt_keeps_the_call_and_the_words_heard() -> None:
    rounds = [[call(0, "c1", "see", "{}"), done()], [text("Вижу стол. На столе книга."), done()]]
    brain, _ = brain_over(rounds)
    run_turn(brain, "что видишь?", {})
    brain.interrupt(Played(sentences=1, partial="На"))
    roles = [m["role"] for m in brain.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert brain.history[-1]["content"] == "Вижу стол. На" + INTERRUPTED


def test_an_interrupt_while_a_call_runs_answers_it_as_not_awaited() -> None:
    brain, _ = brain_over([[call(0, "c1", "see", "{}"), done()]])
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
    assert brain.history[-1] == {"role": "tool", "tool_call_id": "c1", "content": NOT_AWAITED}


def test_compact_tools_keep_the_first_sentence() -> None:
    (small,) = compact([SEE])
    assert small["function"]["description"] == "One picture from the head camera."
    assert SEE["function"]["description"].endswith("what you see.")  # the original is untouched


def test_prewarm_opens_the_connection_without_tokens() -> None:
    brain, completions = brain_over([])
    warm = asyncio.run(brain.prewarm())
    assert warm is not None and warm.usd == 0 and completions.requests == []
