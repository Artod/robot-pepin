"""pepin.tools.registry and pepin.tools.schemas: a function's signature and docstring are its
tool schema, one call path checks the model's arguments and turns every failure into a ``why``,
and the three providers' shapes are renderings of the same schema."""

from __future__ import annotations

from typing import Literal

import pytest

from pepin.tools.registry import (
    Image,
    Registry,
    Result,
    ToolError,
    build_tool,
    fail,
    ok,
    parse_docstring,
    render,
)
from pepin.tools.schemas import (
    anthropic_tool_result,
    anthropic_tools,
    gemini_function_declarations,
    openai_tools,
)


def sample_registry() -> Registry:
    """Three small tools covering every supported type."""
    registry = Registry()

    @registry.tool
    def wave(robot: object, times: int, speed: float = 1.0, loud: bool | None = None) -> Result:
        """Wave the arm.

        A second paragraph of the summary.

        Args:
            times: how many waves.
            speed: how fast, in waves a second;
                a continuation line.
            loud: whether to say hello too.

        Returns:
            nothing a parser should read as an argument.
        """
        return ok(robot=robot, times=times, speed=speed, loud=loud)

    @registry.tool(moves=True)
    def roll(robot: object, way: Literal["left", "right"]) -> Result:
        """Roll one way.

        Args:
            way: left or right.
        """
        if way == "left":
            raise ToolError("the left wheel is off: roll right")
        return ok(way=way)

    @registry.tool
    def crash(robot: object) -> Result:
        """Always breaks."""
        raise RuntimeError("boom")

    return registry


def test_the_schema_is_the_signature_and_the_docstring() -> None:
    wave = sample_registry()["wave"]
    assert wave.description == "Wave the arm.\n\nA second paragraph of the summary."
    assert wave.input_schema() == {
        "type": "object",
        "properties": {
            "times": {"type": "integer", "description": "how many waves."},
            "speed": {
                "type": "number",
                "description": "how fast, in waves a second; a continuation line.",
            },
            "loud": {"type": "boolean", "description": "whether to say hello too."},
        },
        "required": ["times"],
    }
    roll = sample_registry()["roll"]
    assert roll.moves and not wave.moves
    assert roll.input_schema()["properties"]["way"]["enum"] == ["left", "right"]


def test_a_call_checks_types_fills_defaults_and_hands_the_robot_first() -> None:
    registry = sample_registry()
    robot = object()
    result = registry.call("wave", {"times": "3", "speed": "1.5", "loud": "true"}, robot)
    assert result == {"ok": True, "robot": robot, "times": 3, "speed": 1.5, "loud": True}
    assert registry.call("wave", {"times": 2.0}, robot)["speed"] == 1.0
    assert registry.call("wave", {"times": 2, "loud": None}, robot)["loud"] is None


@pytest.mark.parametrize(
    ("name", "arguments", "why"),
    [
        ("fly", {}, "there is no tool 'fly'; the tools are wave, roll, crash"),
        ("wave", {}, "wave needs times: how many waves."),
        ("wave", {"times": 1, "height": 2}, "wave has no argument height; it takes times, speed"),
        ("wave", {"times": 1.5}, "times must be integer, not 1.5"),
        ("wave", {"times": True}, "times must be integer, not True"),
        ("wave", {"times": 1, "loud": "maybe"}, "loud must be boolean, not 'maybe'"),
        ("roll", {"way": "up"}, "way must be one of left, right"),
        ("roll", {"way": None}, "way may not be null"),
        ("roll", {"way": "left"}, "the left wheel is off: roll right"),
        ("crash", {}, "crash broke inside: RuntimeError: boom"),
    ],
)
def test_every_failure_comes_back_as_a_why(
    name: str, arguments: dict[str, object], why: str
) -> None:
    result = sample_registry().call(name, arguments, object())
    assert result["ok"] is False
    assert str(result["why"]).startswith(why)


def test_a_keyboard_interrupt_is_not_swallowed() -> None:
    """Ctrl-C belongs to the caller, who halts the robot with it."""
    registry = Registry()

    @registry.tool
    def wait(robot: object) -> Result:
        """Wait forever."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        registry.call("wait", {}, object())


def test_a_call_is_timed() -> None:
    registry = sample_registry()
    registry.call("wave", {"times": 1}, object())
    assert registry["wave"].latency.count == 1


def undescribed(robot: object, times: int) -> Result:
    """Wave."""
    return ok()


def listed(robot: object, times: list[int]) -> Result:
    """Wave.

    Args:
        times: when.
    """
    return ok()


def silent(robot: object) -> Result:
    return ok()


def over_described(robot: object) -> Result:
    """Wave.

    Args:
        times: when.
    """
    return ok()


def no_robot() -> Result:
    """Wave."""
    return ok()


def CamelCase(robot: object) -> Result:  # noqa: N802 -- the name is what is under test
    """Wave."""
    return ok()


@pytest.mark.parametrize(
    ("fn", "complaint"),
    [
        (undescribed, "argument 'times' has no description"),
        (listed, "unsupported type"),
        (silent, "summary"),
        (over_described, "does not take"),
        (no_robot, "first parameter is the robot"),
        (CamelCase, "lower_snake_case"),
    ],
)
def test_a_tool_no_schema_can_describe_fails_at_import(fn: object, complaint: str) -> None:
    with pytest.raises(TypeError, match=complaint):
        build_tool(fn)  # type: ignore[arg-type]


def test_a_name_is_registered_once() -> None:
    registry = sample_registry()
    with pytest.raises(ValueError, match="registered already"):

        @registry.tool
        def wave(robot: object) -> Result:
            """Another wave."""
            return ok()


def test_the_docstring_parser_reads_only_the_args_section() -> None:
    summary, args = parse_docstring(
        "One.\n\nTwo\nlines.\n\nArgs:\n    a: first\n        more\n    b: second\n"
        "\nReturns:\n    c: no\n"
    )
    assert summary == "One.\n\nTwo lines."
    assert args == {"a": "first more", "b": "second"}
    assert parse_docstring("Only a summary.") == ("Only a summary.", {})


def test_a_picture_travels_beside_the_text_not_in_it() -> None:
    picture = Image(b"\xff\xd8" + b"x" * 2048, "image/jpeg", 800, 600)
    text, images = render(ok(image=picture, pan_deg=10.0))
    assert images == [picture]
    assert '"image": "800x600 image/jpeg, 2 KB, attached"' in text
    assert "\\xff" not in text


def test_the_anthropic_result_carries_text_image_and_the_error_flag() -> None:
    picture = Image(b"abc", "image/png")
    block = anthropic_tool_result("toolu_1", ok(image=picture))
    assert block["tool_use_id"] == "toolu_1" and "is_error" not in block
    assert [part["type"] for part in block["content"]] == ["text", "image"]
    assert block["content"][1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "YWJj",
    }
    failed = anthropic_tool_result("toolu_2", fail("no such place"))
    assert failed["is_error"] is True
    assert '"why": "no such place"' in failed["content"][0]["text"]


def test_every_provider_shape_renders_the_same_schema() -> None:
    registry = sample_registry()
    anthropic = anthropic_tools(registry)
    openai = openai_tools(registry)
    gemini = gemini_function_declarations(registry)
    for a, o, g in zip(anthropic, openai, gemini, strict=True):
        assert a["name"] == o["function"]["name"] == g["name"]
        assert a["description"] == o["function"]["description"] == g["description"]
        assert a["input_schema"] == o["function"]["parameters"]
        assert g.get("parameters", a["input_schema"]) == a["input_schema"]
    assert "parameters" not in gemini[2]  # crash() takes nothing: Gemini refuses empty objects
