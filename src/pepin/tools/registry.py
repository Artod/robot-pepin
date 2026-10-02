"""The registry of the LLM's tools: plain Python functions, their signatures and docstrings.

A tool is a function whose FIRST parameter is the robot it acts through
(:class:`pepin.tools.robot.Robot`, injected, never shown to the model) and whose other
parameters are what the model fills in::

    @tool
    def go_to(robot: Robot, place: str) -> Result:
        '''Drive to a named place and wait until the drive ends.

        Args:
            place: the place's name, exactly as list_places gives it.
        '''

The signature and the docstring ARE the tool's schema: the text above ``Args:`` is the tool's
description, each ``name: text`` entry under it a parameter's, and the annotations its types
(``str``, ``int``, ``float``, ``bool``, a ``Literal`` of strings, any of them ``| None``). The
adapters render this one registry — :mod:`pepin.tools.schemas` as function-calling JSON,
:mod:`pepin.tools.mcp` as an MCP server — so adding a tool is writing one function.

Every tool returns a :data:`Result`: a small dict with ``ok``, its payload, and on failure a
``why`` in words the model can act on (:func:`ok`, :func:`fail`). A client that cannot reach its
service raises :class:`ToolError` carrying such words, and :meth:`Registry.call` turns it into
the failure: a tool body is written for the case where the owners answer.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

Result = dict[str, Any]
"""What every tool returns: ``ok``, the payload, and ``why`` when ``ok`` is false."""

F = TypeVar("F", bound=Callable[..., Result])

NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")  # a name every provider accepts
ARG = re.compile(r"^\s+(\w+):\s*(.*)$")  # "    place: the place's name" under Args:
SCALARS: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolError(Exception):
    """A failure meant for the model: ``why`` says what happened and what can be done."""

    def __init__(self, why: str) -> None:
        """``why``: one or two plain sentences the model can act on."""
        super().__init__(why)
        self.why = why


def ok(**payload: Any) -> Result:
    """A successful result carrying ``payload``."""
    return {"ok": True, **payload}


def fail(why: str, **payload: Any) -> Result:
    """A failed result: ``why`` in plain words, and whatever was learned on the way."""
    return {"ok": False, "why": why, **payload}


@dataclass(frozen=True)
class Image:
    """A picture a tool returns among its payload; adapters send it as an image, not as text."""

    data: bytes
    mime: str = "image/jpeg"
    width: int = 0
    height: int = 0

    def describe(self) -> str:
        """What stands in the result's text for the picture that travels beside it."""
        size = f"{self.width}x{self.height} " if self.width and self.height else ""
        return f"{size}{self.mime}, {len(self.data) / 1024:.0f} KB, attached"


def render(result: Result) -> tuple[str, list[Image]]:
    """A result as the JSON text a model reads and the pictures that go beside it."""
    images = [value for value in result.values() if isinstance(value, Image)]
    text = {k: v.describe() if isinstance(v, Image) else v for k, v in result.items()}
    return json.dumps(text, ensure_ascii=False, default=str), images


@dataclass(frozen=True)
class Param:
    """One argument the model fills in: its JSON schema, and how a value is checked."""

    name: str
    annotation: Any  # the resolved type, None stripped
    optional: bool  # the annotation admits None
    default: Any  # inspect.Parameter.empty when the model must give it
    description: str

    @property
    def required(self) -> bool:
        """Whether the model must give this argument."""
        return self.default is inspect.Parameter.empty

    def schema(self) -> dict[str, Any]:
        """The argument's JSON schema, with its description."""
        if typing.get_origin(self.annotation) is Literal:
            choices = list(typing.get_args(self.annotation))
            return {"type": "string", "enum": choices, "description": self.description}
        return {"type": SCALARS[self.annotation], "description": self.description}

    def coerce(self, value: Any) -> Any:
        """``value`` as this argument's type; :class:`ToolError` naming the argument otherwise.
        Numbers written as strings are taken: models send ``"1.5"`` now and then."""
        if value is None:
            if self.optional:
                return None
            raise ToolError(f"{self.name} may not be null")
        kind = self.annotation
        if typing.get_origin(kind) is Literal:
            choices = typing.get_args(kind)
            if value not in choices:
                raise ToolError(f"{self.name} must be one of {', '.join(map(str, choices))}")
            return value
        try:
            if kind is bool:
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    return value.lower() == "true"
                if isinstance(value, bool):
                    return value
                raise ValueError(value)
            if kind is int and not isinstance(value, bool):
                number = float(value)
                if number.is_integer():
                    return int(number)
                raise ValueError(value)
            if kind is float and not isinstance(value, bool):
                return float(value)
            if kind is str and isinstance(value, (str, int, float)):
                return str(value)
        except (TypeError, ValueError):
            pass
        raise ToolError(f"{self.name} must be {SCALARS[kind]}, not {value!r}")


@dataclass(frozen=True)
class Tool:
    """One registered tool: its name, what the model reads, its arguments, and the function."""

    name: str
    description: str
    params: tuple[Param, ...]
    fn: Callable[..., Result]
    moves: bool = False  # it sets the robot in motion: a caller that gives up must halt it
    latency: LatencyTracker = field(
        compare=False, repr=False, default_factory=lambda: LatencyTracker("tool")
    )

    def input_schema(self) -> dict[str, Any]:
        """The JSON schema of the arguments, as every provider's function calling takes it."""
        return {
            "type": "object",
            "properties": {param.name: param.schema() for param in self.params},
            "required": [param.name for param in self.params if param.required],
        }

    def signature(self) -> inspect.Signature:
        """The arguments the model sees as a Python signature (the robot left out), resolved
        types included: what an adapter that reads signatures (the MCP server) is given."""
        return inspect.Signature(
            [
                inspect.Parameter(
                    param.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=param.default,
                    annotation=param.annotation | None if param.optional else param.annotation,
                )
                for param in self.params
            ]
        )

    def bind(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The model's arguments checked and typed, defaults filled; :class:`ToolError` naming
        the missing, unknown or mistyped one."""
        known = {param.name: param for param in self.params}
        unknown = sorted(set(arguments) - set(known))
        if unknown:
            takes = ", ".join(known) or "no arguments"
            raise ToolError(f"{self.name} has no argument {', '.join(unknown)}; it takes {takes}")
        bound: dict[str, Any] = {}
        for name, param in known.items():
            if name in arguments:
                bound[name] = param.coerce(arguments[name])
            elif param.required:
                raise ToolError(f"{self.name} needs {name}: {param.description}")
            else:
                bound[name] = param.default
        return bound


class Registry:
    """The tools by name, in the order they were registered."""

    def __init__(self) -> None:
        """An empty registry."""
        self._tools: dict[str, Tool] = {}

    @overload
    def tool(self, fn: F, /) -> F: ...

    @overload
    def tool(self, fn: None = None, /, *, moves: bool = False) -> Callable[[F], F]: ...

    def tool(self, fn: F | None = None, /, *, moves: bool = False) -> F | Callable[[F], F]:
        """Register ``fn`` as a tool (``@tool``, or ``@tool(moves=True)`` for one that sets the
        robot in motion); the function itself is returned unchanged."""

        def register(function: F) -> F:
            built = build_tool(function, moves=moves)
            if built.name in self._tools:
                raise ValueError(f"a tool named {built.name!r} is registered already")
            self._tools[built.name] = built
            return function

        return register(fn) if fn is not None else register

    def __iter__(self) -> Iterator[Tool]:
        """The tools, in registration order."""
        return iter(self._tools.values())

    def __len__(self) -> int:
        """How many tools there are."""
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """Whether a tool of that name is registered."""
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        """The tool of that name; KeyError when there is none."""
        return self._tools[name]

    def call(self, name: str, arguments: Mapping[str, Any], robot: object) -> Result:
        """Run one tool as a model asked for it; always a :data:`Result`, never a raise (but for
        KeyboardInterrupt, which the caller turns into a halt). Logged with its latency."""
        chosen = self._tools.get(name)
        if chosen is None:
            return fail(f"there is no tool {name!r}; the tools are {', '.join(self._tools)}")
        started = time.perf_counter()
        try:
            result = chosen.fn(robot, **chosen.bind(arguments))
        except ToolError as error:
            result = fail(error.why)
        except Exception as error:  # a broken tool must reach the model as a failure, not a crash
            logger.exception("tool %s broke", name)
            result = fail(f"{name} broke inside: {type(error).__name__}: {error}")
        elapsed = time.perf_counter() - started
        chosen.latency.add(elapsed)
        logger.info(
            "tool %s(%s) -> %s in %.0f ms",
            name,
            ", ".join(f"{k}={v!r}" for k, v in arguments.items()),
            "ok" if result.get("ok") else f"failed: {result.get('why')}",
            elapsed * 1000,
        )
        return result


def build_tool(fn: Callable[..., Result], *, moves: bool = False) -> Tool:
    """A :class:`Tool` from a function: its name, docstring and typed signature. Raises
    ``TypeError`` for what no schema can say — a type outside the supported ones, an argument
    without a description, a function without a summary — so a bad tool fails at import."""
    name = fn.__name__
    if not NAME.match(name):
        raise TypeError(f"{name!r}: a tool name is lower_snake_case, at most 64 characters")
    summary, described = parse_docstring(inspect.getdoc(fn) or "")
    if not summary:
        raise TypeError(f"{name}: the docstring's summary is what the model reads; write one")
    hints = typing.get_type_hints(fn)
    parameters = list(inspect.signature(fn).parameters.values())
    if not parameters:
        raise TypeError(f"{name}: the first parameter is the robot the tool acts through")
    params = []
    for parameter in parameters[1:]:  # the first is the robot
        if parameter.name not in described:
            raise TypeError(f"{name}: argument {parameter.name!r} has no description under Args:")
        annotation, optional = _unwrap_optional(hints.get(parameter.name))
        if annotation not in SCALARS and typing.get_origin(annotation) is not Literal:
            raise TypeError(f"{name}: argument {parameter.name!r} has an unsupported type")
        params.append(
            Param(
                parameter.name, annotation, optional, parameter.default, described[parameter.name]
            )
        )
    extra = sorted(set(described) - {param.name for param in params})
    if extra:
        raise TypeError(f"{name}: Args: describes {', '.join(extra)}, which it does not take")
    return Tool(name, summary, tuple(params), fn, moves, LatencyTracker(f"tool.{name}"))


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """``(T, True)`` for ``T | None``, ``(T, False)`` for a plain ``T``."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0], True
    return annotation, False


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """A docstring as ``(summary, {argument: description})``: the summary is everything above
    ``Args:`` with its lines joined (paragraphs kept), each ``name: text`` entry indented under
    ``Args:`` an argument, its indented continuation lines joined to it."""
    head, marker, tail = doc.partition("\nArgs:\n")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", head.strip())]
    summary = "\n\n".join(p for p in paragraphs if p)
    described: dict[str, str] = {}
    current: str | None = None
    for line in tail.splitlines() if marker else []:
        if line and not line[0].isspace():
            break  # the section after Args: (Returns:, ...) is not an argument
        entry = ARG.match(line)
        if entry and len(line) - len(line.lstrip()) <= 4:
            current = entry.group(1)
            described[current] = entry.group(2).strip()
        elif current is not None and line.strip():
            described[current] += " " + line.strip()
    return summary, described


TOOLS = Registry()
"""THE registry: every tool of :mod:`pepin.tools` is registered here, at import."""

tool = TOOLS.tool
"""Register a function in :data:`TOOLS`: ``@tool`` or ``@tool(moves=True)``."""
