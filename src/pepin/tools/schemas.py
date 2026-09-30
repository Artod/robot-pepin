"""The registry as function-calling JSON: the shapes Anthropic, OpenAI and Gemini take.

All three are renderings of the same :meth:`pepin.tools.registry.Tool.input_schema`; nothing
here decides what a tool is. A result goes back as :func:`anthropic_tool_result` (text and the
pictures beside it) or, for the others, as :func:`pepin.tools.registry.render`'s JSON text.
"""

from __future__ import annotations

import base64
from typing import Any

from pepin.tools.registry import Registry, Result, render


def anthropic_tools(registry: Registry) -> list[dict[str, Any]]:
    """The Messages API's ``tools``: name, description, input_schema."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema()}
        for t in registry
    ]


def openai_tools(registry: Registry) -> list[dict[str, Any]]:
    """The Chat Completions ``tools``: functions with JSON-schema parameters."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema(),
            },
        }
        for t in registry
    ]


def gemini_function_declarations(registry: Registry) -> list[dict[str, Any]]:
    """Gemini's ``function_declarations``; a tool without arguments carries no ``parameters``
    (Gemini refuses an object schema with no properties)."""
    declarations = []
    for t in registry:
        declaration: dict[str, Any] = {"name": t.name, "description": t.description}
        if t.params:
            declaration["parameters"] = t.input_schema()
        declarations.append(declaration)
    return declarations


def anthropic_tool_result(tool_use_id: str, result: Result) -> dict[str, Any]:
    """One ``tool_result`` block: the result's JSON text, then its pictures as image blocks;
    ``is_error`` when the tool failed."""
    text, images = render(result)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content += [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime,
                "data": base64.standard_b64encode(image.data).decode("ascii"),
            },
        }
        for image in images
    ]
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if not result.get("ok"):
        block["is_error"] = True
    return block
