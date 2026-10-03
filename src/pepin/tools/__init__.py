"""The ONE place where an LLM's tools for Pepin are defined.

Plain functions with typed signatures and docstrings (:mod:`pepin.tools.registry`), each acting
through a :class:`Robot` whose clients speak the owners' sockets (:mod:`pepin.tools.clients`).
Two thin adapters render the same registry: :mod:`pepin.tools.mcp` (an MCP server, so Claude
Code or Claude Desktop drive the robot with no LLM code of ours) and
:mod:`pepin.tools.schemas` (function-calling JSON for our own loop).

Adding a tool is writing one function with ``@tool`` in one of the modules imported below — or
in a new module added to that list. The registry, the adapters and their tests do not change.

    uv run python -m pepin.tools.mcp          # the MCP server, on stdio
"""

from pepin.tools import drive, face, head, memory, speech, status  # noqa: F401 -- they register
from pepin.tools.registry import TOOLS, Image, Registry, Result, ToolError, fail, ok, tool
from pepin.tools.robot import Endpoints, Robot

__all__ = [
    "TOOLS",
    "Endpoints",
    "Image",
    "Registry",
    "Result",
    "Robot",
    "ToolError",
    "fail",
    "ok",
    "tool",
]
