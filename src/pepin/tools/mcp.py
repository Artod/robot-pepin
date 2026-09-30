"""The registry as an MCP server, so Claude Code or Claude Desktop drive Pepin with no LLM code.

Run it (stdio, the way both clients start a local server)::

    uv run python -m pepin.tools.mcp [--board HOST] [--goal-host HOST] [--world URL]

It is FastMCP — named ``MCPServer`` since the SDK's version 2 — given one MCP tool per registry
tool: the description and the input schema are the registry's own (:meth:`Tool.input_schema`),
the call is :meth:`Registry.call`, and a result goes out as its JSON text with any picture as an
image block, ``isError`` when the tool failed.

Tools run on worker threads, so a ``cancel`` is served while a ``go_to`` blocks. A call to a
tool that moves the robot which the client abandons (the person pressed Escape) halts the robot:
nothing a model started keeps moving after the model stopped waiting for it.
"""

from __future__ import annotations

import argparse
import base64
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools import Tool as McpTool
from mcp_types import CallToolResult, ContentBlock, ImageContent, TextContent

from pepin.log import setup_logging
from pepin.tools import TOOLS, Endpoints, Registry, Result, Robot
from pepin.tools.registry import Tool, render

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Pepin is a small home robot: a wheeled cart with a camera head, in one flat. Use"
    " list_places for the names go_to takes; go_to blocks until the drive ends and says how it"
    " ended; cancel stops every drive at once and is always safe. look points the head, see"
    " returns its picture, find/recall/map_tree read the robot's memory of what is where."
    " Every tool answers ok, and on failure why."
)
REPO = Path(__file__).resolve().parents[3]


def build_server(robot: Robot, registry: Registry = TOOLS) -> MCPServer:
    """An MCP server whose tools are ``registry``'s, acting through ``robot``."""
    return MCPServer(
        "pepin",
        instructions=INSTRUCTIONS,
        tools=[mcp_tool(tool, registry, robot) for tool in registry],
    )


def mcp_tool(tool: Tool, registry: Registry, robot: Robot) -> McpTool:
    """One registry tool as an MCP tool: the SDK validates the arguments against the tool's own
    signature, and advertises the registry's schema, not one of its own making."""

    async def run(**arguments: Any) -> CallToolResult:
        return to_mcp(await call(registry, tool, arguments, robot))

    run.__name__ = tool.name
    run.__doc__ = tool.description
    run.__signature__ = tool.signature().replace(return_annotation=CallToolResult)  # type: ignore[attr-defined]
    built = McpTool.from_function(
        run, name=tool.name, description=tool.description, structured_output=False
    )
    return built.model_copy(update={"parameters": tool.input_schema()})


async def call(registry: Registry, tool: Tool, arguments: dict[str, Any], robot: Robot) -> Result:
    """Run a tool on a worker thread. When the caller gives up on a tool that moves the robot,
    the robot is halted before the cancellation goes on."""
    run: Callable[[], Result] = functools.partial(registry.call, tool.name, arguments, robot)
    try:
        return await anyio.to_thread.run_sync(run, abandon_on_cancel=tool.moves)
    except anyio.get_cancelled_exc_class():
        if tool.moves:
            with anyio.CancelScope(shield=True):
                said = await anyio.to_thread.run_sync(robot.halt)
            logger.warning("%s abandoned by the client: %s", tool.name, said)
        raise


def to_mcp(result: Result) -> CallToolResult:
    """A result as MCP content: its JSON text, then its pictures."""
    text, images = render(result)
    content: list[ContentBlock] = [TextContent(text=text)]
    content += [
        ImageContent(data=base64.standard_b64encode(i.data).decode("ascii"), mime_type=i.mime)
        for i in images
    ]
    return CallToolResult(content=content, is_error=not result.get("ok"))


def main(argv: list[str] | None = None) -> None:
    """Parse the sockets' hosts, log to logs/ and stderr (stdout is the protocol), serve."""
    defaults = Endpoints.from_env()
    parser = argparse.ArgumentParser(prog="pepin.tools.mcp", description=__doc__.split("\n")[0])
    parser.add_argument("--board", default=defaults.board, help="the board (base, audio, camera)")
    parser.add_argument("--goal-host", default=defaults.goal_host, help="the goal server's host")
    parser.add_argument("--world", default=defaults.world_url, help="the memory service's URL")
    parser.add_argument(
        "--transport", choices=("stdio", "streamable-http"), default="stdio", help="MCP transport"
    )
    parser.add_argument("--port", type=int, default=8799, help="streamable-http's port")
    args = parser.parse_args(argv)
    setup_logging("tools_mcp", log_dir=REPO / "logs")
    endpoints = Endpoints(board=args.board, goal_host=args.goal_host, world_url=args.world)
    logger.info("pepin tools over MCP (%s): %s, %d tools", args.transport, endpoints, len(TOOLS))
    server = build_server(Robot.connect(endpoints))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
