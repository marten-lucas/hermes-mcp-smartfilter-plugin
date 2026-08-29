"""Hermes MCP Smart Filter plugin."""

from . import schemas
from .tools import create_handler


def register(ctx):
    """Register the Smart Filter tool."""

    ctx.register_tool(
        name="semantic_tool_search",
        toolset="mcp-smart-filter",
        schema=schemas.SEMANTIC_TOOL_SEARCH,
        handler=create_handler(ctx),
    )