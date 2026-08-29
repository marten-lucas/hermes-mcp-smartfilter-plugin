"""Hermes FastEmbed Smart Search plugin."""

import logging

try:
    from . import schemas
    from .tools import create_handler
except (ImportError, ValueError):
    import schemas  # type: ignore
    from tools import create_handler  # type: ignore

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

__version__ = "2.0.0"


def register(ctx):
    """Register the FastEmbed tool_search handler in Hermes Agent."""
    ctx.register_tool(
        name="tool_search",
        toolset="tools",
        schema=schemas.TOOL_SEARCH_SCHEMA,
        handler=create_handler(ctx),
        override=True,
    )