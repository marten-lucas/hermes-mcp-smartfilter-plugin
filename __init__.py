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
    has_override_capability = getattr(ctx, "has_capability", None)
    can_override = False

    if callable(has_override_capability):
        can_override = bool(ctx.has_capability("tools.override"))

    handler = create_handler(ctx)

    if can_override:
        logger.info(
            "[Smart-Filter] 'tools.override' capability granted. "
            "Registering 'tool_search' as native override."
        )
        ctx.register_tool(
            name="tool_search",
            toolset="tools",
            schema=schemas.TOOL_SEARCH_SCHEMA,
            handler=handler,
            override=True,
        )
    else:
        logger.warning(
            "[Smart-Filter] 'tools.override' capability not granted. "
            "Registering as auxiliary 'semantic_tool_search' tool. "
            "To replace native tool_search, run: "
            "hermes plugins enable mcp-smart-filter --allow-tool-override"
        )
        ctx.register_tool(
            name="semantic_tool_search",
            toolset="mcp-smart-filter",
            schema=schemas.SEMANTIC_TOOL_SEARCH,
            handler=handler,
        )