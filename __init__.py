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
    can_override = False
    has_cap = getattr(ctx, "has_capability", None)
    if callable(has_cap):
        can_override = bool(ctx.has_capability("tools.override"))

    if not can_override:
        can_override = bool(getattr(ctx, "allow_tool_override", False))

    handler = create_handler(ctx)

    if can_override:
        ctx.register_tool(
            name="tool_search",
            toolset="tools",
            schema=schemas.TOOL_SEARCH_SCHEMA,
            handler=handler,
            override=True,
        )
    else:
        # Fallback to registering as tool_search with override=True if allowed,
        # or degrade gracefully to semantic_tool_search if host disallows tool override
        try:
            ctx.register_tool(
                name="tool_search",
                toolset="tools",
                schema=schemas.TOOL_SEARCH_SCHEMA,
                handler=handler,
                override=True,
            )
        except Exception as exc:
            logger.warning(
                "[Smart-Filter] Cannot override 'tool_search' (%s). "
                "Registering as 'semantic_tool_search'. "
                "Set plugins.entries.mcp-smart-filter.allow_tool_override: true in config.yaml",
                exc,
            )
            ctx.register_tool(
                name="semantic_tool_search",
                toolset="mcp-smart-filter",
                schema=schemas.SEMANTIC_TOOL_SEARCH,
                handler=handler,
            )