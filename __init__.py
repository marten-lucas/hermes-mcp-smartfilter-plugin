"""Hermes FastEmbed Smart Search plugin."""

import logging

try:
    from . import schemas
    from .tools import create_handler, _write_audit_log
except (ImportError, ValueError):
    import schemas  # type: ignore
    from tools import create_handler, _write_audit_log  # type: ignore

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

__version__ = "2.1.0"


def _patch_native_bridge(handler):
    """
    Hermes 0.20.6 routes tool_search bridge calls directly in model_tools.py to
    tools.tool_search.dispatch_tool_search, bypassing the ToolRegistry.
    Monkey-patching dispatch_tool_search ensures FastEmbed runs whenever the
    LLM issues a tool_search call.
    """
    try:
        import tools.tool_search as ts

        if getattr(ts, "_fastembed_patched", False):
            return

        orig_dispatch = ts.dispatch_tool_search

        def fastembed_dispatch(args, *, current_tool_defs, config=None):
            try:
                # Run FastEmbed handler passing current_tool_defs for discovery
                return handler(args=args, tools=current_tool_defs)
            except Exception as exc:
                logger.error("[Smart-Filter] FastEmbed dispatch failed, falling back to BM25: %s", exc)
                return orig_dispatch(args, current_tool_defs=current_tool_defs, config=config)

        ts.dispatch_tool_search = fastembed_dispatch
        ts._fastembed_patched = True
        _write_audit_log("[PLUGIN LOADED] Native tools.tool_search.dispatch_tool_search successfully monkey-patched with FastEmbed.")
        logger.info("[Smart-Filter] Native tool_search monkey-patched successfully.")
    except Exception as exc:
        _write_audit_log(f"[PLUGIN LOADED] Failed to monkey-patch tools.tool_search: {exc}")
        logger.warning("[Smart-Filter] Could not monkey-patch tools.tool_search: %s", exc)


def register(ctx):
    """Register the FastEmbed tool_search handler in Hermes Agent.

    Bevorzugt Override des nativen tool_search (Capability tools.override).
    Schlägt die Registrierung mit override=True fehl (kein Consent, ältere
    Hermes-Version), wird unter dem eigenen Namen semantic_tool_search
    registriert statt still zu shadowen.
    """
    handler = create_handler(ctx)
    _patch_native_bridge(handler)

    try:
        _write_audit_log("[PLUGIN LOADED] Registering tool_search (override=True)")
        ctx.register_tool(
            name="tool_search",
            toolset="tools",
            schema=schemas.TOOL_SEARCH_SCHEMA,
            handler=handler,
            override=True,
        )
    except Exception as exc:
        _write_audit_log(f"[PLUGIN LOADED] Fallback to semantic_tool_search due to: {exc}")
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