"""Hermes Ollama-backed semantic Smart Search plugin."""

import logging

try:
    from . import schemas
    from .tools import _write_audit_log, create_handler, create_pre_llm_hook
except (ImportError, ValueError):
    import schemas  # type: ignore
    from tools import (  # type: ignore
        _write_audit_log,
        create_handler,
        create_pre_llm_hook,
    )

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

__version__ = "2.5.0"


def _pre_warm_engine_in_background(ctx):
    """
    Pre-warm the Ollama embedding model and initial tool catalog in a background
    thread so the first user turn does not pay the catalog-indexing cost.
    """
    import threading

    def _worker():
        try:
            from .tools import _ENGINE, _extract_tool_info, _get_available_tools
        except (ImportError, ValueError):
            from tools import (  # type: ignore
                _ENGINE,
                _extract_tool_info,
                _get_available_tools,
            )

        try:
            logger.info(
                "[Smart-Filter] Pre-warming Ollama model and tool catalog in background..."
            )
            _write_audit_log("[PRE-WARM] Ollama catalog pre-warm started.")
            raw_tools = _get_available_tools(ctx, {})
            extracted = []
            for item in raw_tools:
                info = _extract_tool_info(item)
                if info["name"]:
                    extracted.append(info)
            if extracted and _ENGINE._sync_tool_catalog(extracted):
                logger.info(
                    "[Smart-Filter] Background pre-warm completed (%d tools indexed).",
                    len(extracted),
                )
                _write_audit_log(
                    f"[PRE-WARM] Ollama embeddings ready for {len(extracted)} tools."
                )
            elif extracted:
                logger.info(
                    "[Smart-Filter] Background pre-warm deferred; keyword fallback remains available."
                )
                _write_audit_log(
                    "[PRE-WARM] Ollama unavailable; keyword fallback remains active."
                )
        except Exception as exc:
            logger.warning(
                "[Smart-Filter] Background pre-warm error (%s).", type(exc).__name__
            )
            _write_audit_log(
                f"[PRE-WARM] Ollama pre-warm failed ({type(exc).__name__})."
            )

    t = threading.Thread(target=_worker, daemon=True, name="smart-filter-prewarm")
    t.start()


def _patch_native_bridge(handler):
    """
    Hermes 0.20.6 routes tool_search bridge calls directly in model_tools.py to
    tools.tool_search.dispatch_tool_search, bypassing the ToolRegistry.
    Monkey-patching dispatch_tool_search ensures the semantic search engine runs whenever the
    LLM issues a tool_search call.
    """
    try:
        import tools.tool_search as ts

        if getattr(ts, "_smart_filter_patched", False):
            return

        orig_dispatch = ts.dispatch_tool_search

        def smart_filter_dispatch(
            args, *, current_tool_defs, config=None, connector_search=None
        ):
            try:
                # Pass current_tool_defs so the plugin can search the full deferred catalog.
                return handler(args=args, tools=current_tool_defs)
            except Exception as exc:
                logger.error(
                    "[Smart-Filter] Dispatch failed, falling back to Hermes BM25: %s",
                    exc,
                )
                # Hermes >= v0.21.4 added a ``connector_search`` keyword to
                # tools.tool_search.dispatch_tool_search. Forward it when the
                # native dispatcher accepts it; older 0.20.x signatures lack it.
                try:
                    return orig_dispatch(
                        args,
                        current_tool_defs=current_tool_defs,
                        config=config,
                        connector_search=connector_search,
                    )
                except TypeError:
                    return orig_dispatch(
                        args, current_tool_defs=current_tool_defs, config=config
                    )

        ts.dispatch_tool_search = smart_filter_dispatch
        ts._smart_filter_patched = True
        _write_audit_log(
            "[PLUGIN LOADED] Native tools.tool_search.dispatch_tool_search patched with Ollama semantic search."
        )
        logger.info("[Smart-Filter] Native tool_search monkey-patched successfully.")
    except Exception as exc:
        _write_audit_log(
            f"[PLUGIN LOADED] Failed to monkey-patch tools.tool_search: {exc}"
        )
        logger.warning(
            "[Smart-Filter] Could not monkey-patch tools.tool_search: %s", exc
        )


def register(ctx):
    """Register the Ollama semantic tool_search handler and pre_llm_call hook in Hermes Agent.

    Bevorzugt Override des nativen tool_search (Capability tools.override).
    Schlägt die Registrierung mit override=True fehl (kein Consent, ältere
    Hermes-Version), wird unter dem eigenen Namen semantic_tool_search
    registriert statt still zu shadowen.
    """
    handler = create_handler(ctx)
    _patch_native_bridge(handler)
    _pre_warm_engine_in_background(ctx)

    # 1. Register pre_llm_call hook to surface relevant tools directly into context
    try:
        pre_llm_callback = create_pre_llm_hook(ctx)
        ctx.register_hook("pre_llm_call", pre_llm_callback)
        _write_audit_log("[PLUGIN LOADED] Registered 'pre_llm_call' hook.")
        logger.info("[Smart-Filter] Registered 'pre_llm_call' hook.")
    except Exception as exc:
        _write_audit_log(f"[PLUGIN LOADED] Failed to register 'pre_llm_call': {exc}")
        logger.warning("[Smart-Filter] Failed to register 'pre_llm_call' hook: %s", exc)

    # 2. Register tool_search handler
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
        _write_audit_log(
            f"[PLUGIN LOADED] Fallback to semantic_tool_search due to: {exc}"
        )
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
