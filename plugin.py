import json
import logging
import os

import requests


logger = logging.getLogger("hermes.plugins.mcp_smart_filter")


TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": (
        "Search for available tools by semantic query or keywords. "
        "Use this when you need a tool for a specific task but its full schema "
        "is not yet visible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Semantic search query or keywords describing what action "
                    "you want to perform."
                ),
            },
        },
        "required": ["query"],
    },
}


def _extract_tool_info(tool):
    """Extract name and description from a Hermes tool definition."""
    if not isinstance(tool, dict):
        return {
            "name": "",
            "description": "",
        }

    function = tool.get("function")

    if isinstance(function, dict):
        source = function
    else:
        source = tool

    return {
        "name": str(source.get("name", "")),
        "description": str(source.get("description", "")),
    }


def _get_available_tools(ctx, kwargs):
    """
    Try to obtain the current Hermes tool catalogue.

    Different Hermes versions expose this differently, so keep the
    lookup defensive.
    """

    # Some Hermes call paths may provide the tools directly.
    tools = kwargs.get("tools")

    if tools:
        return list(tools)

    tools = kwargs.get("available_tools")

    if tools:
        return list(tools)

    # Try PluginContext's manager.
    manager = getattr(ctx, "_manager", None)

    if manager is not None:
        get_tools = getattr(manager, "get_tools", None)

        if callable(get_tools):
            try:
                result = get_tools()

                if isinstance(result, dict):
                    return list(result.values())

                return list(result or [])

            except Exception:
                logger.exception(
                    "[Smart-Filter] Failed to read tools "
                    "from ctx._manager"
                )

    # Try a public get_tools() method.
    get_tools = getattr(ctx, "get_tools", None)

    if callable(get_tools):
        try:
            result = get_tools()

            if isinstance(result, dict):
                return list(result.values())

            return list(result or [])

        except Exception:
            logger.exception(
                "[Smart-Filter] Failed to read tools "
                "from plugin context"
            )

    return []


def register(ctx):
    """Register the Smart Filter tool_search override."""

    service_url = os.environ.get(
        "SMART_FILTER_SERVICE_URL",
        "",
    ).rstrip("/")

    api_key = os.environ.get(
        "SMART_FILTER_API_KEY",
        "",
    )

    debug_mode = os.environ.get(
        "SMART_FILTER_DEBUG",
        "false",
    ).lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    if debug_mode:
        logger.setLevel(logging.DEBUG)

    try:
        max_k = int(
            os.environ.get(
                "SMART_FILTER_MAX_K",
                "8",
            )
        )

        min_k = int(
            os.environ.get(
                "SMART_FILTER_MIN_K",
                "1",
            )
        )

        min_score = float(
            os.environ.get(
                "SMART_FILTER_MIN_SCORE",
                "0.25",
            )
        )

        timeout = float(
            os.environ.get(
                "SMART_FILTER_TIMEOUT",
                "2.5",
            )
        )

    except ValueError as exc:
        logger.warning(
            "[Smart-Filter] Invalid configuration: %s. "
            "Using defaults.",
            exc,
        )

        max_k = 8
        min_k = 1
        min_score = 0.25
        timeout = 2.5

    if not service_url:
        logger.warning(
            "[Smart-Filter] SMART_FILTER_SERVICE_URL "
            "is not configured."
        )

    def handle_semantic_tool_search(args=None, **kwargs):
        """
        Handle tool_search requests.

        Hermes handlers receive the tool-call arguments as the first
        positional argument and may receive additional context via kwargs.
        """

        args = args or {}

        query = (
            args.get("query")
            or args.get("queries")
            or ""
        )

        if isinstance(query, list):
            query = " ".join(
                str(item)
                for item in query
            )

        query_str = str(query).strip()

        if not query_str:
            return json.dumps(
                {
                    "matched_tools": [],
                    "count": 0,
                    "error": "Empty query",
                }
            )

        available_tools = _get_available_tools(
            ctx,
            kwargs,
        )

        payload_tools = []

        for item in available_tools:
            tool_info = _extract_tool_info(item)

            if tool_info["name"]:
                payload_tools.append(tool_info)

        payload = {
            "query": query_str,
            "tools": payload_tools,
            "max_k": max_k,
            "min_k": min_k,
            "min_score": min_score,
        }

        headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            headers["X-API-Key"] = api_key

        if not service_url:
            return json.dumps(
                {
                    "matched_tools": [],
                    "count": 0,
                    "error": (
                        "SMART_FILTER_SERVICE_URL "
                        "is not configured"
                    ),
                }
            )

        try:
            logger.debug(
                "[Smart-Filter] Searching for %r "
                "among %d tools",
                query_str,
                len(payload_tools),
            )

            response = requests.post(
                f"{service_url}/filter",
                json=payload,
                headers=headers,
                timeout=timeout,
            )

            response.raise_for_status()

            response_data = response.json()

            matched_items = response_data.get(
                "tools",
                [],
            )

            matched_names = []

            for item in matched_items:
                if (
                    isinstance(item, dict)
                    and item.get("name")
                ):
                    matched_names.append(
                        item["name"]
                    )

                elif isinstance(item, str):
                    matched_names.append(item)

            logger.info(
                "[Smart-Filter] Query=%r -> %d tools "
                "(top_score=%s)",
                query_str,
                len(matched_names),
                response_data.get(
                    "top_score",
                    "N/A",
                ),
            )

            return json.dumps(
                {
                    "matched_tools": matched_names,
                    "count": len(matched_names),
                    "top_score": response_data.get(
                        "top_score",
                        0.0,
                    ),
                }
            )

        except requests.exceptions.RequestException as exc:
            logger.error(
                "[Smart-Filter] Service error: %s",
                exc,
            )

            return json.dumps(
                {
                    "error": (
                        f"Semantic search unavailable: {exc}"
                    ),
                    "matched_tools": [],
                    "count": 0,
                }
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            logger.error(
                "[Smart-Filter] Invalid response "
                "from semantic service: %s",
                exc,
            )

            return json.dumps(
                {
                    "error": (
                        "Invalid semantic search response: "
                        f"{exc}"
                    ),
                    "matched_tools": [],
                    "count": 0,
                }
            )

    register_tool = getattr(
        ctx,
        "register_tool",
        None,
    )

    if not callable(register_tool):
        raise RuntimeError(
            "Hermes PluginContext does not provide "
            "register_tool()."
        )

    register_tool(
        name="tool_search",
        toolset="tools",
        schema=TOOL_SEARCH_SCHEMA,
        handler=handle_semantic_tool_search,
        description=(
            "Semantically search available tools "
            "using the Smart Filter service."
        ),
        override=True,
    )

    logger.info(
        "[Smart-Filter] tool_search override registered."
    )
