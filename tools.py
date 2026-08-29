"""FastEmbed-backed semantic tool search."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests


logger = logging.getLogger("hermes.plugins.mcp_smart_filter")


def _extract_tool_info(tool: Any) -> dict[str, str]:
    """Extract the searchable name and description from a Hermes tool definition."""
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


def _get_available_tools(ctx: Any, kwargs: dict[str, Any]) -> list[Any]:
    """
    Obtain the current Hermes tool catalogue.

    Hermes versions expose the registry differently, so keep this
    deliberately defensive.
    """

    tools = kwargs.get("tools")

    if tools:
        return list(tools)

    tools = kwargs.get("available_tools")

    if tools:
        return list(tools)

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
                    "[Smart-Filter] Failed to read tools from ctx._manager"
                )

    get_tools = getattr(ctx, "get_tools", None)

    if callable(get_tools):
        try:
            result = get_tools()

            if isinstance(result, dict):
                return list(result.values())

            return list(result or [])

        except Exception:
            logger.exception(
                "[Smart-Filter] Failed to read tools from PluginContext"
            )

    return []


def _load_config() -> tuple[int, int, float, float, float]:
    """Load Smart Filter configuration from environment variables."""

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

        relative_threshold = float(
            os.environ.get(
                "SMART_FILTER_RELATIVE_THRESHOLD",
                "0.70",
            )
        )

        timeout = float(
            os.environ.get(
                "SMART_FILTER_TIMEOUT",
                "2.5",
            )
        )

        return (
            max_k,
            min_k,
            min_score,
            relative_threshold,
            timeout,
        )

    except ValueError as exc:
        logger.warning(
            "[Smart-Filter] Invalid configuration: %s. "
            "Using defaults.",
            exc,
        )

        return (
            8,
            1,
            0.25,
            0.70,
            2.5,
        )


def create_handler(ctx: Any):
    """
    Create the semantic search handler bound to the Hermes plugin context.
    """

    service_url = os.environ.get(
        "SMART_FILTER_SERVICE_URL",
        "",
    ).rstrip("/")

    api_key = os.environ.get(
        "SMART_FILTER_API_KEY",
        "",
    )

    debug_mode = (
        os.environ.get(
            "SMART_FILTER_DEBUG",
            "false",
        ).lower()
        in {"true", "1", "yes", "on"}
    )

    if debug_mode:
        logger.setLevel(logging.DEBUG)

    (
        max_k,
        min_k,
        min_score,
        relative_threshold,
        timeout,
    ) = _load_config()

    if not service_url:
        logger.warning(
            "[Smart-Filter] SMART_FILTER_SERVICE_URL is not configured."
        )

    def handle_semantic_tool_search(
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        args = args or {}

        raw_queries = args.get("queries")

        # Accept Hermes' current array format.
        if isinstance(raw_queries, str):
            queries = [raw_queries]

        elif isinstance(raw_queries, list):
            queries = [
                str(query).strip()
                for query in raw_queries
                if str(query).strip()
            ]

        else:
            # Backwards compatibility with the old plugin API.
            query = args.get("query")

            if isinstance(query, str) and query.strip():
                queries = [query.strip()]
            else:
                queries = []

        if not queries:
            return json.dumps(
                {
                    "queries": [],
                    "total_available": 0,
                    "results": [],
                    "tools": {},
                    "error": (
                        "queries is required and must contain "
                        "at least one non-empty query"
                    ),
                },
                ensure_ascii=False,
            )

        if len(queries) > 10:
            queries = queries[:10]

        available_tools = _get_available_tools(
            ctx,
            kwargs,
        )

        payload_tools: list[dict[str, str]] = []

        for item in available_tools:
            tool_info = _extract_tool_info(item)

            if tool_info["name"]:
                payload_tools.append(tool_info)

        logger.info(
            "[Smart-Filter] Tool catalogue contains %d candidates",
            len(payload_tools),
        )

        if not service_url:
            return json.dumps(
                {
                    "queries": queries,
                    "total_available": len(payload_tools),
                    "results": [
                        {
                            "query": query,
                            "matches": [],
                        }
                        for query in queries
                    ],
                    "tools": {},
                    "error": (
                        "SMART_FILTER_SERVICE_URL "
                        "is not configured"
                    ),
                },
                ensure_ascii=False,
            )

        headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            headers["X-API-Key"] = api_key

        results: list[dict[str, Any]] = []
        tools_map: dict[str, dict[str, Any]] = {}

        for query in queries:
            payload = {
                "query": query,
                "tools": payload_tools,
                "max_k": max_k,
                "min_k": min_k,
                "min_score": min_score,
                "relative_threshold": relative_threshold,
            }

            try:
                logger.info(
                    "[Smart-Filter] Searching query=%r "
                    "against %d tools",
                    query,
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

                matches: list[str] = []

                for item in matched_items:
                    if isinstance(item, dict):
                        name = item.get("name")

                        if name:
                            name = str(name)
                            matches.append(name)

                            tools_map[name] = {
                                "description": str(
                                    item.get(
                                        "description",
                                        "",
                                    )
                                )[:400],
                                "score": item.get(
                                    "_score",
                                ),
                            }

                    elif isinstance(item, str):
                        matches.append(item)

                results.append(
                    {
                        "query": query,
                        "matches": matches,
                    }
                )

                logger.info(
                    "[Smart-Filter] Query=%r -> %d matches "
                    "(top_score=%s)",
                    query,
                    len(matches),
                    response_data.get(
                        "top_score",
                        "N/A",
                    ),
                )

            except requests.exceptions.RequestException as exc:
                logger.error(
                    "[Smart-Filter] Service error for query %r: %s",
                    query,
                    exc,
                )

                results.append(
                    {
                        "query": query,
                        "matches": [],
                        "error": (
                            f"Semantic search unavailable: {exc}"
                        ),
                    }
                )

            except (
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                logger.error(
                    "[Smart-Filter] Invalid response for query %r: %s",
                    query,
                    exc,
                )

                results.append(
                    {
                        "query": query,
                        "matches": [],
                        "error": (
                            f"Invalid semantic search response: {exc}"
                        ),
                    }
                )

        return json.dumps(
            {
                "queries": queries,
                "total_available": len(payload_tools),
                "results": results,
                "tools": tools_map,
            },
            ensure_ascii=False,
        )

    return handle_semantic_tool_search
