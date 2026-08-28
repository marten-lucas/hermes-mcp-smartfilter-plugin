import json
import logging
import os
import requests

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

# Schema für das überschriebene tool_search
TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": (
        "Search for available tools by semantic query or keywords. "
        "Use this when you need a tool for a specific task but its full schema is not yet visible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query or keywords describing what action you want to perform.",
            },
        },
        "required": ["query"],
    },
}


def _extract_tool_info(tool: dict) -> dict:
    """Extrahiert name und description unabhängig vom Schema."""
    if not isinstance(tool, dict):
        return {"name": "", "description": ""}

    fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
    src = fn or tool
    return {
        "name": str(src.get("name", "")),
        "description": str(src.get("description", "")),
    }


def register(ctx):
    """Offizielle Registrierung des überschriebenen tool_search Handlers."""
    SERVICE_URL = os.environ.get("SMART_FILTER_SERVICE_URL", "").rstrip("/")
    API_KEY = os.environ.get("SMART_FILTER_API_KEY", "")
    DEBUG_MODE = os.environ.get("SMART_FILTER_DEBUG", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)

    try:
        MAX_K = int(os.environ.get("SMART_FILTER_MAX_K", "8"))
        MIN_K = int(os.environ.get("SMART_FILTER_MIN_K", "1"))
        MIN_SCORE = float(os.environ.get("SMART_FILTER_MIN_SCORE", "0.25"))
        TIMEOUT = float(os.environ.get("SMART_FILTER_TIMEOUT", "2.5"))
    except ValueError as err:
        logger.warning(
            f"[Smart-Filter] Ungültige Konfiguration: {err}. Nutze Defaults."
        )
        MAX_K, MIN_K, MIN_SCORE, TIMEOUT = 8, 1, 0.25, 2.5

    def handle_semantic_tool_search(args: dict, **kwargs) -> str:
        """Handler, der das interne BM25-Matching durch FastEmbed ersetzt."""
        query = args.get("query") or args.get("queries") or ""
        if isinstance(query, list):
            query = " ".join(query)

        query_str = str(query).strip()
        if not query_str:
            return json.dumps({"tools": [], "count": 0, "error": "Empty query"})

        # Hole den vollständigen Werkzeugkatalog aus den übergebenen Kontexten
        # (Hermes übergibt registrierte Tools in kwargs oder via Registry)
        available_tools = kwargs.get("tools") or kwargs.get("available_tools") or []

        # Falls kwargs keine Liste liefert, hole alle registrierten MCP-Tools
        if not available_tools and hasattr(ctx, "_manager"):
            try:
                available_tools = list(ctx._manager.get_tools().values())
            except Exception:
                available_tools = []

        payload_tools = [_extract_tool_info(t) for t in available_tools]

        payload = {
            "query": query_str,
            "tools": payload_tools,
            "max_k": MAX_K,
            "min_k": MIN_K,
            "min_score": MIN_SCORE,
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        }

        try:
            response = requests.post(
                f"{SERVICE_URL}/filter",
                json=payload,
                headers=headers,
                timeout=TIMEOUT,
            )
            response.raise_for_status()

            res_data = response.json()
            matched_items = res_data.get("tools", [])

            logger.info(
                f"[Smart-Filter Reranker] Query: '{query_str}' -> {len(matched_items)} Tools gefunden "
                f"(Top-Score: {res_data.get('top_score', 'N/A')})"
            )

            # Rückgabe im vom Hermes tool_search erwarteten Format
            result = {
                "matched_tools": [t["name"] for t in matched_items if "name" in t],
                "count": len(matched_items),
                "top_score": res_data.get("top_score", 0.0),
            }
            return json.dumps(result)

        except requests.exceptions.RequestException as err:
            logger.error(
                f"[Smart-Filter Reranker] Service-Fehler: {err}. Nutze Fallback."
            )
            return json.dumps(
                {
                    "error": f"Semantic search unavailable: {err}",
                    "matched_tools": [],
                }
            )

    # Überschreibe das eingebaute tool_search von Hermes
    try:
        ctx.register_tool(
            name="tool_search",
            toolset="mcp_smart_filter",
            schema=TOOL_SEARCH_SCHEMA,
            handler=handle_semantic_tool_search,
            override=True,  # Überschreibt die Hermes-Core-Funktion
        )
        logger.info(
            "[Smart-Filter] Eingebautes 'tool_search' erfolgreich mit Vektor-Reranker überschrieben."
        )
    except Exception as err:
        logger.error(
            f"[Smart-Filter] Fehler beim Überschreiben von tool_search: {err}"
        )
