import logging
import os
import requests

logger = logging.getLogger("hermes.plugins.mcp_tool_selector")


def _extract_tool_info(tool: dict) -> dict:
    """Extrahiert name und description sowohl aus Anthropic- als auch OpenAI-Schemas."""
    if not isinstance(tool, dict):
        return {"name": "", "description": ""}

    fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
    src = fn or tool
    return {
        "name": str(src.get("name", "")),
        "description": str(src.get("description", "")),
    }


def register(ctx):
    """Offizieller Registrierungs-Einstiegspunkt laut Hermes Doku."""
    SERVICE_URL = os.environ.get("SMART_ROUTING_SERVICE_URL", "").rstrip("/")
    API_KEY = os.environ.get("SMART_ROUTING_API_KEY", "")
    DEBUG_MODE = os.environ.get("SMART_ROUTING_DEBUG", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)

    try:
        MAX_K = int(os.environ.get("SMART_ROUTING_MAX_K", "8"))
        MIN_K = int(os.environ.get("SMART_ROUTING_MIN_K", "1"))
        MIN_SCORE = float(os.environ.get("SMART_ROUTING_MIN_SCORE", "0.35"))
        RELATIVE_THRESHOLD = float(
            os.environ.get("SMART_ROUTING_RELATIVE_THRESHOLD", "0.75")
        )
        TIMEOUT = float(os.environ.get("SMART_ROUTING_TIMEOUT", "2.5"))
    except ValueError as err:
        logger.warning(
            f"[Tool-Selector] Ungültige Konfiguration: {err}. Nutze Defaults."
        )
        MAX_K, MIN_K, MIN_SCORE, RELATIVE_THRESHOLD, TIMEOUT = 8, 1, 0.35, 0.75, 2.5

    def filter_llm_request_middleware(request, session_id="", **kwargs):
        """Middleware vom Kind 'llm_request' zum Abfangen und Ersetzen der Provider-Payload."""
        raw_tools = request.get("tools") or []

        # Bei 5 oder weniger Tools ist keine Reduzierung notwendig
        if len(raw_tools) <= 5:
            return None

        # Letzte User-Nachricht als Prompt ermitteln
        user_prompt = ""
        for msg in reversed(request.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_prompt = content
                elif isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    user_prompt = " ".join(text_parts)
                break

        if not user_prompt:
            return None

        # Reduzierte Nutzlast für die Serveranfrage (nur name & description)
        payload_tools = [_extract_tool_info(t) for t in raw_tools]

        payload = {
            "query": user_prompt,
            "tools": payload_tools,
            "max_k": MAX_K,
            "min_k": MIN_K,
            "min_score": MIN_SCORE,
            "relative_threshold": RELATIVE_THRESHOLD,
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            "X-Session-Id": str(session_id),
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
            selected_candidates = res_data.get("tools", [])
            selected_names = {t["name"] for t in selected_candidates if "name" in t}

            if not selected_names:
                return None

            # Ursprüngliche Tool-Objekte (inkl. JSON-Schemas) anhand der Namen filtern
            reduced_tools = [
                t for t in raw_tools if _extract_tool_info(t)["name"] in selected_names
            ]

            logger.info(
                f"[Tool-Selector] Tools erfolgreich reduziert: {len(raw_tools)} -> {len(reduced_tools)} "
                f"(Top-Score: {res_data.get('top_score', 'N/A')})"
            )

            # Korrekter Rückgabe-Contract für Hermes llm_request Middleware
            updated_request = dict(request)
            updated_request["tools"] = reduced_tools
            return {"request": updated_request}

        except requests.exceptions.RequestException as err:
            logger.error(
                f"[Tool-Selector] Service-Fehler ({err}). Fallback: Alle Tools freigegeben."
            )
            return None

    # Offizielle Hermes-Registrierung für Middleware
    ctx.register_middleware("llm_request", filter_llm_request_middleware)
    logger.info("[Tool-Selector] Middleware 'llm_request' erfolgreich registriert.")
