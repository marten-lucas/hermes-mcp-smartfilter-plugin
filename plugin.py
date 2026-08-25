import logging
import requests

logger = logging.getLogger("hermes.plugins.mcp_tool_reducer")


def register(ctx):
    """Eintrittspunkt für das Hermes Agent Plugin System."""
    # 1. Basis-Endpunkt, Auth & Debug-Modus
    SEARCH_SERVICE_URL = ctx.env.get("SMART_ROUTING_SERVICE_URL", "")
    API_KEY = ctx.env.get("SMART_ROUTING_API_KEY", "")

    # Debug Flag (true, 1, yes)
    DEBUG_MODE = (
        ctx.env.get("SMART_ROUTING_DEBUG", "false").lower() in ("true", "1", "yes")
    )

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("[Smart-Routing] Debug-Logging ist aktiviert.")

    # 2. Dynamische Schwellenwerte aus der Hermes-Umgebung laden
    try:
        MAX_K = int(ctx.env.get("SMART_ROUTING_MAX_K", "8"))
        MIN_SCORE = float(ctx.env.get("SMART_ROUTING_MIN_SCORE", "0.35"))
        RELATIVE_THRESHOLD = float(ctx.env.get("SMART_ROUTING_RELATIVE_THRESHOLD", "0.75"))
    except ValueError as err:
        logger.warning(
            f"[Smart-Routing] Ungültiger Konfigurationswert für Filter-Parameter: {err}. "
            "Verwende Fallback-Defaults (max_k=8, min_score=0.35, relative_threshold=0.75)."
        )
        MAX_K = 8
        MIN_SCORE = 0.35
        RELATIVE_THRESHOLD = 0.75

    @ctx.on("on_tools_load")
    def filter_mcp_tools(**event):
        """Lifecycle Interceptor Hook."""
        raw_tools = event.get("tools", [])
        user_prompt = event.get("user_prompt", "")

        user_id = ctx.session.get("user_id") or ctx.env.get(
            "X_ON_BEHALF_OF", "default_user"
        )

        if DEBUG_MODE:
            logger.debug(
                f"[Smart-Routing] Prompt empfangen für User '{user_id}': \"{user_prompt}\" "
                f"({len(raw_tools)} Tools geladen)."
            )

        if len(raw_tools) <= 5 or not user_prompt:
            if DEBUG_MODE:
                logger.debug(
                    "[Smart-Routing] Filter übersprungen (Tool-Anzahl <= 5 oder Prompt leer)."
                )
            return

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            "X-On-Behalf-Of": user_id,
        }

        payload = {
            "query": user_prompt,
            "max_k": MAX_K,
            "min_score": MIN_SCORE,
            "relative_threshold": RELATIVE_THRESHOLD,
            "tools": raw_tools,
        }

        try:
            response = requests.post(
                f"{SEARCH_SERVICE_URL}/filter",
                json=payload,
                headers=headers,
                timeout=2.5,
            )
            response.raise_for_status()

            filtered_result = response.json()
            reduced_tools = filtered_result.get("tools", [])

            if reduced_tools:
                top_score = filtered_result.get("top_score", "N/A")
                logger.info(
                    f"[Smart-Routing] Tools für User '{user_id}' erfolgreich von "
                    f"{len(raw_tools)} auf {len(reduced_tools)} reduziert "
                    f"(Top-Score: {top_score})."
                )

                if DEBUG_MODE:
                    tool_names = [t.get("name") for t in reduced_tools]
                    logger.debug(
                        f"[Smart-Routing] Ausgewählte Tools: {', '.join(tool_names)}"
                    )

                event["tools"] = reduced_tools

        except requests.exceptions.RequestException as err:
            logger.error(
                f"[Smart-Routing] Fehler beim Aufruf des Search-Services: {err}. "
                "Verwende vollständige Tool-Liste als Fallback."
            )