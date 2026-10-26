"""Ollama-backed semantic tool search for Hermes Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

LOG_FILE = os.path.expanduser("~/.hermes/mcp_smart_filter.log")


def _write_audit_log(message: str) -> None:
    """Write guaranteed audit log entry to ~/.hermes/mcp_smart_filter.log."""
    try:
        import datetime

        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except OSError:
        logger.debug("[Smart-Filter] Could not write audit log", exc_info=True)


def _sanitize_mcp_name_component(value: str) -> str:
    """Prefer Hermes Core's ``tools.mcp_tool_schema.sanitize_mcp_name_component``.

    Replaces every char outside ``[A-Za-z0-9_]`` (hyphens included) with ``_``,
    so that names we recommend to the LLM exactly match the names Hermes
    registers in its tool registry. Falls back to an identical local
    implementation when Hermes Core is unavailable (e.g. standalone tests).
    """
    try:
        from tools.mcp_tool_schema import sanitize_mcp_name_component as _core_sanitize

        return _core_sanitize(value)
    except ImportError:
        return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _make_mcp_name(server: str, tool: str) -> str:
    """Build the registry/wire name ``mcp__<sanitizedServer>__<sanitizedTool>``.

    Prefers Hermes Core's ``mcp_prefixed_tool_name`` when available.
    """
    try:
        from tools.mcp_tool_schema import mcp_prefixed_tool_name as _core_prefix

        return _core_prefix(server, tool)
    except ImportError:
        return f"mcp__{_sanitize_mcp_name_component(server)}__{_sanitize_mcp_name_component(tool)}"


def _extract_tool_info(tool: Any) -> dict[str, Any]:
    """
    Extract searchable name, description, and parameter information
    from diverse Hermes tool representations (dicts, functions, objects).
    """
    name = ""
    description = ""
    param_names: list[str] = []
    required_params: list[str] = []

    if isinstance(tool, dict):
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool

        name = str(source.get("name", "")).strip()
        description = str(source.get("description", "")).strip()

        params = (
            source.get("parameters")
            or source.get("input_schema")
            or source.get("schema")
        )
        if isinstance(params, dict):
            properties = params.get("properties")
            if isinstance(properties, dict):
                param_names = list(properties.keys())
            required = params.get("required")
            if isinstance(required, list):
                required_params = [str(name) for name in required if isinstance(name, str)]
    else:
        # Support tool objects with attributes
        name = str(getattr(tool, "name", "")).strip()
        description = str(getattr(tool, "description", "")).strip()
        schema = getattr(tool, "schema", None) or getattr(tool, "parameters", None)
        if isinstance(schema, dict):
            props = schema.get("properties") or {}
            if isinstance(props, dict):
                param_names = list(props.keys())
            required = schema.get("required")
            if isinstance(required, list):
                required_params = [str(name) for name in required if isinstance(name, str)]

    # Build rich searchable representation
    search_parts = [name]
    if description:
        search_parts.append(description)
    if param_names:
        search_parts.append(f"parameters: {', '.join(param_names)}")

    search_text = " - ".join(search_parts)

    return {
        "name": name,
        "description": description,
        "parameters": param_names,
        "required": required_params,
        "search_text": search_text,
    }


def _get_available_tools(ctx: Any, kwargs: dict[str, Any]) -> list[Any]:
    """
    Defensively obtain the current Hermes tool catalogue from arguments,
    context manager, PluginContext, or the local MCP schema cache.
    """
    if kwargs.get("_scope_tool_defs"):
        tools = kwargs.get("tools")
        if isinstance(tools, dict):
            return list(tools.values())
        return list(tools or [])

    tools = kwargs.get("tools") or kwargs.get("available_tools")
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
                if result:
                    return list(result)
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
            if result:
                return list(result)
        except Exception:
            logger.exception("[Smart-Filter] Failed to read tools from PluginContext")

    # Fallback to local MCP schema cache and ToolRegistry
    cached_tools: list[dict[str, Any]] = []
    seen: set[str] = set()

    try:
        from tools.registry import registry

        for name, entry in getattr(registry, "_tools", {}).items():
            if name not in seen:
                seen.add(name)
                cached_tools.append(
                    {
                        "name": name,
                        "description": getattr(entry, "description", "") or "",
                        "parameters": [],
                    }
                )
    except ImportError as exc:
        logger.debug(
            "[Smart-Filter] Hermes ToolRegistry unavailable: %s", type(exc).__name__
        )

    cache_path = os.path.expanduser("~/.hermes/cache/mcp_schema_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            for server_name, server_data in cache.items():
                for t in server_data.get("tools", []):
                    tool_name = _make_mcp_name(server_name, t["name"])
                    if tool_name not in seen:
                        seen.add(tool_name)
                        cached_tools.append(
                            {
                                "name": tool_name,
                                "description": t.get("description") or "",
                                "parameters": list(
                                    t.get("inputSchema", {})
                                    .get("properties", {})
                                    .keys()
                                ),
                            }
                        )
        except (OSError, ValueError, AttributeError, TypeError, KeyError) as exc:
            logger.debug(
                "[Smart-Filter] MCP schema cache unavailable: %s", type(exc).__name__
            )

    return cached_tools


_LIVE_TOOLS_CACHE: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
_LIVE_CACHE_LOCK = threading.Lock()


def _configured_gateway_servers(
    mcp_servers: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Select AgentGateway MCP entries, while leaving unrelated MCP servers alone."""
    selected = [
        (name, config)
        for name, config in mcp_servers.items()
        if name == "agentgateway" or name.startswith("agentgateway_")
    ]
    if selected:
        return selected
    return [("agentgateway", {"url": "https://mcp.cloud.kiga-gramschatz.de/mcp"})]


def _parse_mcp_tools(response: Any) -> list[dict[str, Any]]:
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            return data.get("result", {}).get("tools", [])
    try:
        return response.json().get("result", {}).get("tools", [])
    except (AttributeError, ValueError):
        return []


def _fetch_live_mcp_tools(
    groups: str = "", timeout: float = 12.0, ttl: float = 60.0
) -> list[dict[str, Any]]:
    """Fetch the authorized live catalog from configured AgentGateway routes.

    The legacy ``agentgateway`` entry remains supported. Split gateways use
    ``agentgateway_<route>`` names; direct MCP servers (for example Coolify)
    are deliberately not included. Cache entries are scoped to routes and groups.
    """
    now = time.time()
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        mcp_servers = cfg.get("mcp_servers") or {}
    except Exception as exc:  # noqa: BLE001 - Hermes config is optional to standalone plugin use
        logger.debug(
            "[Smart-Filter] Hermes MCP config unavailable: %s", type(exc).__name__
        )
        mcp_servers = {}

    servers = _configured_gateway_servers(mcp_servers)
    server_key = "|".join(f"{name}={entry.get('url', '')}" for name, entry in servers)
    cache_key = (server_key, groups)
    with _LIVE_CACHE_LOCK:
        cached = _LIVE_TOOLS_CACHE.get(cache_key)
        if cached and now - cached[1] < ttl:
            return cached[0]

    token = os.environ.get("AGENTGATEWAY_BEARER_TOKEN", "")
    if not token:
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("AGENTGATEWAY_BEARER_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except OSError:
                pass
    if not token:
        logger.warning(
            "[Smart-Filter] Cannot perform live MCP discovery: missing AgentGateway token."
        )
        return []

    try:
        import httpx
    except ImportError as exc:
        logger.error(
            "[Smart-Filter] HTTP client unavailable for live MCP discovery: %s", exc
        )
        return []

    tools: list[dict[str, Any]] = []
    successful_routes: list[str] = []
    for server_name, server_config in servers:
        endpoint = server_config.get("url") or ""
        if not endpoint:
            logger.warning(
                "[Smart-Filter] AgentGateway entry %s has no URL.", server_name
            )
            continue
        configured_headers = server_config.get("headers") or {}
        auth_header = next(
            (
                value
                for key, value in configured_headers.items()
                if key.lower() == "authorization"
            ),
            "",
        )
        if not auth_header or "${" in auth_header:
            auth_header = f"Bearer {token}"
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if groups:
            headers["X-User-Groups"] = groups

        try:
            with httpx.Client(headers=headers, timeout=timeout) as client:
                r_init = client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "hermes-smart-filter",
                                "version": "2.6",
                            },
                        },
                    },
                )
                r_init.raise_for_status()
                session_id = r_init.headers.get("mcp-session-id")
                if not session_id:
                    logger.warning(
                        "[Smart-Filter] %s initialize returned no MCP session ID.",
                        server_name,
                    )
                    continue
                client.headers["mcp-session-id"] = session_id
                client.post(
                    endpoint,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
                r_list = client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    },
                )
                raw_tools = _parse_mcp_tools(r_list)

            for raw_tool in raw_tools:
                name = str(raw_tool.get("name", ""))
                if name.startswith("mcp__"):
                    parts = name.split("__", 2)
                    tool_name = (
                        _make_mcp_name(parts[1], parts[2]) if len(parts) == 3 else name
                    )
                else:
                    tool_name = _make_mcp_name(server_name, name)
                desc = raw_tool.get("description") or ""
                params = list(
                    (raw_tool.get("inputSchema") or {}).get("properties", {}).keys()
                )
                tools.append(
                    {
                        "name": tool_name,
                        "description": desc,
                        "parameters": params,
                        "search_text": f"{tool_name} - {desc}"
                        + (f" - parameters: {', '.join(params)}" if params else ""),
                    }
                )
            successful_routes.append(server_name)
        except Exception as exc:  # noqa: BLE001 - one failed gateway route must not hide the others
            logger.error(
                "[Smart-Filter] Failed to query AgentGateway route %s (%s): %s",
                server_name,
                endpoint,
                exc,
            )

    with _LIVE_CACHE_LOCK:
        _LIVE_TOOLS_CACHE[cache_key] = (tools, now)

    logger.info(
        "[Smart-Filter] Live MCP discovery: retrieved %d tools from routes=%s for groups=%r (TTL=%.0fs)",
        len(tools),
        ",".join(successful_routes),
        groups,
        ttl,
    )
    _write_audit_log(
        f"[LIVE MCP DISCOVERY] groups={groups!r} -> fetched {len(tools)} authorized tools from routes={','.join(successful_routes)}"
    )
    return tools


def _normalize_ollama_root(base_url: str) -> str:
    """Convert an Ollama/OpenAI base URL to the native Ollama API root."""
    value = str(base_url or "").strip()
    if not value:
        return ""
    if value.startswith(":"):
        value = "http://127.0.0.1" + value
    elif "://" not in value:
        value = "http://" + value

    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    for suffix in ("/api/tags", "/v1/models", "/api", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _secret_from_hermes(key_env: str) -> str:
    if not key_env:
        return ""
    try:
        from agent.secret_scope import get_secret_str

        value = get_secret_str(key_env, "")
        if value:
            return str(value).strip()
    except Exception as exc:  # noqa: BLE001 - fall back to the process environment if secret resolution fails
        logger.debug(
            "[Smart-Filter] Hermes secret resolution failed: %s", type(exc).__name__
        )
    return os.environ.get(key_env, "").strip()


def _resolve_ollama_connection() -> tuple[str, dict[str, str]]:
    """Reuse either Hermes' legacy custom provider or its providers.ollama entry."""
    explicit_url = os.environ.get("SMART_FILTER_OLLAMA_BASE_URL", "").strip()
    explicit_key = os.environ.get("SMART_FILTER_OLLAMA_API_KEY", "").strip()
    config: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception as exc:  # noqa: BLE001 - standalone mode may not have Hermes config available
        logger.debug(
            "[Smart-Filter] Hermes model config unavailable: %s", type(exc).__name__
        )

    ollama_entry: dict[str, Any] = {}
    providers = config.get("providers") or {}
    if isinstance(providers, dict) and isinstance(providers.get("ollama"), dict):
        ollama_entry = providers["ollama"]
    if not ollama_entry:
        custom_providers = config.get("custom_providers") or []
        if isinstance(custom_providers, list):
            ollama_entry = next(
                (
                    entry
                    for entry in custom_providers
                    if isinstance(entry, dict)
                    and str(entry.get("name", "")).strip().lower() == "ollama"
                ),
                {},
            )

    model_config = config.get("model") or {}
    if not isinstance(model_config, dict):
        model_config = {}
    base_url = (
        explicit_url
        or str(
            ollama_entry.get("base_url")
            or ollama_entry.get("api")
            or ollama_entry.get("url")
            or ""
        ).strip()
        or str(model_config.get("base_url") or "").strip()
    )

    raw_headers = ollama_entry.get("extra_headers") or ollama_entry.get("headers") or {}
    headers = (
        {str(name): str(value) for name, value in raw_headers.items()}
        if isinstance(raw_headers, dict)
        else {}
    )

    api_key = (
        explicit_key
        or str(ollama_entry.get("api_key") or model_config.get("api_key") or "").strip()
    )
    key_env = str(
        ollama_entry.get("key_env") or ollama_entry.get("api_key_env") or ""
    ).strip()
    if not explicit_key and key_env:
        api_key = _secret_from_hermes(key_env) or api_key
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key = _secret_from_hermes(api_key[2:-1])

    if api_key:
        headers = {
            name: value
            for name, value in headers.items()
            if name.lower() != "authorization"
        }
        headers["Authorization"] = f"Bearer {api_key}"
    elif not any(name.lower() == "authorization" for name in headers):
        api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    if not base_url:
        try:
            from hermes_cli.models_local import _get_ollama_base_url

            base_url = _get_ollama_base_url()
        except Exception as exc:  # noqa: BLE001 - continue to environment-based endpoint fallbacks
            logger.debug(
                "[Smart-Filter] Hermes Ollama endpoint unavailable: %s",
                type(exc).__name__,
            )
    if not base_url:
        base_url = (
            os.environ.get("OLLAMA_BASE_URL", "").strip()
            or os.environ.get("OLLAMA_HOST", "").strip()
        )

    base_url = _normalize_ollama_root(base_url)
    if not base_url:
        raise RuntimeError(
            "No Ollama endpoint configured; set SMART_FILTER_OLLAMA_BASE_URL "
            "or configure an Ollama provider in Hermes."
        )
    if not urlsplit(base_url).hostname:
        raise RuntimeError("The configured Ollama endpoint is not a valid HTTP URL.")

    return f"{base_url}/api/embed", headers


class OllamaSearchEngine:
    """Semantic search using the configured Ollama embedding model and an in-memory cache."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        connection_resolver: Callable[[], tuple[str, dict[str, str]]] | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:

        self.model_name = (
            model_name
            or os.environ.get("SMART_FILTER_OLLAMA_MODEL")
            or "qwen3-embedding:latest"
        )
        self.batch_size = self._positive_int("SMART_FILTER_OLLAMA_BATCH_SIZE", 64)
        self.threads = self._positive_int("SMART_FILTER_THREADS", 1)
        self.gpu_layers = self._gpu_layer_count("SMART_FILTER_NUM_GPU", -1)
        self.timeout = self._positive_float("SMART_FILTER_OLLAMA_TIMEOUT", 30.0)
        self.keep_alive = os.environ.get("SMART_FILTER_OLLAMA_KEEP_ALIVE", "5m").strip()
        self._connection_resolver = connection_resolver or _resolve_ollama_connection
        self._client_factory = client_factory
        self._connection: tuple[str, dict[str, str]] | None = None
        self._retry_after = 0.0
        self._cache_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._refreshing_fingerprints: set[str] = set()
        self._cache: (
            tuple[str, list[dict[str, Any]], list[tuple[float, ...]]] | None
        ) = None

        if os.environ.get("SMART_FILTER_MODEL") and not os.environ.get(
            "SMART_FILTER_OLLAMA_MODEL"
        ):
            logger.info(
                "[Smart-Filter] Ignoring legacy SMART_FILTER_MODEL for Ollama; "
                "use SMART_FILTER_OLLAMA_MODEL to select the embedding model."
            )

    @staticmethod
    def _positive_int(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _positive_float(name: str, default: float) -> float:
        try:
            return max(0.1, float(os.environ.get(name, str(default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _gpu_layer_count(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
            return value if value >= -1 else default
        except (TypeError, ValueError):
            return default

    def _init_model(self) -> bool:
        """Resolve the configured Ollama provider without exposing its credentials."""

        if time.monotonic() < self._retry_after:
            return False
        if self._connection is not None:
            return True
        try:
            self._connection = self._connection_resolver()
            logger.info(
                "[Smart-Filter] Ollama embedding provider ready (model=%s, host=%s).",
                self.model_name,
                urlsplit(self._connection[0]).hostname or "unknown",
            )
            return True
        except Exception as exc:  # noqa: BLE001 - provider failures must activate keyword fallback
            self._record_failure(exc)
            return False

    @staticmethod
    def _normalize_vector(values: Any) -> tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Ollama returned an empty or non-finite embedding vector")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return tuple(value / norm for value in vector)

    def _client(self):
        factory = self._client_factory
        if factory is None:
            import httpx

            factory = httpx.Client
        return factory(timeout=self.timeout)

    def _embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        if not self._init_model() or self._connection is None:
            raise RuntimeError("Ollama embedding provider is unavailable")

        endpoint, headers = self._connection
        payload_options = {
            "num_gpu": self.gpu_layers,
            "num_thread": self.threads,
        }
        vectors: list[tuple[float, ...]] = []
        with self._client() as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload: dict[str, Any] = {
                    "model": self.model_name,
                    "input": batch,
                    "options": payload_options,
                }
                if self.keep_alive:
                    payload["keep_alive"] = self.keep_alive
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                embeddings = (
                    result.get("embeddings") if isinstance(result, dict) else None
                )
                if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                    raise ValueError("Ollama returned an unexpected embedding response")
                vectors.extend(self._normalize_vector(vector) for vector in embeddings)
        return vectors

    def _record_failure(self, exc: Exception) -> None:
        self._connection = None
        self._retry_after = time.monotonic() + 15.0
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", HTTP {status}" if status is not None else ""
        logger.warning(
            "[Smart-Filter] Ollama embedding request failed (%s%s); "
            "using keyword fallback.",
            type(exc).__name__,
            suffix,
        )

    @staticmethod
    def _compute_fingerprint(tools: list[dict[str, Any]]) -> str:
        """Compute a stable fingerprint of tool names and searchable descriptions."""
        hasher = hashlib.md5()
        for tool in sorted(tools, key=lambda item: item["name"]):
            hasher.update(tool["name"].encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
            hasher.update(tool["search_text"].encode("utf-8", errors="ignore"))
            hasher.update(b"\x01")
        return hasher.hexdigest()

    def _sync_tool_catalog(
        self, tools: list[dict[str, Any]], *, build_if_missing: bool = True
    ) -> bool:
        """Embed and cache a tool catalog once per model and catalog fingerprint.

        ``build_if_missing=False`` is the latency-sensitive search path: it only
        checks the cache and never waits for a background pre-warm to finish.
        """
        if not tools:
            with self._cache_lock:
                self._cache = None
            return False
        if not self._init_model():
            return False

        fingerprint = f"{self.model_name}:{self._compute_fingerprint(tools)}"
        with self._cache_lock:
            if self._cache is not None and self._cache[0] == fingerprint:
                return True

        if not build_if_missing:
            return False

        with self._build_lock:
            with self._cache_lock:
                if self._cache is not None and self._cache[0] == fingerprint:
                    return True
            try:
                vectors = self._embed_texts([tool["search_text"] for tool in tools])
                if len(vectors) != len(tools):
                    raise ValueError(
                        "Ollama returned an incomplete tool catalog embedding"
                    )
                if len({len(vector) for vector in vectors}) != 1:
                    raise ValueError(
                        "Ollama returned inconsistent embedding dimensions"
                    )
                with self._cache_lock:
                    self._cache = (fingerprint, tools, vectors)
                logger.info(
                    "[Smart-Filter] Cached Ollama embeddings for %d tools "
                    "(dim=%d, fingerprint=%s).",
                    len(tools),
                    len(vectors[0]),
                    fingerprint[-8:],
                )
                return True
            except Exception as exc:  # noqa: BLE001 - embedding failures must activate keyword fallback
                self._record_failure(exc)
                return False

    def _schedule_catalog_refresh(self, tools: list[dict[str, Any]]) -> None:
        """Refresh a changed/missing catalog off the tool-call path."""
        if not tools:
            return
        fingerprint = f"{self.model_name}:{self._compute_fingerprint(tools)}"
        with self._cache_lock:
            if self._cache is not None and self._cache[0] == fingerprint:
                return
        with self._refresh_lock:
            if fingerprint in self._refreshing_fingerprints:
                return
            self._refreshing_fingerprints.add(fingerprint)

        def refresh() -> None:
            try:
                self._sync_tool_catalog(tools)
            finally:
                with self._refresh_lock:
                    self._refreshing_fingerprints.discard(fingerprint)

        try:
            threading.Thread(
                target=refresh, daemon=True, name="smart-filter-catalog-refresh"
            ).start()
        except RuntimeError:
            with self._refresh_lock:
                self._refreshing_fingerprints.discard(fingerprint)
            logger.warning("[Smart-Filter] Could not schedule catalog refresh.")

    def search(
        self,
        query: str,
        tools: list[dict[str, Any]],
        limit: int = 8,
        min_score: float = 0.25,
    ) -> list[tuple[dict[str, Any], float, str]]:
        query_text = query.strip()
        if not query_text:
            return []
        if not self._sync_tool_catalog(tools, build_if_missing=False):
            self._schedule_catalog_refresh(tools)
            return self._keyword_search(query_text, tools, limit=limit)

        try:
            query_vector = self._embed_texts([query_text])[0]
            with self._cache_lock:
                snapshot = self._cache
            if snapshot is None:
                return self._keyword_search(query_text, tools, limit=limit)
            _, cached_tools, cached_vectors = snapshot
            if len(query_vector) != len(cached_vectors[0]):
                raise ValueError(
                    "Ollama query and catalog embedding dimensions do not match"
                )

            ranked = sorted(
                (
                    (sum(a * b for a, b in zip(vector, query_vector)), index)
                    for index, vector in enumerate(cached_vectors)
                ),
                reverse=True,
            )
            matches: list[tuple[dict[str, Any], float, str]] = []
            for score, index in ranked:
                if score < min_score:
                    break
                matches.append((cached_tools[index], float(score), "ollama"))
                if len(matches) >= limit:
                    break
            if matches:
                return matches
        except Exception as exc:  # noqa: BLE001 - query embedding failures use keyword fallback
            self._record_failure(exc)

        return self._keyword_search(query_text, tools, limit=limit)

    @staticmethod
    def _keyword_search(
        query: str,
        tools: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[tuple[dict[str, Any], float, str]]:
        """Keyword / token overlap search fallback."""
        tokens = [t.lower() for t in re.split(r"[\s_\-.:/]+", query) if len(t) > 1]
        if not tokens:
            return []

        scored_tools: list[tuple[dict[str, Any], float, str]] = []
        for tool in tools:
            name_lower = tool["name"].lower()
            desc_lower = tool["description"].lower()
            text_lower = tool["search_text"].lower()

            score = 0.0
            matched_tokens = 0

            for tok in tokens:
                if tok in name_lower:
                    score += 0.4
                    matched_tokens += 1
                elif tok in desc_lower:
                    score += 0.2
                    matched_tokens += 1
                elif tok in text_lower:
                    score += 0.1
                    matched_tokens += 1

            if matched_tokens > 0:
                normalized_score = min(0.95, score / max(1, len(tokens)))
                scored_tools.append((tool, normalized_score, "keyword_fallback"))

        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:limit]


# Global engine instance for the process
_ENGINE = OllamaSearchEngine()


def _load_config() -> tuple[int, int, float]:
    """Load configuration parameters with safe fallbacks."""
    try:
        max_k = int(os.environ.get("SMART_FILTER_MAX_K", "8"))
        min_k = int(os.environ.get("SMART_FILTER_MIN_K", "1"))
        min_score = float(os.environ.get("SMART_FILTER_MIN_SCORE", "0.25"))
        return max_k, min_k, min_score
    except ValueError as exc:
        logger.warning(
            "[Smart-Filter] Invalid environment config: %s. Using defaults.", exc
        )
        return 8, 1, 0.25


def create_handler(ctx: Any):
    """Create the search handler bound to the Hermes plugin context."""
    default_max_k, default_min_k, default_min_score = _load_config()

    debug_mode = os.environ.get("SMART_FILTER_DEBUG", "false").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    if debug_mode:
        logger.setLevel(logging.DEBUG)

    def handle_tool_search(
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        args = args or {}

        # 1. Parse and sanitize queries
        raw_queries = args.get("queries")
        if isinstance(raw_queries, str):
            queries = [raw_queries.strip()]
        elif isinstance(raw_queries, list):
            queries = [str(q).strip() for q in raw_queries if str(q).strip()]
        else:
            # Backward compatibility with single "query"
            raw_query = args.get("query")
            if isinstance(raw_query, str) and raw_query.strip():
                queries = [raw_query.strip()]
            else:
                queries = []

        if not queries:
            return json.dumps(
                {
                    "queries": [],
                    "total_available": 0,
                    "results": [],
                    "tools": {},
                    "error": "queries is required and must contain at least one non-empty query",
                },
                ensure_ascii=False,
            )

        # Cap queries at 10 to avoid unnecessary overhead
        if len(queries) > 10:
            queries = queries[:10]

        # 2. Honor the native bridge's validated limits when it supplied them.
        scoped_call = bool(kwargs.get("_scope_tool_defs"))
        core_config = kwargs.get("config")
        if scoped_call and core_config is None:
            try:
                from tools.tool_search import load_config as load_core_config

                core_config = load_core_config()
            except (ImportError, AttributeError):
                core_config = None

        def core_setting(name: str, fallback: int) -> int:
            value = (
                core_config.get(name, fallback)
                if isinstance(core_config, dict)
                else getattr(core_config, name, fallback)
            )
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        maximum = min(50, default_max_k, core_setting("max_search_limit", 50))
        default_limit = min(
            maximum,
            core_setting("search_default_limit", default_max_k),
        )
        minimum = min(default_min_k, maximum)
        requested_limit = args.get("limit")
        if isinstance(requested_limit, int) and requested_limit > 0:
            limit = min(max(minimum, requested_limit), maximum)
        else:
            limit = max(minimum, default_limit)

        # 3. Read the current session's actual tool definitions when called by
        # Hermes' native deferred-tool bridge. Never widen that scope from a
        # process-wide registry or stale schema cache.
        session_tool_defs = kwargs.get("tools") or []
        if isinstance(session_tool_defs, dict):
            session_tool_defs = list(session_tool_defs.values())
        raw_tools = _get_available_tools(ctx, kwargs)
        if scoped_call:
            try:
                from tools.tool_search import _deferrable_in

                raw_tools = _deferrable_in(raw_tools)
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "[Smart-Filter] Core deferred-tool scope unavailable; refusing to search unscoped tools (%s).",
                    type(exc).__name__,
                )
                raw_tools = []

        connector_entries: list[list[Any]] = [[] for _ in queries]
        connector_failure = None
        connectors_unavailable = None
        if scoped_call:
            try:
                from tools.connectors.search import (
                    connections_in_scope,
                    connector_entries_by_group,
                )
                from tools.connectors.search import (
                    connectors_unavailable as format_connectors_unavailable,
                )

                if connections_in_scope(session_tool_defs):
                    connector_search = kwargs.get("connector_search")
                    call = kwargs.get("call")
                    if connector_search is None:
                        from tools.connectors.gateway.bridge import (
                            connector_search_hits,
                        )

                        connector_search = partial(connector_search_hits, call=call)
                    connector_entries, connector_failure = connector_entries_by_group(
                        queries,
                        connector_search=connector_search,
                        call=call,
                    )
                    connectors_unavailable = format_connectors_unavailable
            except (ImportError, AttributeError) as exc:
                logger.debug(
                    "[Smart-Filter] Native connector search unavailable (%s).",
                    type(exc).__name__,
                )

        extracted_tools: list[dict[str, Any]] = []

        for item in raw_tools:
            info = _extract_tool_info(item)
            if info["name"]:
                extracted_tools.append(info)

        logger.info(
            "[Smart-Filter] Executing tool_search for %d queries across %d available tools (limit=%d)",
            len(queries),
            len(extracted_tools),
            limit,
        )

        results: list[dict[str, Any]] = []
        tools_map: dict[str, dict[str, Any]] = {}

        # 4. Search each query
        connector_names: set[str] = set()
        for query_index, q in enumerate(queries):
            matched_tuples = _ENGINE.search(
                query=q,
                tools=extracted_tools,
                limit=limit,
                min_score=default_min_score,
            )

            matched_names: list[str] = []
            used_method = "ollama"
            for tool_info, score, method in matched_tuples:
                name = tool_info["name"]
                matched_names.append(name)
                used_method = method

                if name not in tools_map:
                    tools_map[name] = {
                        "source": "mcp" if name.startswith("mcp__") else "plugin",
                        "source_name": name.split("__")[1] if "__" in name else "",
                        "description": tool_info["description"][:400],
                        "required": [
                            p
                            for p in tool_info.get("required", [])
                            if isinstance(p, str)
                        ][:32],
                        "score": round(score, 3),
                        "search_method": method,
                    }

            for entry in connector_entries[query_index]:
                name = str(getattr(entry, "name", "") or "")
                if not name:
                    continue
                if name not in matched_names:
                    matched_names.append(name)
                connector_names.add(name)
                if name not in tools_map:
                    try:
                        from tools.tool_search import _shared_tool_record

                        tools_map[name] = _shared_tool_record(entry)
                    except (ImportError, AttributeError):
                        tools_map[name] = {
                            "source": "connectors",
                            "source_name": str(getattr(entry, "source_name", "") or ""),
                            "description": str(getattr(entry, "description", "") or "")[:500],
                            "required": [],
                        }
                used_method = f"{used_method}+connectors"

            if debug_mode:
                import sys

                print(
                    f"\n[Smart-Filter] Query: {q!r} -> {len(matched_names)} matches (engine={used_method})",
                    file=sys.stderr,
                )

            results.append(
                {
                    "query": q,
                    "matches": matched_names,
                    "engine": used_method,
                }
            )

        engines = sorted({result["engine"] for result in results})
        _write_audit_log(
            f"[TOOL_SEARCH OVERRIDE EXECUTED] queries={queries!r} (limit={limit}, available={len(extracted_tools)}) -> engines={engines}, matches={list(tools_map.keys())}"
        )
        response = {
            "queries": queries,
            "total_available": len(extracted_tools) + len(connector_names),
            "search_engine": f"ollama ({_ENGINE.model_name})",
            "results": results,
            "tools": tools_map,
        }
        if connector_failure and connectors_unavailable is not None:
            response["connectors"] = connectors_unavailable(
                connector_failure,
                verb="searched",
            )
        if scoped_call:
            try:
                from tools.tool_search import _record_search_metric

                outcome = (
                    "connector_unavailable"
                    if connector_failure
                    else "hits"
                    if tools_map
                    else "no_hits"
                )
                _record_search_metric(
                    session_tool_defs,
                    hits=len(tools_map),
                    outcome=outcome,
                )
            except (ImportError, AttributeError):
                pass
        return json.dumps(response, ensure_ascii=False)

    return handle_tool_search


def create_pre_llm_hook(ctx: Any):
    """
    Create a pre_llm_call hook callback that semantically routes relevant MCP tools
    directly into the user message context, avoiding blind guessing or bias towards
    arbitrary visible tools.
    """
    min_score = float(os.environ.get("SMART_FILTER_PRE_LLM_MIN_SCORE", "0.55"))
    max_tools = int(os.environ.get("SMART_FILTER_PRE_LLM_MAX_TOOLS", "4"))
    enabled = os.environ.get("SMART_FILTER_PRE_LLM_ENABLED", "true").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    live_rbac_discovery = os.environ.get(
        "SMART_FILTER_LIVE_RBAC_DISCOVERY", "false"
    ).lower() in {"true", "1", "yes", "on"}

    def on_pre_llm_call(
        user_message: Any = None, **kwargs: Any
    ) -> dict[str, str] | None:
        if not enabled:
            return None

        # Extract text from user_message (can be str or dict with content/text)
        text = ""
        if isinstance(user_message, str):
            text = user_message.strip()
        elif isinstance(user_message, dict):
            text = str(
                user_message.get("content") or user_message.get("text") or ""
            ).strip()

        # Clean off prepended [System note: ...] headers (e.g. session-reset notices from Hermes/Talk)
        text = re.sub(
            r"^\[System note:[^\]]+\]\s*", "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        if not text or len(text) < 5:
            return None

        # Ignore slash commands or isolated system signals
        if text.startswith(("/", '"/', "[System note:")):
            return None

        # Resolve current user groups via hermes-x-on-behalf if available
        user_groups = ""
        try:
            import importlib
            import sys

            # Das Paket registriert sich als "hermes_x_on_behalf" (Underscores);
            # der Hermes-Plugin-Loader kann es zusätzlich unter "hermes_plugins.hermes_x_on_behalf"
            # importieren. get_principal() liegt im .context-Modul.
            get_p = None
            for module_name in (
                "hermes_x_on_behalf.context",
                "hermes_plugins.hermes_x_on_behalf.context",
            ):
                mod = sys.modules.get(module_name)
                if mod is None:
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        mod = None
                get_p = getattr(mod, "get_principal", None) if mod else None
                if callable(get_p):
                    break
            p = get_p() if callable(get_p) else None
            if p and getattr(p, "groups", None):
                user_groups = ",".join(sorted(p.groups))
        except Exception as exc:  # noqa: BLE001 - identity plugin failures use the documented fallback
            logger.debug(
                "[Smart-Filter] Could not resolve principal groups: %s",
                type(exc).__name__,
            )

        # Fallback for user identity from turn arguments or session
        if not user_groups:
            sender_id = str(kwargs.get("sender_id") or "").strip().lower()
            if not sender_id:
                meta = kwargs.get("session_metadata") or {}
                sender_id = str(meta.get("user_id") or "").strip().lower()

            # Known role mappings from x_on_behalf config
            if sender_id in ("vorstand", "admin"):
                user_groups = "it-admin,vorstand"
            elif sender_id in ("kiga-team", "elternbeirat"):
                user_groups = sender_id

        # Live RBAC Discovery or Local Schema Cache
        extracted: list[dict[str, Any]] = []
        if live_rbac_discovery:
            live_tools = _fetch_live_mcp_tools(groups=user_groups)
            if live_tools:
                extracted = live_tools

        if not extracted:
            raw_tools = _get_available_tools(ctx, kwargs)
            for item in raw_tools:
                info = _extract_tool_info(item)
                if info["name"]:
                    extracted.append(info)

        if not extracted:
            return None

        matches = _ENGINE.search(
            query=text, tools=extracted, limit=max_tools, min_score=min_score
        )
        if not matches:
            return None

        lines = [
            "### MANDATORY DIRECTIVE: Relevant Pre-Authorized MCP Tools for this request:",
            "The following tools directly serve the user's intent. You MUST use these tools via `tool_call` instead of running terminal/curl commands or using generic dashboard integrations (like Homarr):",
        ]
        matched_names = []
        for tool_info, score, method in matches:
            name = tool_info["name"]
            matched_names.append(name)
            desc = (tool_info.get("description") or "").split("\n")[0].strip()[:140]
            lines.append(
                f'- Tool: `{name}` — {desc} -> Call via: `tool_call(name="{name}", arguments={{...}})`'
            )

        lines.append(
            "\nDo NOT attempt curl or manual credentials prompts when one of the above specialized tools is listed."
        )

        _write_audit_log(
            f"[PRE_LLM_CALL ROUTING] query={text!r} -> matched {len(matched_names)} tools (engine={matches[0][2]}): {matched_names}"
        )
        return {"context": "\n".join(lines)}

    return on_pre_llm_call
