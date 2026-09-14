"""FastEmbed-backed semantic tool search for Hermes Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any

logger = logging.getLogger("hermes.plugins.mcp_smart_filter")

LOG_FILE = os.path.expanduser("~/.hermes/mcp_smart_filter.log")


def _write_audit_log(message: str) -> None:
    """Write guaranteed audit log entry to ~/.hermes/mcp_smart_filter.log."""
    try:
        import datetime
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception:
        pass


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
    except Exception:
        return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _make_mcp_name(server: str, tool: str) -> str:
    """Build the registry/wire name ``mcp__<sanitizedServer>__<sanitizedTool>``.

    Prefers Hermes Core's ``mcp_prefixed_tool_name`` when available.
    """
    try:
        from tools.mcp_tool_schema import mcp_prefixed_tool_name as _core_prefix
        return _core_prefix(server, tool)
    except Exception:
        return f"mcp__{_sanitize_mcp_name_component(server)}__{_sanitize_mcp_name_component(tool)}"


def _extract_tool_info(tool: Any) -> dict[str, Any]:
    """
    Extract searchable name, description, and parameter information
    from diverse Hermes tool representations (dicts, functions, objects).
    """
    name = ""
    description = ""
    param_names: list[str] = []

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
    else:
        # Support tool objects with attributes
        name = str(getattr(tool, "name", "")).strip()
        description = str(getattr(tool, "description", "")).strip()
        schema = getattr(tool, "schema", None) or getattr(tool, "parameters", None)
        if isinstance(schema, dict):
            props = schema.get("properties") or {}
            if isinstance(props, dict):
                param_names = list(props.keys())

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
        "search_text": search_text,
    }


def _get_available_tools(ctx: Any, kwargs: dict[str, Any]) -> list[Any]:
    """
    Defensively obtain the current Hermes tool catalogue from arguments,
    context manager, PluginContext, or the local MCP schema cache.
    """
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
                logger.exception("[Smart-Filter] Failed to read tools from ctx._manager")

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
                cached_tools.append({
                    "name": name,
                    "description": getattr(entry, "description", "") or "",
                    "parameters": [],
                })
    except Exception:
        pass

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
                        cached_tools.append({
                            "name": tool_name,
                            "description": t.get("description") or "",
                            "parameters": list(t.get("inputSchema", {}).get("properties", {}).keys()),
                        })
        except Exception:
            pass

    return cached_tools


_LIVE_TOOLS_CACHE: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
_LIVE_CACHE_LOCK = threading.Lock()


def _fetch_live_mcp_tools(groups: str = "", timeout: float = 12.0, ttl: float = 60.0) -> list[dict[str, Any]]:
    """
    Query configured MCP servers (specifically agentgateway) with the caller's
    X-User-Groups / X-On-Behalf-Of headers to obtain the exact list of tools
    authorized by the gateway at runtime. Results are cached in-memory per group
    string with a configurable TTL.
    """
    import time
    now = time.time()
    cache_key = ("agentgateway", groups)

    with _LIVE_CACHE_LOCK:
        if cache_key in _LIVE_TOOLS_CACHE:
            cached_tools, timestamp = _LIVE_TOOLS_CACHE[cache_key]
            if now - timestamp < ttl:
                return cached_tools

    # Find agentgateway config
    endpoint = ""
    auth_header = ""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        mcp_servers = cfg.get("mcp_servers") or {}
        ag_cfg = mcp_servers.get("agentgateway") or {}
        endpoint = ag_cfg.get("url") or ""
        headers_cfg = ag_cfg.get("headers") or {}
        auth_header = headers_cfg.get("Authorization") or ""
    except Exception:
        pass

    # Fallback to env file if auth token has template string
    if not auth_header or "${" in auth_header:
        token = os.environ.get("AGENTGATEWAY_BEARER_TOKEN", "")
        if not token:
            env_path = os.path.expanduser("~/.hermes/.env")
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if "AGENTGATEWAY_BEARER_TOKEN" in line:
                                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break
                except Exception:
                    pass
        if token:
            auth_header = f"Bearer {token}"

    if not endpoint:
        endpoint = "https://mcp.cloud.kiga-gramschatz.de/mcp"

    if not auth_header:
        logger.warning("[Smart-Filter] Cannot perform live MCP discovery: missing Authorization token.")
        return []

    try:
        import httpx

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if groups:
            headers["X-User-Groups"] = groups

        with httpx.Client(headers=headers, timeout=timeout) as client:
            # 1. Initialize session
            r_init = client.post(
                endpoint,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "hermes-smart-filter", "version": "2.3"},
                    },
                },
            )
            session_id = r_init.headers.get("mcp-session-id")
            if not session_id:
                logger.warning("[Smart-Filter] Live MCP discovery initialize returned no session ID.")
                return []

            client.headers["mcp-session-id"] = session_id

            # 2. tools/list
            r_list = client.post(
                endpoint,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )

        tools: list[dict[str, Any]] = []
        for line in r_list.text.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                raw_tools = data.get("result", {}).get("tools", [])
                for t in raw_tools:
                    name = t.get("name", "")
                    if name.startswith("mcp__"):
                        # Already registry-form: normalize every component anyway
                        parts = name.split("__", 2)
                        if len(parts) == 3:
                            tool_name = _make_mcp_name(parts[1], parts[2])
                        else:
                            tool_name = name
                    else:
                        tool_name = _make_mcp_name("agentgateway", name)
                    desc = t.get("description") or ""
                    params = list(t.get("inputSchema", {}).get("properties", {}).keys())
                    tools.append({
                        "name": tool_name,
                        "description": desc,
                        "parameters": params,
                        "search_text": f"{tool_name} - {desc}" + (f" - parameters: {', '.join(params)}" if params else ""),
                    })
                break

        with _LIVE_CACHE_LOCK:
            _LIVE_TOOLS_CACHE[cache_key] = (tools, now)

        logger.info(
            "[Smart-Filter] Live MCP discovery: retrieved %d tools for groups=%r (TTL=%.0fs)",
            len(tools),
            groups,
            ttl,
        )
        _write_audit_log(
            f"[LIVE MCP DISCOVERY] groups={groups!r} -> fetched {len(tools)} authorized tools from {endpoint}"
        )
        return tools
    except Exception as exc:
        logger.error("[Smart-Filter] Failed to query live MCP tools from %s: %s", endpoint, exc)
        return []


class FastEmbedSearchEngine:
    """
    In-process FastEmbed semantic search engine with in-memory vector caching.
    Computes embeddings once per toolset fingerprint, enabling sub-2ms searches
    across 1000+ tools.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (
            model_name
            or os.environ.get("SMART_FILTER_MODEL")
            or os.environ.get("SMART_ROUTING_MODEL")
            or "BAAI/bge-small-en-v1.5"
        )
        self._model: Any = None
        self._model_failed: bool = False
        # Thread-sicherer Cache: unveränderlicher Snapshot (fingerprint, tools,
        # embeddings), atomar getauscht. Hermes läuft mehrthreadig (Delegations,
        # Background-Worker) — ein geteilter Lock verhindert, dass ein Thread
        # Embeddings mit einer veralteten Tool-Liste indexiert.
        self._cache_lock = threading.Lock()
        self._cache: tuple[str, list[dict[str, Any]], Any] | None = None

    def _init_model(self) -> bool:
        """Lazy load FastEmbed TextEmbedding model."""
        if self._model is not None:
            return True
        if self._model_failed:
            return False

        try:
            from fastembed import TextEmbedding  # type: ignore

            logger.info(
                "[Smart-Filter] Initializing FastEmbed model %r...", self.model_name
            )
            threads = int(os.environ.get("SMART_FILTER_THREADS", "1"))
            self._model = TextEmbedding(model_name=self.model_name, threads=threads)
            logger.info(
                "[Smart-Filter] FastEmbed model %r loaded successfully (threads=%d).",
                self.model_name,
                threads,
            )
            return True
        except ImportError:
            logger.warning(
                "[Smart-Filter] fastembed package is not installed. "
                "Semantic vector search will fallback to keyword search."
            )
            self._model_failed = True
            return False
        except Exception as exc:
            logger.error(
                "[Smart-Filter] Failed to initialize FastEmbed model %r: %s",
                self.model_name,
                exc,
            )
            self._model_failed = True
            return False

    @staticmethod
    def _compute_fingerprint(tools: list[dict[str, Any]]) -> str:
        """Compute MD5 fingerprint of tool names and descriptions."""
        hasher = hashlib.md5()
        for t in sorted(tools, key=lambda x: x["name"]):
            hasher.update(t["name"].encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
            hasher.update(t["search_text"].encode("utf-8", errors="ignore"))
            hasher.update(b"\x01")
        return hasher.hexdigest()

    def _sync_tool_catalog(self, tools: list[dict[str, Any]]) -> bool:
        """
        Check if tool catalog has changed. If changed, recompute and cache embeddings.
        Returns True if embeddings are ready.

        Thread-safety: der Lock schützt Modell-Init und Cache-Tausch. Der
        teure Embedding-Recompute läuft außerhalb des Locks (kein Blockieren
        paralleler Suchen), der Snapshot-Tausch selbst ist atomar.
        """
        if not tools:
            with self._cache_lock:
                self._cache = None
            return False

        current_fp = self._compute_fingerprint(tools)

        with self._cache_lock:
            if self._cache is not None and self._cache[0] == current_fp:
                return True

        if not self._init_model():
            return False

        try:
            import numpy as np  # type: ignore

            texts = [t["search_text"] for t in tools]
            logger.debug(
                "[Smart-Filter] Generating embeddings for %d tools...", len(texts)
            )

            # Generate embeddings in smaller chunks (batch_size=16) to conserve memory
            # and avoid OOM inside constrained cgroups
            raw_embeddings = []
            for i in range(0, len(texts), 32):
                chunk = texts[i : i + 32]
                raw_embeddings.extend(list(self._model.embed(chunk, batch_size=16)))

            embeddings = np.array(raw_embeddings, dtype=np.float32)

            # Normalize embeddings to unit length for fast cosine similarity via dot product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_embeddings = embeddings / norms

            # Atomarer Snapshot-Tausch: Leser sehen immer konsistente
            # (fingerprint, tools, embeddings)-Tripel.
            with self._cache_lock:
                self._cache = (current_fp, tools, normalized_embeddings)

            logger.info(
                "[Smart-Filter] Cached embeddings for %d tools (dim=%d, fingerprint=%s)",
                len(tools),
                normalized_embeddings.shape[1],
                current_fp[:8],
            )
            return True
        except Exception as exc:
            logger.error(
                "[Smart-Filter] Failed to embed tool catalog: %s",
                exc,
            )
            return False

    def search(
        self,
        query: str,
        tools: list[dict[str, Any]],
        limit: int = 8,
        min_score: float = 0.25,
    ) -> list[tuple[dict[str, Any], float, str]]:
        """
        Perform fast semantic search for query against tools catalog.
        Returns list of (tool_dict, score, method).
        """
        query_str = query.strip()
        if not query_str:
            return []

        # Try semantic search with FastEmbed
        if self._sync_tool_catalog(tools):
            try:
                import numpy as np  # type: ignore

                # Konsistenten Snapshot lesen (atomar unter Lock)
                with self._cache_lock:
                    snapshot = self._cache
                if snapshot is None:
                    return self._keyword_search(query_str, tools, limit=limit)
                _, cached_tools, cached_embeddings = snapshot

                raw_q_emb = list(self._model.embed([query_str]))[0]
                q_emb = np.array(raw_q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    q_emb = q_emb / q_norm

                # Compute cosine similarities via matrix multiplication
                scores = np.dot(cached_embeddings, q_emb)

                # Rank by descending score
                ranked_indices = np.argsort(-scores)

                matches: list[tuple[dict[str, Any], float, str]] = []
                for idx in ranked_indices:
                    score = float(scores[idx])
                    if score < min_score:
                        break
                    matches.append((cached_tools[idx], score, "fastembed"))
                    if len(matches) >= limit:
                        break

                if matches:
                    return matches

                logger.debug(
                    "[Smart-Filter] No semantic matches >= %.2f for query %r; falling back to keyword search",
                    min_score,
                    query_str,
                )
            except Exception as exc:
                logger.error(
                    "[Smart-Filter] Error during FastEmbed query execution: %s",
                    exc,
                )

        # Fallback to token / keyword match
        return self._keyword_search(query_str, tools, limit=limit)

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
_ENGINE = FastEmbedSearchEngine()


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
            queries = [
                str(q).strip() for q in raw_queries if str(q).strip()
            ]
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

        # 2. Parse limit parameter
        requested_limit = args.get("limit")
        if isinstance(requested_limit, int) and requested_limit > 0:
            limit = min(max(default_min_k, requested_limit), 50)
        else:
            limit = default_max_k

        # 3. Retrieve available tool catalogue
        raw_tools = _get_available_tools(ctx, kwargs)
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
        for q in queries:
            matched_tuples = _ENGINE.search(
                query=q,
                tools=extracted_tools,
                limit=limit,
                min_score=default_min_score,
            )

            matched_names: list[str] = []
            used_method = "fastembed"
            for tool_info, score, method in matched_tuples:
                name = tool_info["name"]
                matched_names.append(name)
                used_method = method

                if name not in tools_map:
                    tools_map[name] = {
                        "source": "mcp" if name.startswith("mcp__") else "plugin",
                        "source_name": name.split("__")[1] if "__" in name else "",
                        "description": tool_info["description"][:400],
                        "required": [p for p in tool_info.get("parameters", []) if isinstance(p, str)][:32],
                        "score": round(score, 3),
                        "search_method": method,
                    }

            if debug_mode:
                import sys
                print(
                    f"\n[FastEmbed-Override] Query: {q!r} -> {len(matched_names)} matches (engine={used_method})",
                    file=sys.stderr,
                )

            results.append(
                {
                    "query": q,
                    "matches": matched_names,
                    "engine": used_method,
                }
            )

        _write_audit_log(
            f"[TOOL_SEARCH OVERRIDE EXECUTED] queries={queries!r} (limit={limit}, available={len(extracted_tools)}) -> found {len(tools_map)} matches via FastEmbed: {list(tools_map.keys())}"
        )

        return json.dumps(
            {
                "queries": queries,
                "total_available": len(extracted_tools),
                "search_engine": f"fastembed ({_ENGINE.model_name})",
                "results": results,
                "tools": tools_map,
            },
            ensure_ascii=False,
        )

    return handle_tool_search


def create_pre_llm_hook(ctx: Any):
    """
    Create a pre_llm_call hook callback that semantically routes relevant MCP tools
    directly into the user message context, avoiding blind guessing or bias towards
    arbitrary visible tools.
    """
    min_score = float(os.environ.get("SMART_FILTER_PRE_LLM_MIN_SCORE", "0.55"))
    max_tools = int(os.environ.get("SMART_FILTER_PRE_LLM_MAX_TOOLS", "4"))
    enabled = os.environ.get("SMART_FILTER_PRE_LLM_ENABLED", "true").lower() in {"true", "1", "yes", "on"}
    live_rbac_discovery = os.environ.get("SMART_FILTER_LIVE_RBAC_DISCOVERY", "false").lower() in {"true", "1", "yes", "on"}

    def on_pre_llm_call(user_message: Any = None, **kwargs: Any) -> dict[str, str] | None:
        if not enabled:
            return None

        # Extract text from user_message (can be str or dict with content/text)
        text = ""
        if isinstance(user_message, str):
            text = user_message.strip()
        elif isinstance(user_message, dict):
            text = str(user_message.get("content") or user_message.get("text") or "").strip()

        # Clean off prepended [System note: ...] headers (e.g. session-reset notices from Hermes/Talk)
        text = re.sub(r"^\[System note:[^\]]+\]\s*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        if not text or len(text) < 5:
            return None

        # Ignore slash commands or isolated system signals
        if text.startswith("/") or text.startswith('"/') or text.startswith("[System note:"):
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
                    except Exception:
                        mod = None
                get_p = getattr(mod, "get_principal", None) if mod else None
                if callable(get_p):
                    break
            p = get_p() if callable(get_p) else None
            if p and getattr(p, "groups", None):
                user_groups = ",".join(sorted(p.groups))
        except Exception:
            pass

        # Fallback for user identity from turn arguments or session
        if not user_groups:
            sender_id = str(kwargs.get("sender_id") or "").strip().lower()
            if not sender_id:
                try:
                    meta = kwargs.get("session_metadata") or {}
                    sender_id = str(meta.get("user_id") or "").strip().lower()
                except Exception:
                    pass

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

        matches = _ENGINE.search(query=text, tools=extracted, limit=max_tools, min_score=min_score)
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
            lines.append(f"- Tool: `{name}` — {desc} -> Call via: `tool_call(name=\"{name}\", arguments={{...}})`")

        lines.append("\nDo NOT attempt curl or manual credentials prompts when one of the above specialized tools is listed.")

        _write_audit_log(
            f"[PRE_LLM_CALL ROUTING] query={text!r} -> matched {len(matched_names)} tools (engine={matches[0][2]}): {matched_names}"
        )
        return {"context": "\n".join(lines)}

    return on_pre_llm_call
