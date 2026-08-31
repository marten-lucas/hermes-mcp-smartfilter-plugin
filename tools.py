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
    context manager, or plugin context.

    Hinweis: ctx._manager ist private API und kann sich ohne
    Deprecation-Window ändern — der Pfad ist best effort; der öffentliche
    ctx.get_tools()-Pfad ist die stabile Quelle.
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
                return list(result or [])
            except Exception:
                logger.exception("[Smart-Filter] Failed to read tools from ctx._manager")

    get_tools = getattr(ctx, "get_tools", None)
    if callable(get_tools):
        try:
            result = get_tools()
            if isinstance(result, dict):
                return list(result.values())
            return list(result or [])
        except Exception:
            logger.exception("[Smart-Filter] Failed to read tools from PluginContext")

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

            raw_embeddings = list(self._model.embed(texts))
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
                        "description": tool_info["description"][:400],
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
