# Hermes Ollama Smart Filter

Semantic MCP-tool discovery for Hermes Agent. The plugin overrides Hermes' native BM25 `tool_search`, embeds the current tool catalog and each query through the **Ollama provider already configured in Hermes**, and ranks the cached vectors locally. It keeps keyword matching as a fallback when Ollama cannot be reached.

The default embedding model is `qwen3-embedding:latest`; this is separate from Hermes' chat model. The plugin sends batches to Ollama's native `POST /api/embed` endpoint and defaults to Ollama's automatic GPU offload (`num_gpu: -1`), with one CPU thread available for preprocessing. Set `SMART_FILTER_NUM_GPU=0` to force CPU-only embeddings. It does not require FastEmbed, ONNX Runtime, NumPy, or a separate embedding service.

## Why override `tool_search`?

Hermes' built-in BM25 search is fast and dependency-free, but lexical matching can miss relevant tools when the user's wording differs from a tool's name or description. Semantic embeddings provide a better match for large deferred catalogs (1,000+ tools) without putting every schema in the prompt.

| Feature | Hermes BM25 | Ollama Smart Filter |
| --- | --- | --- |
| Matching | Lexical | Semantic vector similarity |
| Embedding runtime | None | Existing configured Ollama provider |
| Catalog processing | Per-search lexical index | Batched once per catalog fingerprint, cached in memory |
| Failure behavior | Native search | Keyword fallback in the plugin; dispatch exceptions fall back to Hermes BM25 |

## Installation and enablement

The plugin has **no Python package dependencies of its own**. It uses Hermes' existing `httpx` and Ollama configuration. Install it as usual, then explicitly consent to overriding the core tool:

```bash
hermes plugins install marten-lucas/hermes-mcp-smartfilter-plugin --no-enable
hermes plugins enable mcp-smart-filter --allow-tool-override
```

Start a new Hermes session or restart the gateway after changing the plugin or its settings.

## Ollama configuration

By default, the plugin reuses Hermes' configured Ollama provider URL and credentials through Hermes' Ollama configuration helpers. It removes an OpenAI-compatible `/v1` suffix and calls the native Ollama `/api/embed` endpoint, which supports per-request CPU options.

The default model is `qwen3-embedding:latest`. Override it in Hermes' `~/.hermes/.env` if desired:

```dotenv
SMART_FILTER_OLLAMA_MODEL=qwen3-embedding:latest
SMART_FILTER_THREADS=1
```

Optional connection overrides are available when the existing provider configuration cannot be reused:

```dotenv
SMART_FILTER_OLLAMA_BASE_URL=http://ollama-host:11434
SMART_FILTER_OLLAMA_API_KEY=...
```

Use the native Ollama API root for `SMART_FILTER_OLLAMA_BASE_URL` (not a URL ending in `/v1`). The API-key override is optional; by default the plugin reuses the credential resolved by Hermes. Never put a real key in the repository or logs.

The old FastEmbed variable `SMART_FILTER_MODEL` is not used by the Ollama backend; set `SMART_FILTER_OLLAMA_MODEL` instead. This prevents a former value such as `BAAI/bge-small-en-v1.5` from being sent to Ollama as though it were an Ollama model.

### CPU and batching controls

| Variable | Default | Description |
| --- | --- | --- |
| `SMART_FILTER_OLLAMA_MODEL` | `qwen3-embedding:latest` | Ollama embedding model; use an embedding model, not the chat model |
| `SMART_FILTER_THREADS` | `1` | CPU threads sent to Ollama with each embedding request |
| `SMART_FILTER_NUM_GPU` | `-1` | Ollama GPU layers (`-1` = automatic offload, `0` = CPU-only) |
| `SMART_FILTER_OLLAMA_BATCH_SIZE` | `64` | Number of tool descriptions embedded per request |
| `SMART_FILTER_OLLAMA_TIMEOUT` | `30` | HTTP timeout in seconds |
| `SMART_FILTER_OLLAMA_KEEP_ALIVE` | `5m` | Ollama model keep-alive after embedding |
| `SMART_FILTER_MAX_K` | `8` | Default maximum candidate tools per query |
| `SMART_FILTER_MIN_K` | `1` | Lower bound for requested candidates |
| `SMART_FILTER_MIN_SCORE` | `0.25` | Minimum cosine similarity for a result |
| `SMART_FILTER_PRE_LLM_ENABLED` | `true` | Inject relevant tools before the LLM call |
| `SMART_FILTER_PRE_LLM_MIN_SCORE` | `0.55` | Minimum score for pre-LLM injection |
| `SMART_FILTER_PRE_LLM_MAX_TOOLS` | `4` | Maximum tools injected by the pre-LLM hook |
| `SMART_FILTER_LIVE_RBAC_DISCOVERY` | `false` | Query Agentgateway live with the caller's identity headers |

Ollama's native `/api/embed` request accepts `options.num_gpu` and `options.num_thread`; the plugin uses `SMART_FILTER_NUM_GPU` and `SMART_FILTER_THREADS` per embedding request. This affects only embedding inference, not Hermes chat inference. The first catalog indexing can take longer; while it runs, searches use the keyword fallback rather than waiting synchronously. Subsequent searches reuse the in-memory vectors until the model or tool catalog changes.

The embedding request sends tool names/descriptions and search queries to the configured Ollama endpoint. Whether that traffic stays on the local network depends on the URL and reverse-proxy routing configured for that provider.

## Search and fallback behavior

1. The plugin obtains the current tool catalog from Hermes, the PluginContext, or the MCP schema cache.
2. Tool descriptions are embedded in batches and cached by model plus catalog fingerprint.
3. Each search query is embedded with the same model; normalized dot products rank the candidates.
4. If configuration, connectivity, model lookup, or embedding fails, or a catalog fingerprint is still being indexed in the background, the plugin immediately uses its keyword fallback instead of blocking the tool-search call. It logs failure type/HTTP status without logging credentials.

The `pre_llm_call` hook uses the same engine to surface a few likely MCP tools before generation. Live RBAC discovery remains optional and continues to use `X-User-Groups` when enabled.

When AgentGateway is split into multiple MCP routes, name the Hermes server entries `agentgateway_<route>` (for example `agentgateway_infra`, `agentgateway_kita`, and `agentgateway_home`). The plugin queries each matching entry independently, prefixes tools with the corresponding Hermes server name, and keeps results from healthy routes if another route is unavailable. Direct MCP servers such as `coolify` are not included. The legacy single `agentgateway` entry remains supported.

## Tests

Run from this directory:

```bash
python3 -m unittest discover -s tests -v
```

Tests use a mocked Ollama HTTP client; they do not require a running model server or download model weights.

## Diagnostics

The plugin writes an audit log to `~/.hermes/mcp_smart_filter.log`:

- `[PLUGIN LOADED]` — registration outcome (native override vs. fallback)
- `[TOOL_SEARCH OVERRIDE EXECUTED]` — query count, candidate count, engines used, and matches
- `[PRE-WARM]` — background catalog warm-up status

If every search reports `keyword_fallback`, check that Hermes can resolve the Ollama provider, that its endpoint exposes native `/api/embed`, and that the configured embedding model is installed and available to that endpoint.
