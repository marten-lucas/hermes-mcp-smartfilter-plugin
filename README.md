# Hermes FastEmbed Smart Search Plugin

A high-performance semantic search plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that overrides the built-in BM25 `tool_search` engine with **in-process FastEmbed dense vector retrieval**.

Designed specifically for large tool surfaces (**1,000+ tools across multiple MCP servers and plugins**), this plugin enables natural-language semantic discovery, sub-2ms query latency via in-memory vector caching, and robust keyword fallback.

---

## Why Override `tool_search`?

When Hermes Agent connects to numerous MCP servers or plugin suites, exposing hundreds or thousands of tool schemas simultaneously causes **context window bloat, inflated token costs, and model distraction**.

Hermes solves this via **Progressive Tool Disclosure** using three bridge tools:
1. `tool_search(queries, limit)` — discovers relevant tools from the deferred catalog.
2. `tool_describe(names)` — loads full JSON schemas for chosen tools on demand.
3. `tool_call(name, arguments)` — executes the tool.

### BM25 vs. FastEmbed Semantic Search

| Feature | Built-in BM25 | FastEmbed Plugin (`mcp-smart-filter`) |
| :--- | :--- | :--- |
| **Search Mechanism** | Keyword frequency & stemming | Dense semantic embeddings (`BAAI/bge-small-en-v1.5`) |
| **Semantic Matching** | Fails on synonyms (e.g. "find tickets" vs `jira_search_issues`) | Understands intent, synonyms, and natural language descriptions |
| **Search Latency (1000+ tools)** | ~5-15 ms (text tokenization) | **< 2 ms** (vectorized NumPy matrix multiplication) |
| **Catalog Caching** | Rebuilds index per turn | **In-memory embedding cache** (invalidated only on catalog hash change) |
| **Fallback Guarantee** | Substring match on zero hits | Automatic keyword/token overlap fallback |

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │                 Hermes Agent                 │
                      │         (1,000+ deferred MCP tools)          │
                      └──────────────────────┬───────────────────────┘
                                             │
                               User asks for action
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │              tool_search(...)                │
                      │      (Overridden by mcp-smart-filter)        │
                      └──────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼ (Primary)                                 ▼ (Fallback)
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │   FastEmbed Vector Search │               │  Keyword / Token Overlap  │
         │   (In-Memory Cached ONNX) │               │   (Substrings & Tokens)   │
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Top-K Matched Tools & Scores                 │
                      │ {"results": [...], "tools": {...}}           │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Hermes calls tool_describe & tool_call       │
                      └──────────────────────────────────────────────┘
```

---

## Installation

### 1. Install via Hermes CLI

```bash
hermes plugins install marten-lucas/hermes-mcp-smartfilter-plugin --no-enable
```

### 2. Enable with Tool Override Consent

Hermes Agent requires explicit consent when a plugin replaces a core tool (`tool_search`). Grant the `tools.override` capability using the `--allow-tool-override` flag:

```bash
hermes plugins enable mcp-smart-filter --allow-tool-override
```

> [!TIP]
> Start a new Hermes session after enabling the plugin so that the tool registry updates cleanly.

---

## Configuration

All configuration is optional and can be set in your `.env` or process environment:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SMART_FILTER_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed ONNX embedding model name. |
| `SMART_FILTER_MIN_SCORE` | `0.25` | Minimum cosine similarity threshold for vector matches. |
| `SMART_FILTER_MAX_K` | `8` | Default maximum tool candidates returned per query. |
| `SMART_FILTER_MIN_K` | `1` | Lower bound for requested search candidates. |
| `SMART_FILTER_DEBUG` | `false` | Enable verbose debug logging (`true`/`false`). |
| `SMART_FILTER_THREADS` | `1` | ONNX intra-op threads for the FastEmbed model. |

---

## Running Tests

The test suite validates schema compatibility, tool extraction, vector caching, limit clamping, and 1,000-tool catalog scaling:

```bash
python3 -m unittest discover -s tests -v
```

## Diagnostics

The plugin writes an audit log to `~/.hermes/mcp_smart_filter.log`:

- `[PLUGIN LOADED]` — registration outcome (native override vs. fallback)
- `[TOOL_SEARCH OVERRIDE EXECUTED]` — every search with queries, limit, catalog size, and matched tool names

## Troubleshooting

**Plugin registers as `semantic_tool_search` instead of overriding `tool_search`**

The `tools.override` capability was not granted. Re-enable with consent:

```bash
hermes plugins enable mcp-smart-filter --allow-tool-override
```

**Searches always use `keyword_fallback` engine**

The `fastembed` package is not installed in Hermes' Python environment:

```bash
pip install "fastembed>=0.3.0,<1"
```

**Plugin does not appear in `hermes plugins list`**

Run discovery diagnostics:

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

---

## License

MIT
