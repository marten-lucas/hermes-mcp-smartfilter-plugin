# Hermes MCP Smart Filter Plugin

An official plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that dynamically reduces large MCP (Model Context Protocol) tool surfaces to the most relevant candidates before each LLM request using semantic vector search.

When running multiple MCP servers (e.g., via Coolify or Agent Gateway), exposing hundreds or thousands of tool schemas simultaneously inflates prompt context, increases API costs, and degrades model tool-calling accuracy. This plugin intercepts outgoing LLM requests via Hermes' native `llm_request` middleware and filters the active tool list using the external [hermes-mcp-smartfilter-service](https://github.com/marten-lucas/hermes-mcp-smartfilter-service).

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │      Hermes Agent      │
                          │                        │
                          │   Discovers ~1000 tools│
                          │   via MCP Gateway      │
                          └───────────┬────────────┘
                                      │
                                User Prompt
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │ llm_request Middleware   │
                        │ (mcp-smart-filter)       │
                        └─────────────┬────────────┘
                                      │
                         Query + Slim Candidates (Names/Descs)
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │  Smart Filter Service    │
                        │  (FastEmbed + NumPy)     │
                        └─────────────┬────────────┘
                                      │
                           Selected Candidate Names
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │ llm_request Middleware   │
                        │ Filters full JSON schemas│
                        └─────────────┬────────────┘
                                      │
                              Reduced Tools (5-15)
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │    LLM Inference Call  │
                          └────────────────────────┘
```

---

## Features

- **LLM Middleware Interception:** Hides excess tools before prompt construction using Hermes' native `ctx.register_middleware("llm_request")`.
- **Payload Optimization:** Sends only lightweight candidate metadata (`name` and `description`) over HTTP to minimize overhead.
- **Full Schema Preservation:** Filters tools locally, preserving full JSON parameters, schemas, and provider formats for the LLM.
- **Fail-Open Safe:** Automatically falls back to the full tool list if the filter service times out or returns an error.
- **Cross-Format Support:** Handles both flat (Anthropic) and nested `function` (OpenAI/Ollama) tool schema definitions.

---

## Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed.
- A running instance of [hermes-mcp-smartfilter-service](https://github.com/marten-lucas/hermes-mcp-smartfilter-service).

---

## Installation

Install the plugin into your Hermes environment using the CLI:

```bash
hermes plugins install marten-lucas/hermes-mcp-smartfilter-plugin
```

After installation, activate the plugin:

```bash
hermes plugins enable mcp-smart-filter
```

---

## Configuration

The plugin uses the following options, configured via export or inside `~/.hermes/config.yaml`:

| Option | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `SMART_FILTER_SERVICE_URL` | **Yes** | — | Base URL of your `hermes-mcp-smartfilter-service` |
| `SMART_FILTER_API_KEY` | **Yes** | — | Secret API key for authenticating with the backend service |
| `SMART_FILTER_MAX_K` | No | `8` | Maximum number of top relevant tools to present to the LLM |
| `SMART_FILTER_MIN_K` | No | `1` | Minimum number of tools to guarantee when candidates match |
| `SMART_FILTER_MIN_SCORE` | No | `0.35` | Absolute minimum similarity score required for inclusion (0.0 to 1.0) |
| `SMART_FILTER_RELATIVE_THRESHOLD` | No | `0.75` | Relative threshold against top score (score $\ge 0.75 \times \text{top\_score}$) |
| `SMART_FILTER_TIMEOUT` | No | `2.5` | HTTP request timeout in seconds before failing open |
| `SMART_FILTER_DEBUG` | No | `false` | Enable verbose log output (`true`, `1`, `yes`) |

---

## Verification & Debugging

Verify plugin registration with the Hermes Plugin Doctor:

```bash
hermes plugins doctor ~/.hermes/plugins/mcp-smart-filter --ci
```

Check active plugins during an agent session:

```bash
/plugins
```

To tail filtering logs in real time:

```bash
hermes logs --level DEBUG | grep "[Smart-Filter]"
```

---

## Companion Repository

- **Service:** [hermes-mcp-smartfilter-service](https://github.com/marten-lucas/hermes-mcp-smartfilter-service) — FastEmbed + NumPy embedding backend service.

---

## License

[MIT](LICENSE)
