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

- Hermes Agent installed.
- A running instance of [hermes-mcp-smartfilter-service](https://github.com/marten-lucas/hermes-mcp-smartfilter-service).

---

## Installation

Install the plugin using the Hermes CLI:

```bash
hermes plugins install marten-lucas/hermes-mcp-smartfilter-plugin
```

Activate the plugin:

```bash
hermes plugins enable mcp-smart-filter
