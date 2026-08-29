"""Tool schemas for the MCP Smart Filter plugin."""

SEMANTIC_TOOL_SEARCH = {
    "name": "semantic_tool_search",
    "description": (
        "Semantically search the available MCP and plugin tools for one or "
        "more capabilities. Use this when you need to discover a specialized "
        "tool without knowing its exact name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": (
                    "One or more natural-language descriptions of the "
                    "capabilities you need. Each query is searched independently."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of matching tools returned per query. "
                    "Defaults to the configured value."
                ),
            },
        },
        "required": ["queries"],
    },
}

