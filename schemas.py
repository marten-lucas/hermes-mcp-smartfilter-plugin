"""Tool schemas for the FastEmbed tool search plugin."""

TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": (
        "Search available deferred tools using semantic vector search and "
        "natural-language queries. Returns relevant tool names, descriptions, "
        "and similarity scores."
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
                    "capabilities or tools you need. Each query is searched independently."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of matching tools returned per query. "
                    "Default is 8 (min: 1, max: 50)."
                ),
            },
        },
        "required": ["queries"],
    },
}

# Backward compatibility alias
SEMANTIC_TOOL_SEARCH = {
    **TOOL_SEARCH_SCHEMA,
    "name": "semantic_tool_search",
}

