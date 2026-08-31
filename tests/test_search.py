"""Unit and integration tests for FastEmbed tool search plugin."""

import json
import unittest
from unittest.mock import MagicMock

from schemas import TOOL_SEARCH_SCHEMA, SEMANTIC_TOOL_SEARCH
from tools import _extract_tool_info, FastEmbedSearchEngine, create_handler
import __init__ as plugin_init


class TestToolExtraction(unittest.TestCase):
    def test_extract_from_openai_dict(self):
        tool = {
            "type": "function",
            "function": {
                "name": "github_create_issue",
                "description": "Create a new issue in a GitHub repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        }
        info = _extract_tool_info(tool)
        self.assertEqual(info["name"], "github_create_issue")
        self.assertEqual(info["description"], "Create a new issue in a GitHub repository.")
        self.assertIn("repo", info["parameters"])
        self.assertIn("title", info["parameters"])
        self.assertIn("github_create_issue", info["search_text"])
        self.assertIn("parameters: repo, title, body", info["search_text"])

    def test_extract_from_flat_dict(self):
        tool = {
            "name": "docker_run_container",
            "description": "Start a new Docker container from an image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "ports": {"type": "array"},
                },
            },
        }
        info = _extract_tool_info(tool)
        self.assertEqual(info["name"], "docker_run_container")
        self.assertEqual(info["description"], "Start a new Docker container from an image.")
        self.assertIn("image", info["parameters"])

    def test_extract_from_object_instance(self):
        class DummyTool:
            name = "sql_query_db"
            description = "Run a SELECT query on Postgres."
            schema = {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }

        info = _extract_tool_info(DummyTool())
        self.assertEqual(info["name"], "sql_query_db")
        self.assertEqual(info["description"], "Run a SELECT query on Postgres.")
        self.assertIn("query", info["parameters"])


class TestFastEmbedSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FastEmbedSearchEngine()
        self.sample_tools = [
            _extract_tool_info({
                "name": "github_search_issues",
                "description": "Search GitHub issues by query string.",
                "parameters": {"properties": {"query": {}, "repo": {}}},
            }),
            _extract_tool_info({
                "name": "slack_send_message",
                "description": "Post a message to a Slack channel.",
                "parameters": {"properties": {"channel": {}, "text": {}}},
            }),
            _extract_tool_info({
                "name": "postgres_execute_sql",
                "description": "Execute arbitrary SQL against the primary database.",
                "parameters": {"properties": {"sql": {}}},
            }),
            _extract_tool_info({
                "name": "weather_get_forecast",
                "description": "Get 7-day weather forecast for a city.",
                "parameters": {"properties": {"city": {}}},
            }),
        ]

    def test_keyword_fallback(self):
        results = self.engine._keyword_search("github issue", self.sample_tools, limit=2)
        self.assertTrue(len(results) > 0)
        top_tool, score, method = results[0]
        self.assertEqual(top_tool["name"], "github_search_issues")
        self.assertEqual(method, "keyword_fallback")
        self.assertGreater(score, 0.0)

    def test_fingerprint_consistency_and_change(self):
        fp1 = self.engine._compute_fingerprint(self.sample_tools)
        fp2 = self.engine._compute_fingerprint(list(self.sample_tools))
        self.assertEqual(fp1, fp2)

        modified = list(self.sample_tools) + [_extract_tool_info({"name": "extra", "description": "extra"})]
        fp3 = self.engine._compute_fingerprint(modified)
        self.assertNotEqual(fp1, fp3)


class TestHandlerExecution(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "jira_find_tickets",
                "description": "Query Jira issues and bug tickets.",
                "parameters": {"properties": {"jql": {}}},
            },
            {
                "name": "s3_upload_file",
                "description": "Upload a binary file to an AWS S3 bucket.",
                "parameters": {"properties": {"bucket": {}, "key": {}}},
            },
        ]

    def test_handler_multi_query_and_response_format(self):
        ctx = MagicMock()
        ctx.get_tools.return_value = self.tools

        handler = create_handler(ctx)

        response_str = handler(
            args={"queries": ["jira bugs", "upload s3 file"], "limit": 5},
            available_tools=self.tools,
        )
        response = json.loads(response_str)

        self.assertIn("queries", response)
        self.assertEqual(len(response["queries"]), 2)
        self.assertIn("total_available", response)
        self.assertEqual(response["total_available"], 2)
        self.assertIn("results", response)
        self.assertEqual(len(response["results"]), 2)
        self.assertIn("tools", response)

        # Check first query matches
        res0 = response["results"][0]
        self.assertEqual(res0["query"], "jira bugs")
        self.assertIn("jira_find_tickets", res0["matches"])
        self.assertIn("jira_find_tickets", response["tools"])

    def test_handler_limit_clamping(self):
        ctx = MagicMock()
        ctx.get_tools.return_value = self.tools

        handler = create_handler(ctx)

        # limit = 1
        response_str = handler(
            args={"queries": ["tickets and s3"], "limit": 1},
            available_tools=self.tools,
        )
        response = json.loads(response_str)
        self.assertEqual(len(response["results"][0]["matches"]), 1)

    def test_handler_empty_query_error(self):
        ctx = MagicMock()
        handler = create_handler(ctx)

        # Kein queries-Feld
        response = json.loads(handler(args={}, available_tools=self.tools))
        self.assertIn("error", response)
        self.assertEqual(response["results"], [])
        self.assertEqual(response["total_available"], 0)

        # Leeres queries-Array
        response = json.loads(handler(args={"queries": ["", "   "]}, available_tools=self.tools))
        self.assertIn("error", response)
        self.assertEqual(response["results"], [])

        # Legacy-Feld "query" mit leerem String
        response = json.loads(handler(args={"query": ""}, available_tools=self.tools))
        self.assertIn("error", response)

    def test_large_catalog_1000_tools(self):
        # Generate 1000 synthetic tools across different domains
        large_toolset = [
            {
                "name": f"mcp_service_{i}_action",
                "description": f"Perform action #{i} for domain {'finance' if i % 3 == 0 else ('devops' if i % 3 == 1 else 'analytics')}",
                "parameters": {
                    "properties": {
                        "id": {"type": "integer"},
                        "filter_name": {"type": "string"},
                    }
                },
            }
            for i in range(1000)
        ]
        # Insert target specialized tool
        large_toolset.append({
            "name": "kubernetes_deploy_helm_chart",
            "description": "Deploy a specified Helm chart release to a Kubernetes cluster namespace.",
            "parameters": {"properties": {"chart": {}, "namespace": {}}},
        })

        ctx = MagicMock()
        ctx.get_tools.return_value = large_toolset
        handler = create_handler(ctx)

        response_str = handler(
            args={"queries": ["deploy helm release k8s"], "limit": 5},
            available_tools=large_toolset,
        )
        response = json.loads(response_str)

        self.assertEqual(response["total_available"], 1001)
        self.assertEqual(len(response["results"]), 1)
        self.assertIn("kubernetes_deploy_helm_chart", response["results"][0]["matches"])


class TestPluginRegistration(unittest.TestCase):
    def test_register_tool_search_override(self):
        ctx = MagicMock()
        plugin_init.register(ctx)

        ctx.register_tool.assert_called_once_with(
            name="tool_search",
            toolset="tools",
            schema=TOOL_SEARCH_SCHEMA,
            handler=unittest.mock.ANY,
            override=True,
        )


if __name__ == "__main__":
    unittest.main()
