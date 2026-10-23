"""Unit tests for the Ollama-backed semantic tool-search plugin."""

import json
import os
import sys
import unittest
from types import ModuleType
from typing import ClassVar
from unittest.mock import MagicMock, patch

import __init__ as plugin_init
import tools as plugin_tools
from schemas import TOOL_SEARCH_SCHEMA
from tools import (
    OllamaSearchEngine,
    _extract_tool_info,
    _normalize_ollama_root,
    _resolve_ollama_connection,
    create_handler,
)


class FakeResponse:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def raise_for_status(self):
        return None

    def json(self):
        return {"embeddings": self._embeddings}


class FakeOllamaClient:
    def __init__(self, requests, encoder, timeout):
        self.requests = requests
        self.encoder = encoder
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, *, headers, json):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse([self.encoder(text) for text in json["input"]])


class KeywordOnlyEngine:
    model_name = "test-keyword-fallback"

    def search(self, query, tools, limit=8, min_score=0.25):
        return OllamaSearchEngine._keyword_search(query, tools, limit=limit)


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
        self.assertEqual(
            info["description"], "Create a new issue in a GitHub repository."
        )
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
        self.assertEqual(
            info["description"], "Start a new Docker container from an image."
        )
        self.assertIn("image", info["parameters"])

    def test_extract_from_object_instance(self):
        class DummyTool:
            name = "sql_query_db"
            description = "Run a SELECT query on Postgres."
            schema: ClassVar[dict] = {"type": "object", "properties": {"query": {}}}

        info = _extract_tool_info(DummyTool())
        self.assertEqual(info["name"], "sql_query_db")
        self.assertEqual(info["description"], "Run a SELECT query on Postgres.")
        self.assertIn("query", info["parameters"])


class TestOllamaSearchEngine(unittest.TestCase):
    def setUp(self):
        FakeOllamaClient.requests = []
        self.sample_tools = [
            _extract_tool_info(
                {
                    "name": "github_search_issues",
                    "description": "Search GitHub issues by query string.",
                    "parameters": {"properties": {"query": {}, "repo": {}}},
                }
            ),
            _extract_tool_info(
                {
                    "name": "slack_send_message",
                    "description": "Post a message to a Slack channel.",
                    "parameters": {"properties": {"channel": {}, "text": {}}},
                }
            ),
            _extract_tool_info(
                {
                    "name": "postgres_execute_sql",
                    "description": "Execute SQL against the primary database.",
                    "parameters": {"properties": {"sql": {}}},
                }
            ),
            _extract_tool_info(
                {
                    "name": "weather_get_forecast",
                    "description": "Get a 7-day weather forecast for a city.",
                    "parameters": {"properties": {"city": {}}},
                }
            ),
        ]

    @staticmethod
    def _encoder(text):
        lowered = text.lower()
        if (
            "file a defect" in lowered
            or "github_search_issues" in lowered
            or "github issues" in lowered
        ):
            return [1.0, 0.0, 0.0]
        if "slack" in lowered:
            return [0.0, 1.0, 0.0]
        if "postgres" in lowered or "weather" in lowered:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 1.0]

    def _engine(self, **kwargs):
        return OllamaSearchEngine(
            model_name="qwen3-embedding:test",
            connection_resolver=lambda: (
                "https://ollama.example/api/embed",
                {"Authorization": "Bearer test-only"},
            ),
            client_factory=lambda timeout: FakeOllamaClient(
                FakeOllamaClient.requests, self._encoder, timeout
            ),
            **kwargs,
        )

    def test_semantic_search_batches_catalog_and_caches_vectors(self):
        engine = self._engine()
        engine.batch_size = 2

        # The query intentionally shares no keywords with the GitHub tool.
        results = engine.search(
            "file a defect in the code hosting platform", self.sample_tools, limit=2
        )

        self.assertEqual(results[0][0]["name"], "github_search_issues")
        self.assertEqual(results[0][2], "ollama")
        self.assertEqual(
            len(FakeOllamaClient.requests), 3
        )  # two catalog batches + query
        first_request = FakeOllamaClient.requests[0]
        self.assertEqual(first_request["url"], "https://ollama.example/api/embed")
        self.assertEqual(first_request["headers"]["Authorization"], "Bearer test-only")
        self.assertEqual(first_request["json"]["model"], "qwen3-embedding:test")
        self.assertEqual(first_request["json"]["options"]["num_gpu"], 0)
        self.assertEqual(first_request["json"]["options"]["num_thread"], 1)

        engine.search(
            "file a defect in the code hosting platform", self.sample_tools, limit=2
        )
        self.assertEqual(
            len(FakeOllamaClient.requests), 4
        )  # only the query is re-embedded

    def test_large_catalog_is_embedded_in_batches(self):
        large_tools = [
            {
                "name": f"mcp_tool_{index}",
                "description": f"Unrelated capability number {index}",
                "search_text": f"mcp_tool_{index} - Unrelated capability number {index}",
                "parameters": [],
            }
            for index in range(1000)
        ]
        large_tools.append(
            {
                "name": "kubernetes_deploy_helm_chart",
                "description": "Deploy a Helm chart to Kubernetes.",
                "search_text": "kubernetes_deploy_helm_chart - Deploy a Helm chart to Kubernetes.",
                "parameters": [],
            }
        )

        def encoder(text):
            return (
                [1.0, 0.0]
                if "kubernetes_deploy_helm_chart" in text or "deploy the chart" in text
                else [0.0, 1.0]
            )

        engine = OllamaSearchEngine(
            connection_resolver=lambda: ("http://ollama.example/api/embed", {}),
            client_factory=lambda timeout: FakeOllamaClient(
                FakeOllamaClient.requests, encoder, timeout
            ),
        )
        results = engine.search("deploy the chart", large_tools, limit=5)

        self.assertEqual(results[0][0]["name"], "kubernetes_deploy_helm_chart")
        self.assertEqual(results[0][2], "ollama")
        self.assertEqual(
            len(FakeOllamaClient.requests), 17
        )  # 16 catalog batches + query
        self.assertTrue(
            all(
                len(request["json"]["input"]) <= 64
                for request in FakeOllamaClient.requests
            )
        )

    def test_reuses_hermes_custom_ollama_provider_and_secret(self):
        hermes_cli = ModuleType("hermes_cli")
        hermes_cli.__path__ = []
        hermes_config = ModuleType("hermes_cli.config")
        hermes_config.load_config = lambda: {
            "custom_providers": [
                {
                    "name": "ollama",
                    "base_url": "https://configured.example/v1",
                    "key_env": "OLLAMA_CONFIG_KEY",
                }
            ]
        }
        hermes_cli.config = hermes_config

        agent = ModuleType("agent")
        agent.__path__ = []
        secret_scope = ModuleType("agent.secret_scope")
        secret_scope.get_secret_str = lambda key, default="": (
            "configured-test-key" if key == "OLLAMA_CONFIG_KEY" else default
        )
        agent.secret_scope = secret_scope

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules,
                {
                    "hermes_cli": hermes_cli,
                    "hermes_cli.config": hermes_config,
                    "agent": agent,
                    "agent.secret_scope": secret_scope,
                },
            ),
        ):
            endpoint, headers = _resolve_ollama_connection()

        self.assertEqual(endpoint, "https://configured.example/api/embed")
        self.assertEqual(headers["Authorization"], "Bearer configured-test-key")

    def test_old_fastembed_model_setting_is_not_used_for_ollama(self):
        with patch.dict(
            os.environ, {"SMART_FILTER_MODEL": "BAAI/bge-small-en-v1.5"}, clear=True
        ):
            engine = OllamaSearchEngine()
        self.assertEqual(engine.model_name, "qwen3-embedding:latest")

    def test_explicit_base_url_and_key_are_normalized(self):
        with patch.dict(
            os.environ,
            {
                "SMART_FILTER_OLLAMA_BASE_URL": "https://ollama.example/v1/",
                "SMART_FILTER_OLLAMA_API_KEY": "test-only",
            },
            clear=True,
        ):
            endpoint, headers = _resolve_ollama_connection()
        self.assertEqual(endpoint, "https://ollama.example/api/embed")
        self.assertEqual(headers["Authorization"], "Bearer test-only")
        self.assertEqual(
            _normalize_ollama_root("https://ollama.example/v1"),
            "https://ollama.example",
        )

    def test_endpoint_failure_uses_keyword_fallback(self):
        class FailingClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def post(self, *_args, **_kwargs):
                raise TimeoutError("simulated timeout")

        engine = OllamaSearchEngine(
            connection_resolver=lambda: ("http://ollama.example/api/embed", {}),
            client_factory=FailingClient,
        )
        results = engine.search("github issue", self.sample_tools, limit=2)
        self.assertEqual(results[0][0]["name"], "github_search_issues")
        self.assertEqual(results[0][2], "keyword_fallback")

    def test_fingerprint_changes_with_catalog(self):
        original = self.sample_tools
        changed = original + [
            _extract_tool_info({"name": "extra_tool", "description": "extra"})
        ]
        self.assertEqual(
            OllamaSearchEngine._compute_fingerprint(original),
            OllamaSearchEngine._compute_fingerprint(list(original)),
        )
        self.assertNotEqual(
            OllamaSearchEngine._compute_fingerprint(original),
            OllamaSearchEngine._compute_fingerprint(changed),
        )

    def test_keyword_fallback(self):
        results = OllamaSearchEngine._keyword_search(
            "github issue", self.sample_tools, limit=2
        )
        self.assertTrue(results)
        self.assertEqual(results[0][0]["name"], "github_search_issues")
        self.assertEqual(results[0][2], "keyword_fallback")
        self.assertGreater(results[0][1], 0.0)


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

    def _keyword_engine(self):
        class Engine:
            model_name = "test-keyword-fallback"

            def search(self, query, tools, limit=8, min_score=0.25):
                return OllamaSearchEngine._keyword_search(query, tools, limit=limit)

        return Engine()

    def test_handler_multi_query_and_response_format(self):
        ctx = MagicMock()
        ctx.get_tools.return_value = self.tools

        with patch.object(plugin_tools, "_ENGINE", self._keyword_engine()):
            handler = create_handler(ctx)
            response = json.loads(
                handler(
                    args={"queries": ["jira bugs", "upload s3 file"], "limit": 5},
                    available_tools=self.tools,
                )
            )

        self.assertEqual(len(response["queries"]), 2)
        self.assertEqual(response["total_available"], 2)
        self.assertEqual(len(response["results"]), 2)
        self.assertIn("jira_find_tickets", response["results"][0]["matches"])
        self.assertIn("jira_find_tickets", response["tools"])

    def test_handler_limit_clamping(self):
        ctx = MagicMock()
        ctx.get_tools.return_value = self.tools

        with patch.object(plugin_tools, "_ENGINE", self._keyword_engine()):
            handler = create_handler(ctx)
            response = json.loads(
                handler(
                    args={"queries": ["tickets and s3"], "limit": 1},
                    available_tools=self.tools,
                )
            )

        self.assertEqual(len(response["results"][0]["matches"]), 1)

    def test_handler_empty_query_error(self):
        ctx = MagicMock()
        handler = create_handler(ctx)

        response = json.loads(handler(args={}, available_tools=self.tools))
        self.assertIn("error", response)
        self.assertEqual(response["results"], [])

        response = json.loads(
            handler(args={"queries": ["", "   "]}, available_tools=self.tools)
        )
        self.assertIn("error", response)
        self.assertEqual(response["results"], [])

        response = json.loads(handler(args={"query": ""}, available_tools=self.tools))
        self.assertIn("error", response)

    def test_large_catalog_1000_tools(self):
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
        large_toolset.append(
            {
                "name": "kubernetes_deploy_helm_chart",
                "description": "Deploy a specified Helm chart release to a Kubernetes cluster namespace.",
                "parameters": {"properties": {"chart": {}, "namespace": {}}},
            }
        )

        ctx = MagicMock()
        ctx.get_tools.return_value = large_toolset
        with patch.object(plugin_tools, "_ENGINE", self._keyword_engine()):
            handler = create_handler(ctx)
            response = json.loads(
                handler(
                    args={"queries": ["deploy helm release k8s"], "limit": 5},
                    available_tools=large_toolset,
                )
            )

        self.assertEqual(response["total_available"], 1001)
        self.assertEqual(len(response["results"]), 1)
        self.assertIn("kubernetes_deploy_helm_chart", response["results"][0]["matches"])


class TestPluginRegistration(unittest.TestCase):
    def test_register_tool_search_override(self):
        ctx = MagicMock()
        with (
            patch.object(plugin_init, "_pre_warm_engine_in_background"),
            patch.object(plugin_init, "_patch_native_bridge"),
        ):
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
