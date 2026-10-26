"""Unit tests for the Ollama-backed semantic tool-search plugin."""

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
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
                    "required": ["repo", "title"],
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
        self.assertEqual(info["required"], ["repo", "title"])
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

    @staticmethod
    def _core_scope_modules(deferrable, *, connections=False, remote_entries=None):
        tools_package = ModuleType("tools")
        tools_package.__path__ = []
        core_search = ModuleType("tools.tool_search")
        core_search._deferrable_in = deferrable
        core_search._record_search_metric = MagicMock()
        core_search.load_config = lambda: SimpleNamespace(
            search_default_limit=5,
            max_search_limit=25,
        )
        core_search._shared_tool_record = lambda entry: {
            "source": entry.source,
            "source_name": entry.source_name,
            "description": entry.description,
            "required": ["query"],
        }
        tools_package.tool_search = core_search

        connectors_package = ModuleType("tools.connectors")
        connectors_package.__path__ = []
        connectors_search = ModuleType("tools.connectors.search")
        connectors_search.connections_in_scope = lambda _defs: connections
        connectors_search.connector_entries_by_group = (
            lambda queries, connector_search=None, call=None: (
                remote_entries or [[] for _ in queries],
                None,
            )
        )
        connectors_search.connectors_unavailable = lambda *args, **kwargs: {}
        connectors_package.search = connectors_search
        tools_package.connectors = connectors_package
        return {
            "tools": tools_package,
            "tools.tool_search": core_search,
            "tools.connectors": connectors_package,
            "tools.connectors.search": connectors_search,
        }

    def test_native_scope_search_excludes_non_deferred_tools(self):
        ctx = MagicMock()
        ctx.get_tools.return_value = [
            {"name": "mcp__cache__unregistered_tool", "description": "stale cache entry"}
        ]
        definitions = [
            {"type": "function", "function": {"name": "terminal", "description": "Search vaultwarden logins"}},
            {"type": "function", "function": {"name": "mcp__agentgateway_infra__vaultwarden_search", "description": "Search Vaultwarden login entries", "parameters": {"properties": {"query": {}}}}},
        ]
        modules = self._core_scope_modules(
            lambda defs: [
                item for item in defs
                if item["function"]["name"].startswith("mcp__")
            ]
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(plugin_tools, "_ENGINE", self._keyword_engine()),
        ):
            response = json.loads(
                create_handler(ctx)(
                    args={"queries": ["vaultwarden login"]},
                    tools=definitions,
                    _scope_tool_defs=True,
                )
            )

        self.assertEqual(response["total_available"], 1)
        self.assertEqual(
            response["results"][0]["matches"],
            ["mcp__agentgateway_infra__vaultwarden_search"],
        )
        self.assertNotIn("terminal", response["tools"])
        self.assertNotIn("mcp__cache__unregistered_tool", response["tools"])

    def test_native_scope_honors_core_result_limit(self):
        ctx = MagicMock()
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": f"mcp__agentgateway__vaultwarden_search_{index}",
                    "description": "Search Vaultwarden login entries",
                },
            }
            for index in range(4)
        ]
        modules = self._core_scope_modules(lambda defs: defs)
        config = SimpleNamespace(search_default_limit=2, max_search_limit=3)
        with (
            patch.dict(sys.modules, modules),
            patch.object(plugin_tools, "_ENGINE", self._keyword_engine()),
        ):
            response = json.loads(
                create_handler(ctx)(
                    args={"queries": ["vaultwarden login"], "limit": 10},
                    tools=definitions,
                    config=config,
                    _scope_tool_defs=True,
                )
            )

        self.assertEqual(response["total_available"], 4)
        self.assertEqual(len(response["results"][0]["matches"]), 3)

    def test_native_scope_search_merges_hosted_connectors_with_call_context(self):
        ctx = MagicMock()
        call_ref = object()
        connector_search = object()
        definitions = [
            {"type": "function", "function": {"name": "manage_connections", "description": "Manage connectors"}},
            {"type": "function", "function": {"name": "mcp__agentgateway_infra__vaultwarden_search", "description": "Search Vaultwarden login entries", "parameters": {"properties": {"query": {}}}}},
        ]
        remote_entry = type(
            "CatalogEntry",
            (),
            {
                "name": "connector__gmail__search_messages",
                "description": "Search Gmail messages",
                "source": "connectors",
                "source_name": "gmail",
                "schema": {"function": {"parameters": {"required": ["query"]}}},
            },
        )()
        connector_calls = []

        def connector_entries(queries, *, connector_search=None, call=None):
            connector_calls.append((queries, connector_search, call))
            return [[remote_entry] for _ in queries], None

        modules = self._core_scope_modules(
            lambda defs: [item for item in defs if item["function"]["name"].startswith("mcp__")],
            connections=True,
        )
        modules["tools.connectors.search"].connector_entries_by_group = connector_entries
        with patch.dict(sys.modules, modules), patch.object(
            plugin_tools, "_ENGINE", self._keyword_engine()
        ):
            response = json.loads(
                create_handler(ctx)(
                    args={"queries": ["vaultwarden login"]},
                    tools=definitions,
                    connector_search=connector_search,
                    call=call_ref,
                    _scope_tool_defs=True,
                )
            )

        self.assertEqual(len(connector_calls), 1)
        self.assertIs(connector_calls[0][1], connector_search)
        self.assertIs(connector_calls[0][2], call_ref)
        self.assertIn("connector__gmail__search_messages", response["tools"])
        self.assertIn(
            "connector__gmail__search_messages",
            response["results"][0]["matches"],
        )


class TestNativeBridgeCallCompatibility(unittest.TestCase):
    @staticmethod
    def _fake_tools_modules(native_dispatch):
        package = ModuleType("tools")
        package.__path__ = []
        module = ModuleType("tools.tool_search")
        module.dispatch_tool_search = native_dispatch
        module._smart_filter_patched = False
        package.tool_search = module
        return package, module

    def test_core_call_reference_reaches_semantic_handler(self):
        package, module = self._fake_tools_modules(lambda *_args, **_kwargs: "native")
        handler_calls = []
        call_ref = object()
        connector_search = object()
        tool_defs = [{"function": {"name": "mcp__gateway__list_tools"}}]

        def handler(**kwargs):
            handler_calls.append(kwargs)
            return "semantic"

        with (
            patch.dict(sys.modules, {"tools": package, "tools.tool_search": module}),
            patch.object(plugin_init, "_write_audit_log"),
        ):
            plugin_init._patch_native_bridge(handler)
            result = module.dispatch_tool_search(
                {"queries": ["list tools"]},
                current_tool_defs=tool_defs,
                connector_search=connector_search,
                call=call_ref,
            )

        self.assertEqual(result, "semantic")
        self.assertEqual(len(handler_calls), 1)
        self.assertIs(handler_calls[0]["call"], call_ref)
        self.assertIs(handler_calls[0]["connector_search"], connector_search)
        self.assertIs(handler_calls[0]["tools"], tool_defs)

    def test_legacy_native_dispatcher_fallback_drops_unsupported_call_kwarg(self):
        native_calls = []

        def legacy_dispatch(
            args,
            *,
            current_tool_defs,
            config=None,
            connector_search=None,
        ):
            native_calls.append((args, current_tool_defs, connector_search))
            return "native-fallback"

        package, module = self._fake_tools_modules(legacy_dispatch)

        def failing_handler(**_kwargs):
            raise RuntimeError("semantic search failed")

        call_ref = object()
        connector_search = object()
        tool_defs = [{"function": {"name": "mcp__gateway__list_tools"}}]
        with (
            patch.dict(sys.modules, {"tools": package, "tools.tool_search": module}),
            patch.object(plugin_init, "_write_audit_log"),
        ):
            plugin_init._patch_native_bridge(failing_handler)
            result = module.dispatch_tool_search(
                {"queries": ["list tools"]},
                current_tool_defs=tool_defs,
                connector_search=connector_search,
                call=call_ref,
            )

        self.assertEqual(result, "native-fallback")
        self.assertEqual(native_calls, [({"queries": ["list tools"]}, tool_defs, connector_search)])


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
