import sys
import unittest
from types import ModuleType
from typing import ClassVar
from unittest.mock import patch

import tools as plugin_tools


class FakeResponse:
    def __init__(self, *, headers=None, payload=None, text=""):
        self.headers = headers or {}
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpxClient:
    calls: ClassVar[list[tuple[str, dict, str]]] = []

    def __init__(self, *, headers, timeout):
        self.headers = dict(headers)
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, *, json):
        self.calls.append((url, dict(self.headers), json["method"]))
        if json["method"] == "initialize":
            return FakeResponse(headers={"mcp-session-id": f"session-{url}"})
        if json["method"] == "tools/list":
            return FakeResponse(
                text="data: "
                + __import__("json").dumps(
                    {
                        "result": {
                            "tools": [
                                {
                                    "name": "list_items",
                                    "description": "List items",
                                    "inputSchema": {
                                        "properties": {"query": {"type": "string"}}
                                    },
                                }
                            ]
                        }
                    }
                )
            )
        return FakeResponse()


class TestLiveMcpDiscovery(unittest.TestCase):
    def setUp(self):
        plugin_tools._LIVE_TOOLS_CACHE.clear()
        FakeHttpxClient.calls = []

    def _configure(self, servers):
        hermes_cli = ModuleType("hermes_cli")
        config_module = ModuleType("hermes_cli.config")
        config_module.load_config = lambda: {"mcp_servers": servers}
        hermes_cli.config = config_module
        return patch.dict(
            sys.modules, {"hermes_cli": hermes_cli, "hermes_cli.config": config_module}
        )

    def test_queries_only_gateway_routes_and_uses_hermes_server_names(self):
        servers = {
            "agentgateway_infra": {"url": "https://mcp.example/infra/mcp"},
            "agentgateway_kita": {"url": "https://mcp.example/kita/mcp"},
            "agentgateway_home": {"url": "https://mcp.example/home/mcp"},
            "coolify": {"url": "https://coolify.example/mcp"},
        }
        with (
            self._configure(servers),
            patch.dict("os.environ", {"AGENTGATEWAY_BEARER_TOKEN": "test-token"}),
            patch.dict(sys.modules, {"httpx": ModuleType("httpx")}),
        ):
            sys.modules["httpx"].Client = FakeHttpxClient
            tools = plugin_tools._fetch_live_mcp_tools(groups="it-admin", ttl=0)

        self.assertEqual(
            {tool["name"] for tool in tools},
            {
                "mcp__agentgateway_infra__list_items",
                "mcp__agentgateway_kita__list_items",
                "mcp__agentgateway_home__list_items",
            },
        )
        self.assertEqual(len(FakeHttpxClient.calls), 9)
        self.assertTrue(
            all(
                "https://coolify.example" not in call[0]
                for call in FakeHttpxClient.calls
            )
        )
        self.assertTrue(
            all(
                call[1]["Authorization"] == "Bearer test-token"
                for call in FakeHttpxClient.calls
            )
        )
        self.assertTrue(
            all(
                call[1]["X-User-Groups"] == "it-admin" for call in FakeHttpxClient.calls
            )
        )

    def test_legacy_agentgateway_entry_remains_supported(self):
        servers = {"agentgateway": {"url": "https://mcp.example/mcp"}}
        with (
            self._configure(servers),
            patch.dict("os.environ", {"AGENTGATEWAY_BEARER_TOKEN": "test-token"}),
            patch.dict(sys.modules, {"httpx": ModuleType("httpx")}),
        ):
            sys.modules["httpx"].Client = FakeHttpxClient
            tools = plugin_tools._fetch_live_mcp_tools(ttl=0)

        self.assertEqual(tools[0]["name"], "mcp__agentgateway__list_items")

    def test_one_route_failure_does_not_hide_other_route_tools(self):
        class PartiallyFailingClient(FakeHttpxClient):
            def post(self, url, *, json):
                if "kita" in url:
                    raise RuntimeError("route unavailable")
                return super().post(url, json=json)

        servers = {
            "agentgateway_infra": {"url": "https://mcp.example/infra/mcp"},
            "agentgateway_kita": {"url": "https://mcp.example/kita/mcp"},
        }
        with (
            self._configure(servers),
            patch.dict("os.environ", {"AGENTGATEWAY_BEARER_TOKEN": "test-token"}),
            patch.dict(sys.modules, {"httpx": ModuleType("httpx")}),
        ):
            sys.modules["httpx"].Client = PartiallyFailingClient
            tools = plugin_tools._fetch_live_mcp_tools(ttl=0)

        self.assertEqual(
            [tool["name"] for tool in tools], ["mcp__agentgateway_infra__list_items"]
        )


if __name__ == "__main__":
    unittest.main()
