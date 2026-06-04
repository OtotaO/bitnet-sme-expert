"""MCP server tests via the SDK's in-memory transport (no subprocess, no LM).

``list_experts`` exercises the full build → register → initialize path without
any provider call, so it runs offline. ``ask_expert`` needs an LM and is covered
by the eval harness instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="requires the [mcp] extra")

from mcp.shared.memory import create_connected_server_and_client_session as connect

from app.mcp_server import mcp


async def test_tools_registered() -> None:
    async with connect(mcp._mcp_server) as client:
        await client.initialize()
        tools = await client.list_tools()
        names = {t.name for t in tools.tools}
        assert {"ask_expert", "list_experts"} <= names


async def test_ask_expert_schema_has_optional_domain() -> None:
    async with connect(mcp._mcp_server) as client:
        await client.initialize()
        tools = await client.list_tools()
        ask = next(t for t in tools.tools if t.name == "ask_expert")
        props = ask.inputSchema["properties"]
        assert "question" in props
        assert "domain" in props
        # `question` is required; `domain` is not.
        assert ask.inputSchema.get("required", []) == ["question"]


async def test_list_experts_tool_runs_offline() -> None:
    async with connect(mcp._mcp_server) as client:
        await client.initialize()
        result = await client.call_tool("list_experts", {})
        assert not result.isError
        text = " ".join(getattr(b, "text", "") for b in result.content)
        assert "Expert" in text  # e.g. "Math Expert"
