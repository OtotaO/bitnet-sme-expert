"""MCP server exposing the SME experts as Model Context Protocol tools.

Uses the official `mcp` SDK (`mcp.server.fastmcp.FastMCP`) — not the third-party
`fastmcp` package. MCP is the de-facto standard for surfacing tools to agent
hosts (Claude Desktop, IDEs, etc.), so this lets any MCP client route questions
to the math / code / general experts behind the same DSPy programs the HTTP API
uses.

Run it over stdio (the usual transport for local MCP hosts):

    uv run dspy-sme-mcp          # console script
    uv run python -m app.mcp_server

Requires the ``[mcp]`` extra: ``uv sync --extra mcp``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.schemas.base import ExpertDomain
from app.services.expert_service import ExpertService

mcp = FastMCP("dspy-sme-expert")

_service: ExpertService | None = None

# Valid domain strings for the ``domain`` argument, surfaced in error messages.
_DOMAINS = tuple(d.value for d in ExpertDomain)


async def _get_service() -> ExpertService:
    """Build the expert service once, lazily, on first tool call."""
    global _service  # noqa: PLW0603 — lazy module-level singleton
    if _service is None:
        from app.bootstrap import build_expert_service

        _service = await build_expert_service()
    return _service


@mcp.tool()
async def ask_expert(question: str, domain: str | None = None) -> str:
    """Answer a question with the best-matching subject-matter expert.

    Domains: ``math`` (symbolic/calculus via sympy), ``code`` (Python generation),
    ``general`` (open-ended knowledge). Omit ``domain`` to let the router classify
    the question automatically.
    """
    if domain is not None and domain not in _DOMAINS:
        return f"Unknown domain {domain!r}. Valid: {', '.join(_DOMAINS)} (or omit to auto-route)."

    service = await _get_service()
    if domain is not None:
        target = ExpertDomain(domain)
    else:
        target, _confidence = await service.route(question)

    experts = await service.get_experts_by_domain(target)
    if not experts:
        all_experts = await service.get_all_experts()
        if not all_experts:
            return "No experts are available."
        expert = next(iter(all_experts.values()))
    else:
        expert = experts[0]

    result = await expert.generate(question, context={})
    return str(result.get("response", ""))


@mcp.tool()
async def list_experts() -> list[dict]:
    """List the registered domain experts and the model each is backed by."""
    service = await _get_service()
    return await service.list_experts()


def main() -> None:
    """Console-script entry point — serve over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
