"""``dspy-sme`` CLI — thin entrypoint registered via ``[project.scripts]``.

Exists primarily so ``pip install -e .`` exposes a stable command. Subcommands:

* ``serve``   — start the uvicorn server.
* ``query``   — one-shot query against the expert service via DSPy.
* ``optimize`` — run the eval harness + optimizer (defers to ``scripts/optimize.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.config import settings

    uvicorn.run(
        "app.main:app",
        host=args.host or settings.API_HOST,
        port=args.port or settings.API_PORT,
        reload=args.reload,
        workers=args.workers,
    )
    return 0


def _query(args: argparse.Namespace) -> int:
    async def run() -> int:
        from app.experts import CodeExpert, GeneralExpert, MathExpert
        from app.llm import configure_dspy
        from app.schemas.base import ExpertDomain

        configure_dspy()
        cls = {
            ExpertDomain.MATH.value: MathExpert,
            ExpertDomain.CODE.value: CodeExpert,
            ExpertDomain.GENERAL.value: GeneralExpert,
        }[args.domain]
        expert = cls()
        await expert.initialize()
        result = await expert.generate(args.question, context={})
        print(result["response"])
        return 0

    return asyncio.run(run())


def _optimize(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parent.parent / "scripts" / "optimize.py"
    if not script.exists():
        print(f"optimize script not found at {script}", file=sys.stderr)
        return 2
    import runpy

    sys.argv = [str(script), *args.argv]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dspy-sme")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the FastAPI server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.add_argument("--workers", type=int, default=1)
    p_serve.set_defaults(func=_serve)

    p_q = sub.add_parser("query", help="One-shot query against an expert")
    p_q.add_argument("question")
    p_q.add_argument("--domain", choices=("math", "code", "general"), default="general")
    p_q.set_defaults(func=_query)

    p_o = sub.add_parser("optimize", help="Run the optimizer (delegates to scripts/optimize.py)")
    p_o.add_argument("argv", nargs=argparse.REMAINDER)
    p_o.set_defaults(func=_optimize)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
