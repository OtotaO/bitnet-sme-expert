"""Execution-based grading for the code domain.

The substring ``code_metric`` saturates (it only checks that a few tokens appear),
so it can't tell a correct solution from a plausible-looking wrong one. This
runs the model's generated code against committed assertions and grades on
*behavior* — the approach HumanEval-style code eval uses.

Safety: the code being run is produced by the model under eval (not adversarial
user input), but we still isolate defensively — a separate Python subprocess
with a wall-clock timeout, a scrubbed environment (no inherited secrets), a
throwaway working directory, and CPU/address-space rlimits where the platform
supports them. For grading genuinely untrusted input, use the Deno/Pyodide
sandbox in ``app/dspy_modules/code_module.py`` instead; this runner is scoped to
the eval harness. Opt-in via ``RUN_CODE_EXEC=1`` so it never runs untrusted code
in a default test run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

DEFAULT_TIMEOUT_S = 10.0
_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB address-space cap (POSIX only)


def _limit_resources() -> None:  # pragma: no cover - POSIX child-process hook
    """Best-effort CPU/memory caps applied in the child before exec."""
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_CPU, (int(DEFAULT_TIMEOUT_S) + 1, int(DEFAULT_TIMEOUT_S) + 1)
        )
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
    except Exception:
        pass  # not available on this platform; the timeout still bounds runaway


def run_code_test(code: str, test: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> bool:
    """Return True iff ``code`` + ``test`` runs to a clean exit within ``timeout``.

    ``test`` is assertion code that exercises whatever ``code`` defines; a failing
    assertion, an exception, a non-zero exit, or a timeout all grade as False.
    """
    program = f"{code}\n\n{test}\n"
    # Minimal env: keep PATH (interpreter needs it) but drop inherited secrets.
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    preexec = _limit_resources if os.name == "posix" else None
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=cwd,
                env=env,
                capture_output=True,
                timeout=timeout,
                preexec_fn=preexec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0


def code_exec_metric(example, prediction, *_, **__) -> float:
    """DSPy-compatible metric: 1.0 if the generated code passes the gold ``test``.

    Expects the gold example to carry a ``test`` field and the prediction a
    ``code`` field. Falls back to 0.0 if either is missing.
    """
    code = str(getattr(prediction, "code", ""))
    test = str(getattr(example, "test", ""))
    if not code or not test:
        return 0.0
    return float(run_code_test(code, test))
