#!/usr/bin/env bash
# One-command BitNet throughput demo.
#
# Sends a fixed prompt through the *DSPy path* (a `dspy.LM` pointed at the local
# bitnet.cpp llama-server, exactly as the experts use it), measures wall-clock
# time and the server-reported completion-token count, and prints the measured
# tokens/sec. Writes a dated receipt to eval/receipts/bitnet-<date>.txt.
#
# Prereq: a running server — `make bitnet-setup && make bitnet-serve` first.
# This must run on real hardware (the build is heavy); it is not a CI job.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${BITNET_HOST:-127.0.0.1}"
PORT="${BITNET_PORT:-8080}"
API_BASE="http://${HOST}:${PORT}/v1"
MODEL="${BITNET_ALIAS:-bitnet}"          # matches --alias in bitnet_serve.sh
PROMPT="${BITNET_PROMPT:-Explain in one paragraph why 1-bit weight quantization reduces memory bandwidth.}"
MAX_TOKENS="${BITNET_MAX_TOKENS:-256}"

# Skip gracefully (exit 0) when no server is reachable, so this script is safe
# to run in CI / on a clean checkout with no model present. The measured-tok/s
# path only runs on real hardware with a live server.
if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 \
   && ! curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
  echo "SKIP: no bitnet.cpp llama-server reachable at http://${HOST}:${PORT}." >&2
  echo "      This demo measures real tok/s and needs a live server + model." >&2
  echo "      Build and start one first: 'make bitnet-setup && make bitnet-serve'." >&2
  echo "      Skipping (no measurement produced); this is expected with no model." >&2
  exit 0
fi

STAMP="$(date +%Y-%m-%d)"
RECEIPT="eval/receipts/bitnet-${STAMP}.txt"
mkdir -p eval/receipts

API_BASE="$API_BASE" MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" \
RECEIPT="$RECEIPT" STAMP="$STAMP" uv run python - <<'PY'
import os
import time

import dspy

api_base = os.environ["API_BASE"]
model = os.environ["MODEL"]
prompt = os.environ["PROMPT"]
max_tokens = int(os.environ["MAX_TOKENS"])
receipt = os.environ["RECEIPT"]
stamp = os.environ["STAMP"]

# The DSPy path: the same dspy.LM abstraction the experts route through, just
# pointed at the local OpenAI-compatible bitnet.cpp server.
lm = dspy.LM(
    f"openai/{model}",
    api_base=api_base,
    api_key=os.environ.get("BITNET_DUMMY_KEY", "local"),
    max_tokens=max_tokens,
    temperature=0.0,
    cache=False,
)

t0 = time.perf_counter()
lm(prompt)
elapsed = time.perf_counter() - t0

# Prefer the server-reported completion token count; fall back to a clearly
# labelled word-count estimate only if usage is absent.
usage = (lm.history[-1].get("usage") or {}) if lm.history else {}
completion_tokens = usage.get("completion_tokens")
basis = "server-reported completion_tokens"
if not completion_tokens:
    text = ""
    if lm.history:
        out = lm.history[-1].get("outputs") or []
        text = out[0] if out else ""
    completion_tokens = max(1, len(str(text).split()))
    basis = "word-count ESTIMATE (server returned no usage)"

tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0

lines = [
    f"BitNet throughput demo — {stamp}",
    f"endpoint:           {api_base} (model={model})",
    f"prompt:             {prompt!r}",
    f"max_tokens:         {max_tokens}",
    f"completion_tokens:  {completion_tokens}  [{basis}]",
    f"elapsed_seconds:    {elapsed:.3f}",
    f"measured_tok_s:     {tok_s:.2f}",
]
report = "\n".join(lines) + "\n"
print(report)
with open(receipt, "w") as fh:
    fh.write(report)
print(f"Wrote receipt to {receipt}")
PY
