#!/usr/bin/env bash
# Start the bitnet.cpp llama-server (OpenAI-compatible).
set -euo pipefail

BITNET_DIR="${BITNET_DIR:-$HOME/.cache/bitnet.cpp}"
MODEL_REPO="${BITNET_MODEL_REPO:-microsoft/BitNet-b1.58-2B-4T-gguf}"
MODEL_FILE="${BITNET_MODEL_FILE:-ggml-model-i2_s.gguf}"
HOST="${BITNET_HOST:-127.0.0.1}"
PORT="${BITNET_PORT:-8080}"

MODEL_PATH="$BITNET_DIR/models/$(basename "$MODEL_REPO")/$MODEL_FILE"
SERVER="$BITNET_DIR/build/bin/llama-server"

if [ ! -x "$SERVER" ]; then
  echo "llama-server not found at $SERVER. Run scripts/bitnet_setup.sh first." >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH. Run scripts/bitnet_setup.sh first." >&2
  exit 1
fi

exec "$SERVER" \
  --model "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  --parallel 1 --ctx-size 4096 \
  --alias bitnet
