#!/usr/bin/env bash
# Build bitnet.cpp and download the BitNet b1.58 2B 4T model.
#
# After this script finishes, an OpenAI-compatible llama-server can be started
# (see ./scripts/bitnet_serve.sh) which dspy.LM can call via:
#
#     export DSPY_LM_MATH="openai/bitnet-b1.58-2B-4T"
#     export DSPY_LM_MATH_API_BASE="http://localhost:8080/v1"
#     export DSPY_LM_MATH_API_KEY_ENV="BITNET_DUMMY_KEY"
#     export BITNET_DUMMY_KEY="local"
#
# bitnet.cpp inherits its HTTP server from llama.cpp, so the API surface is
# /v1/chat/completions, /v1/completions, /v1/models.

set -euo pipefail

BITNET_DIR="${BITNET_DIR:-$HOME/.cache/bitnet.cpp}"
MODEL_REPO="${BITNET_MODEL_REPO:-microsoft/BitNet-b1.58-2B-4T-gguf}"
MODEL_FILE="${BITNET_MODEL_FILE:-ggml-model-i2_s.gguf}"

echo "=> Cloning bitnet.cpp into $BITNET_DIR"
if [ ! -d "$BITNET_DIR/.git" ]; then
  git clone --recursive https://github.com/microsoft/BitNet.git "$BITNET_DIR"
else
  echo "   (already cloned; pulling latest)"
  git -C "$BITNET_DIR" pull --rebase
  git -C "$BITNET_DIR" submodule update --init --recursive
fi

echo "=> Setting up Python deps + building llama-server"
cd "$BITNET_DIR"
python -m pip install -r requirements.txt
python setup_env.py -md "models/$(basename "$MODEL_REPO")" -q i2_s

# After setup_env.py the cmake build is in ./build with the llama-server binary.
if [ ! -f "$BITNET_DIR/build/bin/llama-server" ]; then
  echo "ERROR: llama-server was not built. See $BITNET_DIR/build for details." >&2
  exit 1
fi

echo "=> Done."
echo "   Model dir: $BITNET_DIR/models/$(basename "$MODEL_REPO")"
echo "   Server:    $BITNET_DIR/build/bin/llama-server"
echo
echo "Start the server with:"
echo "  $BITNET_DIR/build/bin/llama-server \\"
echo "    --model $BITNET_DIR/models/$(basename "$MODEL_REPO")/$MODEL_FILE \\"
echo "    --host 127.0.0.1 --port 8080 --parallel 1 --ctx-size 4096"
