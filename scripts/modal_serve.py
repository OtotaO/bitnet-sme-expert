"""Deploy any HF base model on Modal as an OpenAI-compatible vLLM endpoint.

Usage (one-off):

    modal serve scripts/modal_serve.py

Usage (deploy as a persistent service):

    modal deploy scripts/modal_serve.py

Configuration is via env vars at deploy time (set them in your shell, Modal
captures them into the app):

* ``MODEL_NAME``         — HF repo id (default ``Qwen/Qwen2.5-7B-Instruct``).
* ``GPU``                — Modal GPU spec (default ``H100``; try ``A10G`` for
                           smaller bases or ``H100:2`` for tensor-parallel).
* ``MIN_CONTAINERS``     — keep this many warm (default ``0`` = cold-start ok).
* ``MAX_CONTAINERS``     — autoscale ceiling (default ``1``).
* ``API_KEY``            — bearer token clients must send (default
                           ``local-dev`` — set something real for prod).
* ``VLLM_VERSION``       — pin (default ``0.7.0``).

Once deployed, Modal prints a URL like ``https://<workspace>--vllm-serve.modal.run``.
Point DSPy at it via:

    export DSPY_LM_MATH=openai/Qwen/Qwen2.5-7B-Instruct
    export DSPY_LM_MATH_API_BASE=https://<workspace>--vllm-serve.modal.run/v1
    export DSPY_LM_MATH_API_KEY=<the-API_KEY-you-set>

The endpoint is OpenAI-compatible so it Just Works with LiteLLM and DSPy
through the standard ``openai/<model>`` provider — no custom adapter needed.

Mirrors the BitNet local-serve story in ``scripts/bitnet_serve.sh`` for the
hosted-Modal substrate.
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError as exc:  # pragma: no cover - script-time only
    raise SystemExit(
        "modal is not installed. `pip install modal` or `uv pip install modal`, "
        "then `modal setup` once to authenticate."
    ) from exc


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("GPU", "H100")
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "1"))
API_KEY = os.environ.get("API_KEY", "local-dev")
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.7.0")
VLLM_PORT = 8000
TIMEOUT_S = 30 * 60  # 30 min idle timeout

app = modal.App("dspy-sme-vllm")

# A Modal Volume caches HF weights across container restarts so cold-starts
# stay quick (~30s) instead of re-downloading the model every time.
WEIGHTS_VOLUME = modal.Volume.from_name("dspy-sme-vllm-weights", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]>=0.27",
        "fastapi>=0.115",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE: WEIGHTS_VOLUME},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=TIMEOUT_S,
    scaledown_window=60 * 5,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve() -> None:
    """Boot vLLM's OpenAI-compatible HTTP server on the container."""
    import subprocess

    # Binding 0.0.0.0 inside the Modal container is required — Modal handles
    # the public TLS surface in front of us. noqa: S104.
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--host",
        "0.0.0.0",  # noqa: S104 - container-internal; Modal proxies TLS
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
        "--served-model-name",
        MODEL_NAME,
        # Sensible defaults; override at deploy time by editing this file.
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "8192",
    ]
    # Run vLLM in the foreground so Modal supervises it.
    subprocess.Popen(cmd)


if __name__ == "__main__":
    print(
        f"Configured to serve {MODEL_NAME} on Modal ({GPU}, "
        f"min={MIN_CONTAINERS}, max={MAX_CONTAINERS}).\n"
        f"Run `modal serve scripts/modal_serve.py` for an ephemeral session or "
        f"`modal deploy scripts/modal_serve.py` to persist."
    )
