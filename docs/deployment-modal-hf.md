# Substrate playbook: HF Inference Providers, Modal, BitNet, and frontier APIs

This project is *substrate-agnostic*. The expert layer (DSPy programs +
optional RAG + optional sandbox) is the constant; the model running underneath
is a config flip. Pick whichever combination fits your envelope.

This document covers the four supported substrates side by side, with
copy-paste recipes for each.

## Decision matrix

| Substrate | Best when | Per-1M-token cost (rough) | Latency | Setup |
|---|---|---|---|---|
| **HF Inference Providers** | You want frontier-grade open weights with zero ops. HF auto-routes to the fastest provider. | ~$0.10-$3 depending on model + provider | ~200ms TTFT | env vars only |
| **Modal vLLM (hosted)** | You want a specific HF base on a specific GPU and pay only when it runs. | ~$0.50-$2 amortized (warm) | 200-500ms TTFT warm, 30-60s cold | one Modal deploy |
| **BitNet (local)** | You want laptop-grade latency, sovereign data, and the cost floor. Capability ceiling is 2B params. | $0 marginal | <100ms TTFT | `make bitnet-setup && make bitnet-serve` |
| **Frontier APIs** | You want maximum capability and don't have time to specialize. | $0.50-$15 | 100-300ms TTFT | env vars only |

Mix and match — per-role config means each expert can sit on a different
substrate. The hybrid sweet spot for most teams:

- `RouterProgram` + `GeneralExpert` on a frontier API (low volume, high knowledge)
- `MathExpert` + `CodeExpert` on a fine-tuned Modal-hosted base or a BitNet
  local model (high volume, narrow scope, where specialization pays)

## Recipe 1 — HF Inference Providers (zero code)

[HF Inference Providers](https://huggingface.co/docs/inference-providers/index)
unifies 15+ partners (Together, Fireworks, Hyperbolic, Replicate, Nebius,
Cerebras, SambaNova, etc.) under a single OpenAI-compatible endpoint at
`https://router.huggingface.co/v1`. One HF token, one bill, automatic routing
to the fastest available provider per model.

Add the model string `huggingface/auto/<org>/<model>` (or pin a provider:
`huggingface/together/meta-llama/Llama-3.3-70B-Instruct`) and you're done.

```bash
export HF_TOKEN=hf_...
# Point a role at Llama-3.3-70B via the auto-routed HF Inference Providers:
export DSPY_LM_GENERAL="huggingface/auto/meta-llama/Llama-3.3-70B-Instruct"
# Or pin :cheapest / :preferred:
export DSPY_LM_CODE="huggingface/auto/Qwen/Qwen2.5-Coder-32B-Instruct:cheapest"
make dev
```

LiteLLM handles the rest — DSPy doesn't need to know it's not talking to OpenAI.

## Recipe 2 — Modal hosted vLLM

When you want full control over the base, GPU, and tensor-parallel layout,
serve it yourself on Modal. `scripts/modal_serve.py` is a single-file
deployment that brings up a vLLM OpenAI-compatible endpoint with a weights
cache that survives cold starts.

```bash
# One-time: install Modal locally and authenticate.
uv pip install modal
modal setup

# One-time: store your HF token as a Modal Secret named "huggingface".
modal secret create huggingface HF_TOKEN=hf_...

# Configure + deploy:
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export GPU="H100"
export API_KEY="$(openssl rand -hex 16)"
modal deploy scripts/modal_serve.py
```

Modal prints a URL. Wire it up:

```bash
export DSPY_LM_MATH="openai/Qwen/Qwen2.5-7B-Instruct"
export DSPY_LM_MATH_API_BASE="https://<workspace>--dspy-sme-vllm-serve.modal.run/v1"
# API_KEY_ENV names the env var holding the key (app/llm.py reads it indirectly):
export DSPY_LM_MATH_API_KEY_ENV="MODAL_VLLM_KEY"
export MODAL_VLLM_KEY="<the API_KEY you set>"
make dev
```

The endpoint is OpenAI-compatible so it Just Works through LiteLLM. Set
`MIN_CONTAINERS=1` if cold starts are unacceptable; otherwise it scales to
zero between requests and you pay nothing while idle.

## Recipe 3 — Modal-hosted fine-tuning, then HF Hub

`scripts/modal_finetune.py` runs LoRA SFT with Unsloth + TRL on a Modal
H100, then pushes the adapter to a HF repo. Same `SFTConfig` shape as
`app/services/fine_tuning.py` so the local-workstation and hosted-Modal paths
share knobs.

```bash
# Set up Modal secret as in Recipe 2, then:
modal run scripts/modal_finetune.py::train \
    --base-model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-repo your-org/your-domain-sft \
    --output-repo your-org/llama-3.3-70b-domain-lora
```

The adapter lands at `https://huggingface.co/your-org/llama-3.3-70b-domain-lora`.
To deploy the fine-tuned model:

```bash
# Re-serve the merged model with Recipe 2, pointed at the fine-tune output:
export MODEL_NAME="your-org/llama-3.3-70b-domain-lora"
modal deploy scripts/modal_serve.py
```

For 70B class models, you typically want `GPU=H100:2` (tensor-parallel) or
`GPU=A100-80GB:2`.

## Recipe 4 — BitNet local (small + sovereign)

The original local-first path. Real cost floor: $0 marginal per token, no
egress, runs on a laptop. The capability ceiling is the 2B-parameter
envelope.

```bash
make bitnet-setup       # one-time: build bitnet.cpp, download weights
make bitnet-serve       # llama-server on :8080
export DSPY_LM_MATH="openai/bitnet"
export DSPY_LM_MATH_API_BASE="http://localhost:8080/v1"
export DSPY_LM_MATH_API_KEY_ENV="BITNET_DUMMY_KEY"   # names the var below
export BITNET_DUMMY_KEY="local"                      # bitnet.cpp ignores the value
make dev
```

See `docs/specialization.md` for when BitNet's ceiling stops being the
binding constraint.

## Mixing substrates — a worked example

A realistic hybrid for a team with HF + Modal credits:

```bash
# Router: small, cheap, fast.
export DSPY_LM_ROUTER="huggingface/auto/meta-llama/Llama-3.2-3B-Instruct:fastest"

# Math: BitNet local with sympy fast-path. Bounded vocabulary; specialization wins.
export DSPY_LM_MATH="openai/bitnet"
export DSPY_LM_MATH_API_BASE="http://localhost:8080/v1"
export DSPY_LM_MATH_API_KEY_ENV="BITNET_DUMMY_KEY"
export BITNET_DUMMY_KEY="local"

# Code: fine-tuned Qwen2.5-Coder on Modal. Narrow domain vocabulary; SFT wins.
export DSPY_LM_CODE="openai/your-org/qwen2.5-coder-32b-domain"
export DSPY_LM_CODE_API_BASE="https://<workspace>--dspy-sme-vllm-serve.modal.run/v1"
export DSPY_LM_CODE_API_KEY_ENV="MODAL_VLLM_KEY"
export MODAL_VLLM_KEY="$YOUR_MODAL_KEY"

# General: frontier model. Open-ended; not a candidate for specialization.
export DSPY_LM_GENERAL="anthropic/claude-sonnet-4-6"

make dev
```

Each expert costs and behaves according to its substrate. Switch any one of
them by flipping the env var — no code changes, no redeploys beyond pointing
to a different URL.

## Specialization is what stays constant

The point of the project is the specialization layer (DSPy programs +
optional ReAct tools + optional RAG + optional cache compilation), not any
particular runtime. BitNet remains supported and documented because the
laptop-grade-latency story is real, but it's not the only path — and for
teams with cloud credits, *probably not the recommended one for a v1
deployment*. The "incompressible model" framing earns its place when you've
already exhausted the easier wins.

## References

- [Modal docs — vLLM OpenAI-compatible inference](https://modal.com/docs/examples/vllm_inference)
- [Modal docs — efficient LLM fine-tuning with Unsloth](https://modal.com/docs/examples/unsloth_finetune)
- [HF Inference Providers — OpenAI-compatible API](https://huggingface.co/changelog/inference-providers-openai-compatible)
- [HF Inference Providers — getting started](https://huggingface.co/inference/get-started)
- [LiteLLM HF provider](https://docs.litellm.ai/docs/providers/huggingface)
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [BitNet b1.58 model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
