# BitNet and the specialization thesis

BitNet is Microsoft's family of 1.58-bit ternary models. Each weight is one
of {-1, 0, +1}, encoded in roughly 1.58 bits (`log2(3)`). The result is a
model whose inference cost is a fraction of a comparable fp16/bf16 model
while staying competitive on common benchmarks.

## Where this project uses it

bitnet.cpp is the canonical inference runtime. It ships an OpenAI-compatible
HTTP server (`llama-server`), which means DSPy talks to it through the
standard LiteLLM OpenAI provider — no custom adapter. Bring it up with
`./scripts/bitnet_setup.sh` (one-time build + weight download) then
`./scripts/bitnet_serve.sh` (run on :8080).

Point an expert at it via env:

```bash
export DSPY_LM_MATH=openai/bitnet-b1.58-2B-4T
export DSPY_LM_MATH_API_BASE=http://localhost:8080/v1
export DSPY_LM_MATH_API_KEY=local
```

Per-role config means MathExpert can run on BitNet while RouterAgent stays
on a small cheap frontier model. This hybrid posture is the recommendation,
not all-local.

## The three knobs of specialization

1. **Weight precision** — frozen at 1.58-bit for BitNet. You can't compress
   further without changing the architecture.
2. **Program structure (DSPy compilation)** — the lever that produces most
   of the recovered capability when running small models. MIPROv2 +
   demonstration bootstrapping, or GEPA's reflection-based instruction
   evolution.
3. **Weight adaptation (SFT)** — LoRA via Unsloth + TRL on a higher-precision
   base, then quantize back to ternary with BitDistiller. Heavier lift,
   biggest gains.

The three compose. You don't pick one.

## Where the SME framing actually wins

A 2B-parameter specialized model routinely beats frontier general models on
bounded extraction, classification, format-following, and domain Q&A with
retrieval. It does not beat them yet on free-form math, novel code
synthesis, or long-horizon planning — pick the verticals where the
specialization ceiling isn't the binding constraint.

See `docs/specialization.md` for the worked example: dataset → baseline →
swap to BitNet → GEPA compile → optional SFT → lock the eval in CI.
