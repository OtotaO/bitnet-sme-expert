# Specialization strategy: BitNet + DSPy for domain SMEs

## The vision

Take a model whose weights are already at the compression floor — a 1.58-bit
ternary model like BitNet b1.58 — and turn it into a *subject-matter expert*
that beats much larger general-purpose models in a narrow domain. The weights
don't shrink further; everything else (the program that wraps them, the
demonstrations they learn from, the tools they call) does the work.

This document explains how the pieces in this repo combine to make that real,
where the approach works, where it doesn't, and a concrete recommended
deployment.

## The three knobs

There are three places to put optimization effort. They compose; you don't
have to pick one.

### 1. Weight precision — frozen at 1.58-bit

[bitnet.cpp](https://github.com/microsoft/BitNet) is the inference runtime for
1.58-bit ternary models. It ships an OpenAI-compatible HTTP server (the
`llama-server` binary), so DSPy can talk to it through the standard LiteLLM
OpenAI provider — no custom adapter needed.

```bash
./scripts/bitnet_setup.sh        # one-time: clone + build + download weights
./scripts/bitnet_serve.sh        # run the OpenAI-compatible server on :8080
```

Point DSPy at it via env vars (see `app/llm.py`):

```bash
export DSPY_LM_MATH="openai/bitnet-b1.58-2B-4T"
export DSPY_LM_MATH_API_BASE="http://localhost:8080/v1"
export DSPY_LM_MATH_API_KEY="local"   # bitnet.cpp ignores the value
```

You don't have to use BitNet for every role. The `DSPY_LM_<ROLE>` pattern lets
each agent (router, math, code, general) pick its own backend independently —
that's the foundation of the hybrid recommendation below.

### 2. Program structure — DSPy compilation

This is where the biggest wins live when the underlying weights are frozen.
DSPy doesn't change weights; it optimizes the *program* around them:

- **The signature** — what fields go in, what fields come out
- **The instruction string** — the natural-language description of the task
- **The few-shot demonstrations** — labeled examples baked into the prompt

[MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/) bootstraps demos from a
gold dataset and Bayesian-searches over instruction + demo combinations.
[GEPA](https://dspy.ai/api/optimizers/GEPA/overview/) uses a reflection loop:
it inspects failures, proposes new instruction wording, and iterates. GEPA in
particular punches above its weight on small models because the slack that a
big model absorbs in vague wording is exactly what a small model loses.

```bash
make optimize-math              # dspy.MIPROv2 baseline
DSPY_OPTIMIZER=gepa make optimize-math   # GEPA reflection loop
```

Compiled programs land in `compiled/<domain>.json` and are auto-loaded at
startup by `app/llm.py`. Re-running the optimizer rotates them in place — no
deploy needed beyond restarting the process.

### 3. Weight adaptation — fine-tuning

If you have enough labeled domain data, fine-tuning beats prompting. The
service in `app/services/fine_tuning.py` wraps Unsloth + TRL `SFTConfig` for
LoRA / QLoRA SFT on standard transformer base models. This is the right tool
when you want to specialize a Qwen-1.5B or Llama-3.2-1B and accept Q4
quantization on the output.

For a **true BitNet output**, the SFT recipe is different — you want to train
at higher precision and distill back to ternary. The canonical path is the
microsoft/BitNet training recipe or [BitDistiller](https://github.com/DD-DuDa/BitDistiller).
This repo doesn't wire those in directly; the docs link is included so the
upgrade path is obvious if a team wants to take it.

## Where this approach wins (and where it doesn't)

Small models specialize cleanly on tasks where the *output shape* and *domain
vocabulary* are well-bounded. They struggle when the task demands open-ended
reasoning.

| Task shape | Small + DSPy + (optional) SFT |
|---|---|
| Extract N fields from a known document type | Wins routinely after ~1k labeled examples |
| Classify into a fixed taxonomy | Wins; often the first thing people deploy |
| Q&A over a specific corpus with retrieval | Competitive once the retriever is decent |
| Format-following (SQL, regex, JSON) | Wins on schema adherence; large models over-hedge |
| Domain-specific code completion (one framework) | Wins on patterns it has seen; loses on novel idioms |
| Free-form mathematical reasoning | Hard ceiling; offload to a tool (sympy, Wolfram) |
| Novel algorithm design / long-horizon planning | Don't try this with a 2B model yet |
| Multi-hop reasoning over unfamiliar facts | Ceiling — use a frontier model |

The `MathExpert` in this repo is built with this asymmetry in mind. It uses
`dspy.ReAct` with sympy tools and a deterministic sympy fast-path *before* the
LLM gets involved. The LLM picks the strategy and explains the answer; the
math itself is offloaded. That's the pattern that lets a small model match a
big one on math benchmarks — not because the model is better, but because the
program structure does work the model would otherwise have to do.

## Recommended deployment: hybrid by role

Going all-local is rarely the right call. Going all-frontier is expensive and
slow at scale. The realistic deployment uses different backends per role,
matched to where each role spends its tokens:

| Role | Backend | Reason |
|---|---|---|
| `RouterProgram` | small frontier model (e.g. Haiku 4.5) | Low volume per request, latency-sensitive, needs general knowledge to classify edge cases |
| `GeneralExpert` | frontier model (Opus / Sonnet / GPT-4.x) | Open-ended; not a candidate for specialization |
| `MathExpert` | BitNet b1.58 + sympy tools + GEPA-compiled | Bounded vocabulary, tool-augmented, compilation pays off |
| `CodeExpert` | Fine-tuned coder (e.g. Qwen2.5-Coder-1.5B + LoRA) + DSPy demos | Domain vocabulary is narrow per team; SFT on the team's own codebase wins |

Cost / latency math, for an illustrative mix where 60% of traffic hits
MathExpert and CodeExpert (the verticals), 30% hits GeneralExpert, and 10%
hits Router:

```
All-frontier  : 1.00x cost  /  1.00x latency  (baseline)
Hybrid as above: 0.25-0.35x cost  /  0.4-0.6x latency
All-local     : 0.05x cost  /  0.7x latency  /  noticeable quality drop on GeneralExpert
```

These are rough; actual numbers depend on your provider mix, GPU, and traffic
shape. The point is that the hybrid captures most of the cost win without
giving up the quality where it matters.

## How to actually do this

A worked example for the `MathExpert` lane:

1. **Curate a gold dataset.** Start with `tests/eval/datasets/math.jsonl`. Add
   100-500 examples that reflect the domain you care about (calculus,
   probability, finance math, etc.). Keep questions short, answers checkable.

2. **Establish a baseline.** Trigger the eval workflow against `main` with
   whatever LM you plan to ship:

   ```bash
   gh workflow run eval.yml --field domain=math --field lm=openai/gpt-4o-mini
   ```

   Or locally: `RUN_EVAL=1 pytest tests/eval/test_eval.py -k math`. Record
   the score from the JSON artifact (`eval-results.json`) the workflow uploads.

3. **Bring up the BitNet server.** `./scripts/bitnet_setup.sh && ./scripts/bitnet_serve.sh`.
   Re-run the workflow with `--field lm=openai/bitnet-b1.58-2B-4T` — expect a drop.

4. **Compile with GEPA.** `DSPY_OPTIMIZER=gepa make optimize-math`. The
   resulting `compiled/math.json` is auto-loaded next start. Open a PR with
   the compiled artifact and re-run eval with `--field pr_number=<n>` so the
   score lands as a comment on the PR. This is usually where you recover
   most of the gap.

5. **(Optional) Fine-tune the base.** If GEPA leaves you short, train a LoRA
   on the same dataset via `app/services/fine_tuning.py`, then quantize to
   1.58-bit using BitDistiller. Repeat steps 3-4 with the new weights.

6. **Lock the eval in CI.** The `eval.yml` workflow can be wired into branch
   protection (require successful eval before merge) once your gold dataset
   and thresholds are stable. Until then, run it manually on PRs that touch
   programs, signatures, or compiled artifacts.

## What "incompressible" actually means here

BitNet weights are at a hard floor — you can't compress 1.58 bits/weight any
further without changing the architecture. That's the *good* property: the
inference cost is fixed and known, the memory footprint is fixed and known,
and the model fits on hardware where larger models don't. Specialization is
about making *the rest of the stack* better around that fixed floor:

- A better program (DSPy) extracts more capability from the same weights
- A better dataset (SFT or BitDistiller) shifts the weights themselves
- A better tool layer (sympy, retrieval, calculators) offloads what the model
  can't do anyway

The "subject matter expert" framing is accurate as long as the subject is
narrow enough that the small model's ceiling isn't the binding constraint.
Pick the verticals carefully; the ones where this works obviously will be
obvious after the first eval run.

## References

- [BitNet b1.58 paper and model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [microsoft/BitNet repo](https://github.com/microsoft/BitNet)
- [bitnet.cpp OpenAI server (issue #432)](https://github.com/microsoft/BitNet/issues/432)
- [DSPy GEPA optimizer](https://dspy.ai/api/optimizers/GEPA/overview/)
- [DSPy MIPROv2 optimizer](https://dspy.ai/api/optimizers/MIPROv2/)
- [BitDistiller for ternary distillation](https://github.com/DD-DuDa/BitDistiller)
- [Unsloth fine-tuning](https://github.com/unslothai/unsloth)
- [TRL SFTConfig reference](https://huggingface.co/docs/trl/sft_trainer)
