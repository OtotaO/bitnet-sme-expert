# dspy-sme-expert — project overview

dspy-sme-expert is a FastAPI service that exposes three domain "subject
matter experts" (math, code, general) built as DSPy 3.x programs. A router
dispatches incoming questions to the right expert, and each expert wraps a
specialized `dspy.Module` — `dspy.ReAct` for the agentic ones, `dspy.ChainOfThought`
for the simpler paths.

## What's an "SME" here

A subject-matter expert is a small, specialized LLM program that beats
general-purpose models on a narrow domain. The thesis: weights frozen at
1.58-bit (BitNet) plus prompt-level optimization (DSPy MIPROv2 / GEPA) plus
optional fine-tuning (Unsloth + TRL) is enough to make a 2B-parameter model
competitive with frontier models *inside its lane*. See
`docs/specialization.md` for the full game plan.

## The three experts

- **MathExpert** — `dspy.ReAct` over sympy tools, with a deterministic
  arithmetic fast-path that answers `2 + 2` without ever calling the LLM.
- **CodeExpert** — `dspy.ChainOfThought` by default; switches to `dspy.ReAct`
  with a Pyodide sandbox when `CODE_SANDBOX_ENABLED=1`, so the model can
  write candidate code, execute it, and iterate.
- **GeneralExpert** — `dspy.ChainOfThought` by default; switches to
  `dspy.ReAct` with a hybrid-RAG retrieve tool (LanceDB + BGE-M3 + BGE-Reranker)
  when `RAG_ENABLED=1`.

## Hybrid deployment is the recommendation

Don't go all-local. Router + GeneralExpert on a frontier API; MathExpert
and CodeExpert on a local BitNet + DSPy-compiled stack. The per-role LM
selection (`DSPY_LM_<ROLE>` env vars) is what makes this trivial — point each
expert at whatever model is right for that domain.

## What this isn't

- It's not a multi-tenant SaaS platform. Single-instance FastAPI.
- It's not a model trainer in the loop; SFT is offered via the `[finetune]`
  extra but kicks off externally.
- It's not a RAG-first system. Retrieval is opt-in and only meaningful for
  the GeneralExpert path.
