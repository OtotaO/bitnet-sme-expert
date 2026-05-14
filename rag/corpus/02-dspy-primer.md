# DSPy primer for dspy-sme-expert

DSPy is a framework for programming — not prompting — language models. You
declare typed signatures, compose them into modules, and let optimizers
(MIPROv2, GEPA) discover the best instruction wording and few-shot demos
against a metric.

## Core concepts used in this repo

### Signature
A typed contract: what fields go in, what fields come out, plus a docstring
DSPy treats as the high-level instruction. See `app/dspy_modules/signatures.py`
for `RouteQuestion`, `SolveMathProblem`, `GenerateCode`, `AnswerGeneralQuestion`.

### Module
A composable program built around one or more predictors. The default
predictor is `dspy.Predict`; `dspy.ChainOfThought` adds a reasoning step,
`dspy.ReAct` adds a tool-using loop. Each domain expert in `app/dspy_modules/`
is a subclass of `dspy.Module`.

### Optimizer
Tunes the prompt scaffold (instruction wording + few-shot demonstrations)
against a metric over a gold dataset. The optimized program is serialized to
`compiled/<domain>.json` and loaded automatically by `app/experts/base_expert.py`
on startup. Optimization is deliberately separate from inference — no API
calls happen during regular request handling beyond the LM you've configured.

### Tool (in ReAct)
A plain Python callable with a docstring. DSPy introspects the docstring +
type hints to teach the model when and how to call the tool. The MathExpert
exposes sympy tools (simplify, solve, integrate, differentiate, series).
The CodeExpert exposes a sandboxed Python interpreter when sandbox mode is
enabled. The GeneralExpert exposes a `retrieve(query)` tool when RAG is
enabled.

## Why this design

Two reasons. First, optimizer-driven prompt search makes small models
competitive — the DSPy ReAct + MIPROv2 path lifted a benchmark from 24% to
51% on gpt-4o-mini in the official tutorial, which is exactly the kind of
specialization story this project is built around. Second, the signature
boundary makes refactoring fearless: change the model, change the prompt
adapter, change the optimizer — none of that touches caller code.

## What we don't use yet

- `dspy.GRPO` — online RL alignment. On the roadmap, not in scope today.
- Multi-turn streaming — `dspy.streamify` exists; not wired in.
- Native MLflow tracing integration is wired (opt-in via `MLFLOW_TRACKING_URI`).
