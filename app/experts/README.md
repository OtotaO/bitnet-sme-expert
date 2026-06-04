# Experts

Thin wrappers that adapt a `dspy.Module` (the actual model code, in
`app/dspy_modules/`) to the service's API contract. An expert holds a program,
runs it under a domain-specific LM via `dspy.context(...)`, optionally loads a
compiled/optimized version from `compiled/<domain>.json`, and converts the
`dspy.Prediction` into the canonical `ExpertOutput` envelope.

## Bundled experts

| Expert | File | Domain | LM role | Program (`app/dspy_modules/`) |
| --- | --- | --- | --- | --- |
| `MathExpert` | `math_expert.py` | `MATH` | `math` | `MathProgram` — `dspy.ReAct` over sympy tools + deterministic fast-path |
| `CodeExpert` | `code_expert.py` | `CODE` | `code` | `CodeProgram` — `dspy.ChainOfThought` (opt-in sandboxed execution) |
| `GeneralExpert` | `general_expert.py` | `GENERAL` | `general` | `GeneralProgram` — `dspy.ChainOfThought` (opt-in hybrid RAG) |

## Base class

`base_expert.py` defines **`DSPyExpert`** (subclass of `app/models/expert.py`'s
abstract `BaseExpert`). It owns: program construction, compiled-artifact loading
(with fall-back to the uncompiled program on load failure), per-call LM
selection, timing/metadata stamping, and `ExpertOutput` validation. Subclasses
fill in a small contract.

## Add a new domain expert (end to end)

1. **Add the domain** to `ExpertDomain` in `app/schemas/base.py`.
2. **Write the signature + program** in `app/dspy_modules/` — a `dspy.Signature`
   describing inputs/outputs and a `dspy.Module` (e.g. `dspy.ChainOfThought` or
   `dspy.ReAct`). Export it from `app/dspy_modules/__init__.py`.
3. **Create the expert** here, subclassing `DSPyExpert`:

   ```python
   from ..dspy_modules import MyProgram
   from ..schemas.base import ExpertDomain
   from .base_expert import DSPyExpert

   class MyExpert(DSPyExpert):
       DOMAIN = ExpertDomain.MY_DOMAIN
       LM_ROLE = "my_domain"          # used by app/llm.py:get_lm()

       def _build_program(self) -> dspy.Module:
           return MyProgram()

       # Optional overrides:
       # _format_prediction(self, prediction) -> {"response", "metadata", ...}
       # _invoke(self, input_text, context, **kwargs) -> dspy.Prediction
       #   (default calls self.program(question=input_text))
   ```

4. **Add an LM spec** for the new role in `app/llm.py` (`_DEFAULT_SPECS`), or rely
   on a `DSPY_LM_<ROLE>` env override.
5. **Register it** in `app/main.py:_register_experts(...)` via
   `service.register_expert_class(domain=..., expert_class=..., config=...)`.
6. **Add a gold split + metric** for the eval gate: a
   `tests/eval/datasets/<domain>.train.jsonl` / `.holdout.jsonl` pair, the input
   fields in `tests/eval/loader.py:DOMAIN_INPUTS`, a metric in
   `tests/eval/test_eval.py`, and a threshold in
   `scripts/run_eval.py:DEFAULT_THRESHOLDS`.

## Usage

Experts are normally reached through the API (`POST /api/v1/query`) or the
`ExpertService`. Direct use:

```python
expert = MathExpert()
result = await expert.generate("What is the derivative of x^2?", context={})
print(result["response"])   # ExpertOutput-shaped dict
```

See `docs/specialization.md` for the optimization loop and `eval/receipts/` for
how compiled-program wins (or non-wins) are recorded.
