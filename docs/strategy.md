# Strategy, goals & outcomes

> A living document. It states what this project is for, why it should exist,
> how we'll know it's working, and what is explicitly out of scope. Last
> reviewed 2026-06-04.

## North star

**The reference open-source pattern for a self-optimizing, multi-domain SME
service.** One `dspy.Module` per domain, auto-compiled against a committed gold
set, regression-gated in CI behind a single typed FastAPI contract, with an
optional 1-bit BitNet local provider for offline / PII / air-gapped workloads.

The thesis (see [`specialization.md`](./specialization.md)): you don't need a
bigger model to win a narrow domain. Freeze the weights at the compression floor
and put the optimization effort into the *program* around them — the structure
(`dspy.Module`), the demonstrations (MIPROv2 / GEPA), and the tools they call.

## Why this should exist

Most "AI expert" repos are demos: a prompt, a wrapper, no way to tell whether a
change made things better or worse. This project's bet is that the durable value
isn't the model or the framework — it's the **measurement discipline** around
them. That is also the moat, and it is deliberately the hard part to copy.

## Differentiation (the moat)

1. **Honesty engineering.** A committed train/holdout split (not a runtime
   shuffle), a CI gate scored on held-out data, and committed receipts under
   `eval/receipts/` — *including non-positive deltas*. The first committed
   optimizer receipt is a −0.13 result, kept on purpose. A doctored win is worth
   less than a recorded loss.
2. **DSPy self-optimization as a first-class loop**, not a one-off prompt: a
   metric, a gold set, MIPROv2/GEPA compilation, a persisted compiled artifact
   loaded at startup.
3. **One typed contract across providers.** The same `ExpertOutput` envelope
   whether a request is served by GPT, Claude, Gemini, or a local BitNet server.
4. **Optional offline BitNet.** A 1-bit local fallback for PII-sensitive /
   air-gapped use — a *provider*, not the centerpiece.

The moat is **corpus + held-out eval + receipts**, not "we run BitNet." Effort
should compound that, not chase novelty.

## Strategy

The engineering substrate is largely built and sound (real DSPy modules, clean
per-role LM routing, a working eval gate, a properly-sandboxed code path). The
leverage now is **credibility and distribution, not more features**:

- Make the eval signal *trustworthy* (large enough holdouts, deterministic
  runs, per-item visibility) so receipts mean something.
- Keep the docs *honest* (claims match code) so the "v2 was vapor, v3 is real"
  story survives scrutiny.
- Lower the barrier to *adding a domain* so the pattern is reusable, which is
  the whole point of a "reference" repo.

## Goals (next, concrete and measurable)

Ordered by leverage. Each is a checkable outcome, not a vibe.

1. **Trustworthy eval.** *(Mostly done.)* Holdouts are now 50 items/domain (150
   total); eval is pinned to `temperature=0`; a 95% Wilson interval is reported
   per domain. An **execution-based code metric** (`tests/eval/exec.py`: runs the
   generated code against committed assertions in a timeout-bounded, scrubbed
   subprocess) plus a seed `code_exec.jsonl` set is the path off the saturating
   substring metric. **Remaining:** expand that set and flip the code gate onto
   it (needs an opt-in CI job); gate on the Wilson *lower* bound rather than the
   point estimate (needs larger N); add an LLM-judge for `general`; fixed seed.
2. **A real optimizer win, recorded.** *(Done — first instance.)*
   `eval/receipts/math-miprov2.json` records MIPROv2-light lifting math from 0.62
   to 0.80 (**+0.18**) on the 50-item holdout at temp 0. **Remaining:** confirm
   stability across ≥3 seeds and extend to other domains once their metrics have
   headroom (code/general still saturate).
3. **Close the feedback loop.** Persist `/feedback` with a `query_id`
   correlation so the signal can feed future optimization. *Done when:*
   feedback is stored and queryable, not just logged.
4. **Cost-safe by construction.** Rate-limit `/query` and `/collaborate`
   (currently only health/ops endpoints are limited); add a per-run eval cost
   ceiling. *Done when:* the expensive endpoints enforce `RATE_LIMIT` and the
   eval workflow can't run unbounded spend.
5. **Measured BitNet receipt.** Replace the third-party tok/s estimate with a
   `make bitnet-demo` measurement committed under `eval/receipts/`.
6. **Documented extension path.** A reader can add a new domain expert from the
   docs alone (see `app/experts/README.md`).

## Definition of done — "credible reference repo"

- CI fully green, including an eval gate that *actually gates* (it does now,
  after the score-scale fix) and is statistically meaningful (goal 1).
- Every concrete README/doc claim is backed by code (the optional-feature
  surface — fallback chains, MLflow optimizer tracking, fine-tuning — is either
  implemented or labelled as designed-for, not asserted as present).
- A documented, repeatable add-a-domain workflow.
- At least one honest, reproducible optimizer win on record.
- A measured local-inference (BitNet) throughput receipt.

## Non-goals / scope guardrails

- **Not a model-training project.** The fine-tuning surface is experimental;
  the centerpiece is program optimization over frozen weights.
- **Not a general agent framework.** It's a narrow-domain SME service with a
  fixed, typed contract.
- **BitNet is a provider, not the pitch.** Don't let "we run a 1-bit model"
  crowd out the eval/receipts story.
- **No feature should outrun its eval.** New capability lands with a gold split
  and a metric, or it doesn't land.

## Honest current state (2026-06-04)

Built and working: real per-domain `dspy.Module`s; per-role LM routing with env
overrides; a pre-merge eval gate scored on a committed holdout; committed
receipts (incl. a negative one); an opt-in, default-deny code-execution sandbox
(Deno/Pyodide); production fail-closed config validation; green CI.

Recently hardened: holdouts grown to 50/domain (150 total) and eval pinned to
`temperature=0`; `/query` and `/collaborate` are now rate-limited; the math
fast-path is exact and its sympy tools are bounded against DoS; ~690 lines of
dead code removed; authorization moved to per-route dependencies; feedback is
persisted; optimizer runs log to MLflow. The experts are now exposed over **MCP**
(`app/mcp_server.py`) — aligning with the 2026 de-facto agent-interface standard
the research flagged — and an execution-based code-eval harness lays the
foundation for behavioral metrics.

Known limitations, tracked as the goals above and in the audit follow-ups issue:
code/general eval metrics are lenient substring proxies (saturate at 1.00) — the
Wilson 95% interval is now reported per domain, but the gate is still on the
point estimate (lower-bound gating needs larger N); `LMSpec.fallbacks` is
reserved rather than wired. Feedback is now persisted, auth is per-route, and
optimizer runs log to MLflow when configured — those earlier gaps are closed.
The remaining items are roadmap, not secrets; the docs say so where each appears.
