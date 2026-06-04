# Eval receipts

Committed proof artifacts from the optimizer. Each `scripts/optimize.py` run
writes `<domain>-<optimizer>.json` here with the **held-out** baseline and
compiled scores, the delta, the LM used, and the split sizes. The current
`math-miprov2.json` (verbatim) — a real **+0.18** held-out win:

```json
{
  "domain": "math",
  "optimizer": "miprov2",
  "auto": "light",
  "lm": "openai/gpt-4o-mini",
  "split": "train->holdout",
  "baseline": 0.62,
  "optimized": 0.8,
  "delta": 0.18000000000000005,
  "n_train": 50,
  "n_holdout": 50
}
```

The honesty bit: the *first* committed receipt for this domain was a **−0.13**
on a 15-item holdout — MIPROv2 looked like it hurt. We kept that negative
receipt as-is while it stood; it turned out to be a small-sample artifact, and a
larger 50/50 split revealed the real +0.18 win. Commit whatever the run
produces — a recorded loss is data, a massaged win is not.

Because the score is measured on `holdout` (data the optimizer never trained
on, see `tests/eval/loader.py`), the `delta` is an honest generalization number
rather than a memorization artifact.

## How to produce one

```bash
# Requires a provider key (OPENAI_API_KEY) or a local bitnet.cpp server.
make optimize-math           # MIPROv2 light
make optimize-math-gepa      # GEPA
```

Then commit the resulting `eval/receipts/math-miprov2.json` alongside the
README receipt table. **Commit whatever the run produces — including a
non-positive delta.** A committed "no improvement yet" is a real result; a
doctored positive one is not.

> First committed receipt: `math-miprov2.json` (2026-06-04, `gpt-4o-mini`) — a
> **−0.13 holdout delta**. MIPROv2 light did not beat the baseline; committed
> as-is per the policy above. See the README "Eval gate & receipts" section.
