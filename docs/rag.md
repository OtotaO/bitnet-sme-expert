# Agentic hybrid RAG for the GeneralExpert

Brings the GeneralExpert from "ChainOfThought over parametric knowledge" up
to the 2026 production floor: **agentic hybrid retrieval with cross-encoder
rerank**, wrapped in `dspy.ReAct` so the model decides when (and how) to
retrieve.

## Stack at a glance

```
question
   │
   ▼
dspy.ReAct(GeneralProgram)
   │   tools: [retrieve(query)]
   ▼
LanceDBRetriever
   │   hybrid: dense ANN + BM25/FTS  (RRF fusion, built-in)
   ▼
top-30 candidates
   │
   ▼
BGE-Reranker-v2-m3 (cross-encoder)
   │
   ▼
top-5 → cited in the answer
```

Models, all opt-in via the `[rag]` extra:

* **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** — embedder. Single model
  produces dense + sparse + multi-vector representations; we use the dense
  output for ANN and rely on LanceDB's built-in FTS for the sparse side.
  Upgrade path to full ColBERT multi-vector is a swap of two methods.
* **[BGE-Reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)** —
  cross-encoder. 2026 safe default; ~568M params, multilingual, runs on CPU
  for top-k≈30.
* **[LanceDB](https://lancedb.com)** — embedded vector + FTS store. See
  the LanceDB vs DuckDB section below for the comparison.

## Quickstart — one command for a working system

The repo ships with a starter corpus at `rag/corpus/` so a fresh checkout
gets you a queryable GeneralExpert in one Make target:

```bash
make rag-up
RAG_ENABLED=1 RAG_INDEX_PATH=rag/index make dev
```

`make rag-up` is `make rag-install` + `make rag-index`. It installs the
`[rag]` extra (lancedb + FlagEmbedding + torch + sentence-transformers) and
builds the LanceDB index from the bundled corpus. Once running, the
GeneralExpert can answer questions about the project itself — DSPy basics,
the specialization thesis, BitNet, env knobs, the RAG architecture — with
citations back to the source markdown files.

To point at your own corpus instead, replace the files under `rag/corpus/`
or pass `--source-dir`:

```bash
uv run python scripts/build_rag_index.py --source-dir my-docs --index-path my-index
RAG_ENABLED=1 RAG_INDEX_PATH=my-index make dev
```

When RAG is enabled, every `GeneralExpert` response includes the cited
sources via the canonical `ExpertOutput.sources` field — no special-case
handling required downstream.

## Why agentic, not one-shot

The 2026 consensus (multiple production write-ups, [Towards Data Science
analysis](https://towardsdatascience.com/agentic-rag-vs-classic-rag-from-a-pipeline-to-a-control-loop/),
[2026 production guide](https://www.marsdevs.com/guides/agentic-rag-2026-guide)):
one-shot RAG is fine for "doc lookup" questions but breaks on multi-hop,
ambiguous, or low-signal queries. The agentic loop costs 3-10× tokens and
2-5× latency but earns it on the queries that actually matter — the model can
re-query, evaluate result quality, and stop when it has enough.

The DSPy ReAct + MIPROv2 path on `gpt-4o-mini` lifted an agentic retrieval
benchmark from **24% → 51%** in the [official DSPy
tutorial](https://dspy.ai/tutorials/rag/). That's exactly the lever this
module exposes for the GeneralExpert.

## Why hybrid, not pure dense

Pure dense retrieval silently fails on exact identifiers (model names, error
codes, rare jargon). BM25 covers that case. LanceDB's `query_type="hybrid"`
runs both in parallel, fuses via RRF, and the cross-encoder rerank cleans up
the precision. Three-way retrieval (dense + sparse + late-interaction) is
the [IBM-research-recommended setup](https://infiniflow.org/blog/best-hybrid-search-solution);
we ship dense + sparse here, with the late-interaction door open.

## LanceDB vs DuckDB

The two embedded options that came up in research; here's the side-by-side
that drove the call to use LanceDB:

| Property | LanceDB | DuckDB-VSS |
|---|---|---|
| Vector ANN | HNSW + IVF_PQ + scalar quantization | HNSW (via usearch) |
| Full-text search | Built-in FTS (`create_fts_index`) | `fts` extension |
| Hybrid (vector + FTS) | **Native `query_type="hybrid"` w/ RRF** | Hand-rolled SQL join |
| Multivector / ColBERT | **First-class** (`list_(list_(float32))`) | Not first-class |
| ColBERT reranker | **Built-in** (`lancedb.rerankers.ColbertReranker`) | DIY |
| Stability | Stable | VSS extension still flagged **experimental** |
| Same-process | Yes | Yes |
| SQL analytics on the same data | Via `Lance × DuckDB` integration | Native |

DuckDB-VSS is closing the gap fast, and the [Lance × DuckDB
integration](https://www.lancedb.com/blog/lance-x-duckdb-sql-retrieval-on-the-multimodal-lakehouse-format)
means picking LanceDB now doesn't preclude SQL analytics later — we just use
DuckDB on top of the Lance files when we need it. The decisive factors
today were native hybrid + native multivector + the ColBERT upgrade path.

## Env knobs

| Env var | Default | Purpose |
|---|---|---|
| `RAG_ENABLED` | `0` | `1`/`true`/`yes` to enable. Otherwise GeneralProgram stays ChainOfThought. |
| `RAG_INDEX_PATH` | `rag/index` | Where the LanceDB store lives. |
| `RAG_TABLE_NAME` | `documents` | Table inside the store. |
| `RAG_RERANK` | `1` | Set `0` to skip the cross-encoder stage. |
| `RAG_PREFETCH_K` | `30` | Candidates pulled from the hybrid stage before rerank. |
| `RAG_MAX_ITERS` | `3` | ReAct loop budget. Higher → more retrieval rounds per question. |
| `RAG_EMBEDDER_MODEL` | `BAAI/bge-m3` | HF id for the dense embedder. |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HF id for the cross-encoder. |

## Failure modes

The factory in `app/retrieval/factory.py` catches build errors (missing
extra, missing index, missing model) and returns `None`. `GeneralProgram`
then falls back to plain `ChainOfThought` instead of failing the request.
Logs surface `rag.retriever.build_failed` so the operator can fix the
underlying issue without an outage.

## Re-indexing

The ingest script writes a fresh table every time (`drop_table` → `create_table`)
so re-runs are idempotent. `make rag-clean` drops the local index; the next
`make rag-index` rebuilds it from `rag/corpus/`. For an incremental update
strategy (add new rows without rebuilding), see [LanceDB's `table.add()` docs](https://lancedb.com/docs/concepts/data-management/);
left as a follow-up since the ingest is fast enough at hundreds-of-thousands
of chunks to just rebuild.

## The starter corpus

`rag/corpus/` ships with five markdown files chosen to make the system
*useful immediately* without any external content:

- `01-project-overview.md` — what dspy-sme-expert is, the three experts.
- `02-dspy-primer.md` — DSPy concepts used in the codebase.
- `03-bitnet-and-specialization.md` — BitNet + the three-knobs thesis.
- `04-rag-architecture.md` — this pipeline.
- `05-env-knobs.md` — every env var the service reads.

These are real, accurate, project-specific docs — not Lorem Ipsum. After
`make rag-up`, you can ask the GeneralExpert questions like *"What does
`CODE_SANDBOX_ENABLED` do?"* or *"Why hybrid retrieval over pure dense?"*
and get cited answers grounded in this corpus.

## References

- [DSPy RAG tutorial — 24→51% lift on `gpt-4o-mini`](https://dspy.ai/tutorials/rag/)
- [Agentic RAG vs Classic RAG (Towards Data Science)](https://towardsdatascience.com/agentic-rag-vs-classic-rag-from-a-pipeline-to-a-control-loop/)
- [Agentic RAG: The 2026 Production Guide](https://www.marsdevs.com/guides/agentic-rag-2026-guide)
- [Hybrid Search in Production: Why BM25 Still Wins (2026)](https://tianpan.co/blog/2026-04-12-hybrid-search-production-bm25-dense-embeddings)
- [LanceDB hybrid search blog post](https://www.lancedb.com/blog/hybrid-search-rag-for-real-life-production-grade-applications-e1e727b3965a)
- [LanceDB multivector docs](https://docs.lancedb.com/search/multivector-search)
- [Lance × DuckDB integration](https://www.lancedb.com/blog/lance-x-duckdb-sql-retrieval-on-the-multimodal-lakehouse-format)
- [duckdb-vss extension repo](https://github.com/duckdb/duckdb-vss)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [BGE-Reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Embedding Model Leaderboard: MTEB March 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)
