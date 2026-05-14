# Agentic Hybrid RAG — architecture

The GeneralExpert's optional retrieval path. Brings the expert from
"ChainOfThought over parametric knowledge" to "ReAct over a retrieve(query)
tool backed by hybrid search with cross-encoder rerank."

## The pipeline

1. **Embed the query** with BGE-M3 (dense 1024-d vector).
2. **LanceDB hybrid search** — runs dense ANN and BM25/FTS in parallel,
   fuses with RRF, returns the top 30 candidates.
3. **BGE-Reranker-v2-m3** cross-encoder scores each (query, passage) pair
   jointly, reorders, returns top 5.
4. **DSPy ReAct loop** — the LLM sees the retrieved passages, decides
   whether to refine the query and retrieve again (up to `RAG_MAX_ITERS`),
   or stop and write the final answer with citations.

## Why each piece

- **Agentic, not one-shot**: the 2026 consensus is that one-shot RAG breaks
  on multi-hop and ambiguous queries. The agentic loop costs 3-10× tokens
  but earns it on the queries that actually matter.
- **Hybrid, not pure dense**: dense embeddings silently fail on exact
  identifiers (model names, error codes, rare jargon). BM25 covers that.
  LanceDB's `query_type="hybrid"` runs both with built-in RRF fusion.
- **Cross-encoder rerank**: dense ANN + BM25 give you recall; the
  cross-encoder gives you precision by jointly scoring (query, passage)
  pairs instead of comparing pre-computed vectors.
- **LanceDB**: embedded (no server), native hybrid + multivector, ColBERT
  reranker built in. The ColBERT upgrade path is a method swap, not a
  rewrite.

## Failure modes

The factory (`app/retrieval/factory.py`) catches build errors — missing
extra, missing index, missing model — and returns `None`. GeneralProgram
falls back to plain ChainOfThought instead of failing requests. The log
event `rag.retriever.build_failed` surfaces the underlying cause so an
operator can diagnose without an outage.

## Bring it up

```bash
make rag-up                                          # install + build the index
RAG_ENABLED=1 RAG_INDEX_PATH=rag/index make dev      # run with retrieval enabled
```

The default index is built from `rag/corpus/` — including this file. Replace
the corpus directory with your own to specialize the GeneralExpert for your
domain.
