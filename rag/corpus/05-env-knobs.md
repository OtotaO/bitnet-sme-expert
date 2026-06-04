# Environment variables — dspy-sme-expert

Every behavior change in this project is an env-var flip. No config files,
no YAML, no profiles. The full set is documented inline in `.env.example`;
this is the cheat sheet.

## Core service

| Var | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | One of `development`/`testing`/`staging`/`production`. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Where uvicorn binds. |
| `LOG_LEVEL` | `INFO` | Standard Python levels. |
| `LOG_FORMAT` | `json` | `json` or `text`. Structured logging is JSON by default. |

## Per-role LM selection

Per-role overrides choose the LiteLLM model string for each expert:

```bash
DSPY_LM_ROUTER=openai/gpt-4o-mini
DSPY_LM_MATH=anthropic/claude-haiku-4-5-20251001
DSPY_LM_CODE=anthropic/claude-sonnet-4-6
DSPY_LM_GENERAL=openai/gpt-4o-mini
```

For local BitNet via bitnet.cpp on :8080:

```bash
DSPY_LM_MATH=openai/bitnet
DSPY_LM_MATH_API_BASE=http://localhost:8080/v1
DSPY_LM_MATH_API_KEY_ENV=BITNET_DUMMY_KEY
BITNET_DUMMY_KEY=local          # bitnet.cpp ignores the value
```

## CodeExpert sandbox

| Var | Default | Notes |
|---|---|---|
| `CODE_SANDBOX_ENABLED` | `0` | `1` flips CodeExpert from ChainOfThought to ReAct + Pyodide sandbox. Requires Deno on PATH. |
| `CODE_SANDBOX_MAX_ITERS` | `3` | ReAct loop budget. |

## GeneralExpert RAG

| Var | Default | Notes |
|---|---|---|
| `RAG_ENABLED` | `0` | `1` flips GeneralExpert from ChainOfThought to ReAct + retrieve tool. Requires `[rag]` extra and a built index. |
| `RAG_INDEX_PATH` | `rag/index` | Where the LanceDB store lives. Build with `make rag-index`. |
| `RAG_TABLE_NAME` | `documents` | Table inside the store. |
| `RAG_RERANK` | `1` | `0` to skip the cross-encoder rerank stage. |
| `RAG_PREFETCH_K` | `30` | Candidates pulled before rerank. |
| `RAG_MAX_ITERS` | `3` | ReAct loop budget. |
| `RAG_EMBEDDER_MODEL` | `BAAI/bge-m3` | HF id for the dense embedder. |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HF id for the cross-encoder. |

## Observability

| Var | Default | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | _(unset)_ | Setting this enables DSPy's MLflow autolog. Tracks every program call and optimizer run. |
| `MLFLOW_EXPERIMENT_NAME` | `dspy-sme-expert` | Experiment to log under. |
