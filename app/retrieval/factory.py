"""Env-driven retriever factory.

Reads:

* ``RAG_ENABLED`` — ``1``/``true``/``yes`` to enable. Anything else returns
  ``None`` and the caller falls back to the no-retrieval path.
* ``RAG_INDEX_PATH`` — directory of the LanceDB store (default
  ``rag/index``). Must exist; the ingest script creates it.
* ``RAG_TABLE_NAME`` — table inside the store (default ``documents``).
* ``RAG_RERANK`` — ``0`` to skip the cross-encoder rerank stage.
* ``RAG_EMBEDDER_MODEL`` — HF id for the embedder (default ``BAAI/bge-m3``).
* ``RAG_RERANKER_MODEL`` — HF id for the reranker (default
  ``BAAI/bge-reranker-v2-m3``).
* ``RAG_PREFETCH_K`` — candidates pulled before rerank (default 30).

Kept env-driven so the same code path serves dev, CI, and production without
config files.
"""

from __future__ import annotations

import logging
import os

from .base import Retriever
from .embedder import BGEM3Embedder
from .reranker import BGEReranker, IdentityReranker

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = "rag/index"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def build_retriever() -> Retriever | None:
    """Construct the retriever from environment, or return ``None`` if disabled.

    Errors during construction (missing index, missing extra) are logged and
    ``None`` is returned — the GeneralExpert then degrades to plain
    ChainOfThought instead of failing the request.
    """
    if not _flag("RAG_ENABLED"):
        return None
    try:
        from .lancedb_store import LanceDBRetriever

        embedder = BGEM3Embedder(
            model_name=os.environ.get("RAG_EMBEDDER_MODEL", "BAAI/bge-m3"),
        )
        reranker = (
            BGEReranker(model_name=os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
            if _flag("RAG_RERANK", "1")
            else IdentityReranker()
        )
        retriever = LanceDBRetriever(
            db_path=os.environ.get("RAG_INDEX_PATH", DEFAULT_INDEX_PATH),
            embedder=embedder,
            reranker=reranker,
            table_name=os.environ.get("RAG_TABLE_NAME", "documents"),
            prefetch_k=int(os.environ.get("RAG_PREFETCH_K", "30")),
        )
        logger.info(
            "rag.retriever.built",
            extra={
                "index": os.environ.get("RAG_INDEX_PATH", DEFAULT_INDEX_PATH),
                "rerank": _flag("RAG_RERANK", "1"),
            },
        )
        return retriever
    except Exception:
        logger.exception("rag.retriever.build_failed")
        return None
