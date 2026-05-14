"""Cross-encoder rerank stage.

BGE-Reranker-v2-m3 is the 2026 safe default per the BEIR/MTEB landscape: small
(~568M params), multilingual, well-tested, and runs on CPU acceptably for
top-k≈30. Lazy-imported so the rest of the app doesn't pull torch at startup.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .base import Document

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[Document], top_k: int) -> list[Document]: ...


class BGEReranker:
    """Cross-encoder reranker. Constructs the underlying model lazily."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("rag.reranker.load", extra={"model": self.model_name})
            self._model = FlagReranker(self.model_name, use_fp16=True, device=self.device)
        return self._model

    def rerank(self, query: str, docs: Sequence[Document], top_k: int = 5) -> list[Document]:
        if not docs:
            return []
        model = self._ensure_model()
        pairs = [[query, d.text] for d in docs]
        scores = model.compute_score(pairs)  # type: ignore[attr-defined]
        # FlagReranker returns float for a single pair, list[float] for many.
        if isinstance(scores, float):
            scores = [scores]
        scored = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            Document(text=d.text, score=float(s), source=d.source, metadata=d.metadata)
            for d, s in scored[:top_k]
        ]


class IdentityReranker:
    """Passthrough — preserves ANN ordering. Used in tests and as the
    ``RAG_RERANK=0`` opt-out."""

    def rerank(self, query: str, docs: Sequence[Document], top_k: int = 5) -> list[Document]:
        return list(docs)[:top_k]
