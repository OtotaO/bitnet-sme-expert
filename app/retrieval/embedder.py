"""Embedding callable for BGE-M3.

Lazy-imports ``FlagEmbedding`` so the rest of the app starts cleanly without
the ``[rag]`` extra installed. Returns dense float vectors; BGE-M3 also
produces sparse + multi-vector outputs, but for v1 we let LanceDB's built-in
FTS handle sparse and stick to dense vectors for the ANN side. Upgrading to
the full multi-vector ColBERT path is a swap of two methods, not a rewrite.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024  # BGE-M3 dense output dimension.


@runtime_checkable
class Embedder(Protocol):
    """Embeds a batch of strings to ``list[list[float]]``. Stateless from
    the caller's perspective; backends may cache the loaded model internally."""

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...


class BGEM3Embedder:
    """BGE-M3 dense embedder. Heavy; load once and reuse."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: object | None = None
        self._dim = DEFAULT_DIM

    def _ensure_model(self) -> object:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("rag.embedder.load", extra={"model": self.model_name})
            self._model = BGEM3FlagModel(self.model_name, use_fp16=True, device=self.device)
        return self._model

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        out = model.encode(
            list(texts), return_dense=True, return_sparse=False, return_colbert_vecs=False
        )  # type: ignore[attr-defined]
        return [list(map(float, v)) for v in out["dense_vecs"]]

    @property
    def dim(self) -> int:
        return self._dim


def deterministic_stub_embedder(dim: int = 8) -> Callable[[Sequence[str]], list[list[float]]]:
    """A hashing-based stub embedder for tests — no model download required."""
    import hashlib

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(b - 128) / 128.0 for b in digest[:dim]]
            out.append(vec)
        return out

    _embed.dim = dim  # type: ignore[attr-defined]
    return _embed
