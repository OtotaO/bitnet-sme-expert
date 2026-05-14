"""Retrieval primitives for the GeneralExpert agentic RAG path.

The package is gated behind the ``[rag]`` extra; importing this module without
``lancedb`` installed will raise ``ImportError`` only on instantiation, not at
import time, so the main DSPy service still starts on a minimal install.

Public API:

* :class:`Document` — what retrievers return.
* :class:`Retriever` — the protocol every backend implements.
* :class:`LanceDBRetriever` — hybrid (dense + sparse) retriever with optional
  cross-encoder rerank.
* :func:`build_retriever` — env-driven factory used by ``GeneralProgram``.
"""

from __future__ import annotations

from .base import Document, Retriever
from .factory import build_retriever

__all__ = ["Document", "Retriever", "build_retriever"]
