"""Retriever protocol + shared types.

A retriever takes a natural-language query and returns ranked
:class:`Document` snippets. Backends differ in storage (LanceDB, in-memory,
etc.) and in the ranking strategy (dense, hybrid, ColBERT-late-interaction),
but the surface stays uniform so ``GeneralProgram`` doesn't care.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    """A retrieved passage plus its provenance."""

    text: str
    score: float = 0.0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> dict[str, Any]:
        """Render to the ``sources`` shape ``ExpertOutput`` expects."""
        return {
            "source": self.source or "unknown",
            "score": round(self.score, 4),
            "snippet": self.text[:240],
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@runtime_checkable
class Retriever(Protocol):
    """Minimal retrieval interface. Implementations may be sync; ``GeneralProgram``
    calls ``search`` directly inside a ``dspy.ReAct`` tool."""

    def search(self, query: str, k: int = 5) -> Sequence[Document]: ...
