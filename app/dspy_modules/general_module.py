"""GeneralProgram — answers general-knowledge questions.

Two modes, selected at construction time from the environment:

* **Plain mode (default)**: ``dspy.ChainOfThought(AnswerGeneralQuestion)``.
  No retrieval. Fast, no extra deps.
* **Agentic hybrid RAG mode** (``RAG_ENABLED=1``): ``dspy.ReAct`` with a
  ``retrieve(query)`` tool backed by :mod:`app.retrieval`. The model decides
  when to retrieve, can re-query on weak signal, and cites passages back to
  the user. Matches the tool-augmented pattern ``MathProgram`` uses with
  sympy and ``CodeProgram`` uses with the sandbox.

The agentic path produces grounded answers with citations; the plain path
relies entirely on the LM's parametric knowledge. Switching between them is
an env-var flip — no code changes.
"""

from __future__ import annotations

import logging
import os

import dspy

from .signatures import AnswerGeneralQuestion

logger = logging.getLogger(__name__)


class GeneralProgram(dspy.Module):
    """Answer general-knowledge questions, optionally with hybrid retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self.retriever = self._maybe_build_retriever()
        if self.retriever is not None:
            self.predict = dspy.ReAct(
                AnswerGeneralQuestion,
                tools=[self._retrieve_tool],
                max_iters=int(os.environ.get("RAG_MAX_ITERS", "3")),
            )
            logger.info("general_module.rag.enabled")
        else:
            self.predict = dspy.ChainOfThought(AnswerGeneralQuestion)

    @staticmethod
    def _maybe_build_retriever() -> object | None:
        try:
            from app.retrieval import build_retriever
        except ImportError:
            return None
        return build_retriever()

    def _retrieve_tool(self, query: str) -> str:
        """Search the knowledge base for passages relevant to ``query``.

        Use this tool when the question requires factual grounding, a specific
        source, or context you don't have parametric knowledge of. Returns the
        top-ranked passages with their sources, ready to cite. If the first
        query returns weak results, refine the query and call again.

        Args:
            query: A short, focused search query.

        Returns:
            A newline-separated list of ``[source] passage`` entries, or
            ``"[no results]"`` if the index turned up nothing.
        """
        if self.retriever is None:
            return "[retrieval unavailable]"
        try:
            docs = list(self.retriever.search(query, k=5))  # type: ignore[union-attr]
        except Exception as exc:
            return f"[retrieval error] {type(exc).__name__}: {exc}"
        if not docs:
            return "[no results]"
        return "\n\n".join(f"[{doc.source or 'unknown'}] {doc.text}" for doc in docs)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)
