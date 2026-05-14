"""LanceDB-backed hybrid retriever.

Why LanceDB over the alternatives, for *this* project's "small, local, no-ops"
philosophy:

* Embedded (no server) — same operational shape as the rest of the stack.
* Native hybrid search (dense ANN + FTS/BM25) in a single query, with built-in
  RRF fusion; we don't have to hand-roll the merge.
* First-class multivector support if we later swap dense BGE-M3 for the full
  ColBERT-style late-interaction path.
* Columnar Lance format — cheap to scan, cheap to incrementally add to.

DuckDB-VSS was the obvious comparison. It's promising (HNSW via usearch, FTS
via the ``fts`` extension), but as of 2026 the VSS extension is still flagged
experimental and multivector support is not first class — meaning the ColBERT
upgrade path requires hand-rolled multi-table joins. We picked LanceDB to
preserve the upgrade path.

The retriever exposes a single sync ``search`` so ``GeneralProgram`` can use
it as a plain DSPy tool function without async juggling.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import Document, Retriever
from .embedder import Embedder
from .reranker import IdentityReranker, Reranker

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "documents"


class LanceDBRetriever:
    """Hybrid retriever (dense + BM25 FTS) over a LanceDB table.

    Construct with an open :class:`Embedder` and (optionally) a :class:`Reranker`.
    The table schema is fixed at ingest time: ``id: str``, ``text: str``,
    ``vector: float32[dim]``, ``source: str``, ``metadata: str`` (JSON).
    """

    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder,
        reranker: Reranker | None = None,
        table_name: str = DEFAULT_TABLE,
        prefetch_k: int = 30,
    ) -> None:
        self.db_path = str(db_path)
        self.embedder = embedder
        self.reranker = reranker or IdentityReranker()
        self.table_name = table_name
        self.prefetch_k = prefetch_k
        self._db: object | None = None
        self._table: object | None = None

    # ------------------------------------------------------------------
    # Lazy init — keeps lancedb out of the import path until used.
    # ------------------------------------------------------------------

    def _ensure_table(self) -> object:
        if self._table is not None:
            return self._table
        import lancedb

        self._db = lancedb.connect(self.db_path)
        self._table = self._db.open_table(self.table_name)  # type: ignore[union-attr]
        return self._table

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> Sequence[Document]:
        """Hybrid retrieve → optional rerank → top-k."""
        table = self._ensure_table()
        query_vec = self.embedder([query])[0]

        # LanceDB hybrid search: dense (vector) + FTS (BM25) with RRF fusion.
        # ``query_type="hybrid"`` requires an FTS index on ``text``; the
        # ingest script enforces that.
        results = (
            table.search(  # type: ignore[attr-defined]
                query=(query_vec, query),
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .limit(self.prefetch_k)
            .to_list()
        )
        docs = [self._row_to_doc(row) for row in results]
        return self.reranker.rerank(query, docs, top_k=k)

    @staticmethod
    def _row_to_doc(row: dict[str, Any]) -> Document:
        import json

        raw_meta = row.get("metadata") or "{}"
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
        except (ValueError, TypeError):
            meta = {}
        return Document(
            text=str(row.get("text", "")),
            score=float(row.get("_relevance_score", row.get("_distance", 0.0)) or 0.0),
            source=row.get("source"),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Ingest helpers — kept here so the same import surface owns both reads and
# writes. The ingest script in ``scripts/build_rag_index.py`` calls these.
# ---------------------------------------------------------------------------


def create_or_replace_table(
    db_path: str | Path,
    rows: list[dict[str, Any]],
    embedder: Embedder,
    table_name: str = DEFAULT_TABLE,
) -> None:
    """Write ``rows`` to a fresh table, recomputing dense vectors.

    Each row must have ``id``, ``text``, ``source``, and ``metadata`` keys.
    Adds a dense ``vector`` column from ``embedder(text)`` and an FTS index
    over ``text`` so ``query_type="hybrid"`` works.
    """
    if not rows:
        raise ValueError("nothing to ingest — rows is empty")
    import json

    import lancedb
    import pyarrow as pa

    texts = [r["text"] for r in rows]
    vectors = embedder(texts)
    dim = len(vectors[0])

    enriched = [
        {
            "id": r["id"],
            "text": r["text"],
            "vector": vec,
            "source": r.get("source") or "unknown",
            "metadata": json.dumps(r.get("metadata") or {}),
        }
        for r, vec in zip(rows, vectors, strict=True)
    ]

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source", pa.string()),
            pa.field("metadata", pa.string()),
        ]
    )

    db = lancedb.connect(str(db_path))
    if table_name in db.table_names():
        db.drop_table(table_name)
    table = db.create_table(table_name, data=enriched, schema=schema)
    table.create_fts_index("text", replace=True)
    logger.info(
        "rag.ingest.done",
        extra={"db_path": str(db_path), "table": table_name, "rows": len(enriched), "dim": dim},
    )


# Help static type checkers see the Protocol conformance.
_: Retriever = LanceDBRetriever.__new__(LanceDBRetriever)
del _
