"""Ingest a directory of documents into the LanceDB index used by GeneralExpert.

Walks ``--source-dir`` for ``*.md``, ``*.txt``, ``*.rst`` files (configurable),
chunks each on paragraph boundaries, embeds with BGE-M3, and writes a fresh
LanceDB table with an FTS index over the text column.

Example:

    uv run python scripts/build_rag_index.py \\
        --source-dir docs \\
        --index-path rag/index \\
        --chunk-chars 800

The resulting index is what ``RAG_INDEX_PATH=rag/index RAG_ENABLED=1`` reads
from at runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_rag_index")


def _iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _chunk(text: str, target_chars: int) -> list[str]:
    """Split on blank lines, then re-glue into ~``target_chars`` chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) > target_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=Path("rag/index"))
    parser.add_argument("--table-name", default="documents")
    parser.add_argument("--chunk-chars", type=int, default=800)
    parser.add_argument(
        "--suffixes",
        default=".md,.txt,.rst",
        help="Comma-separated list of file extensions to ingest.",
    )
    parser.add_argument(
        "--embedder-model",
        default="BAAI/bge-m3",
        help="HF model id for the dense embedder.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    suffixes = {s.strip().lower() for s in args.suffixes.split(",") if s.strip()}
    rows: list[dict[str, object]] = []
    for path in _iter_files(args.source_dir, suffixes):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping non-utf8 file: %s", path)
            continue
        for chunk in _chunk(text, args.chunk_chars):
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": str(path.relative_to(args.source_dir)),
                    "metadata": {"path": str(path)},
                }
            )

    if not rows:
        logger.error("no documents found under %s with suffixes %s", args.source_dir, suffixes)
        return 1

    from app.retrieval.embedder import BGEM3Embedder
    from app.retrieval.lancedb_store import create_or_replace_table

    embedder = BGEM3Embedder(model_name=args.embedder_model)
    args.index_path.mkdir(parents=True, exist_ok=True)
    create_or_replace_table(
        db_path=args.index_path,
        rows=rows,
        embedder=embedder,
        table_name=args.table_name,
    )
    logger.info("ingested %d chunks into %s/%s", len(rows), args.index_path, args.table_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
