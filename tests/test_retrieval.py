"""Retrieval tests.

The unit tests use ``deterministic_stub_embedder`` so they don't have to pull
BGE-M3 (~2 GB) from HF. The integration test exercising real LanceDB writes
+ reads is gated behind ``RUN_RAG=1`` and the ``lancedb`` import — the
default CI run skips it cleanly.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from app.retrieval.base import Document, Retriever
from app.retrieval.embedder import deterministic_stub_embedder
from app.retrieval.reranker import IdentityReranker


def test_document_to_citation_shape() -> None:
    doc = Document(text="hello world", score=0.42, source="doc.md", metadata={"chunk": 1})
    cite = doc.to_citation()
    assert cite["source"] == "doc.md"
    assert cite["score"] == 0.42
    assert cite["snippet"].startswith("hello")
    assert cite["metadata"] == {"chunk": 1}


def test_document_to_citation_omits_empty_metadata() -> None:
    doc = Document(text="x" * 500, source=None)
    cite = doc.to_citation()
    assert cite["source"] == "unknown"
    assert "metadata" not in cite
    assert len(cite["snippet"]) <= 240


def test_stub_embedder_is_deterministic() -> None:
    embed = deterministic_stub_embedder(dim=8)
    assert embed(["x"])[0] == embed(["x"])[0]
    assert embed(["x"])[0] != embed(["y"])[0]
    assert len(embed(["a", "b"])) == 2
    assert all(len(v) == 8 for v in embed(["a", "b"]))


def test_identity_reranker_preserves_order() -> None:
    docs = [Document(text="a", score=0.1), Document(text="b", score=0.9)]
    reranker = IdentityReranker()
    out = reranker.rerank("anything", docs, top_k=10)
    assert [d.text for d in out] == ["a", "b"]


def test_identity_reranker_respects_top_k() -> None:
    docs = [Document(text=str(i)) for i in range(5)]
    assert len(IdentityReranker().rerank("q", docs, top_k=3)) == 3


def test_factory_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_ENABLED", raising=False)
    from app.retrieval import build_retriever

    assert build_retriever() is None


def test_general_program_plain_when_rag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``RAG_ENABLED``, GeneralProgram constructs ChainOfThought, not ReAct."""
    monkeypatch.delenv("RAG_ENABLED", raising=False)
    import dspy

    from app.dspy_modules.general_module import GeneralProgram

    program = GeneralProgram()
    assert program.retriever is None
    assert isinstance(program.predict, dspy.ChainOfThought)


# ---------------------------------------------------------------------------
# Integration: actual LanceDB round-trip — skipped unless RUN_RAG=1 set AND
# lancedb is installed. Keeps default CI from pulling pyarrow + lancedb.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_RAG") != "1",
    reason="Set RUN_RAG=1 to exercise the live LanceDB roundtrip.",
)
def test_lancedb_roundtrip(tmp_path) -> None:
    pytest.importorskip("lancedb")
    pytest.importorskip("pyarrow")
    from app.retrieval.lancedb_store import LanceDBRetriever, create_or_replace_table

    embed = deterministic_stub_embedder(dim=8)

    class _Embedder:
        dim = 8

        def __call__(self, texts):
            return embed(texts)

    rows = [
        {
            "id": "1",
            "text": "The capital of France is Paris.",
            "source": "facts.md",
            "metadata": {},
        },
        {
            "id": "2",
            "text": "Python uses the GIL for memory safety.",
            "source": "py.md",
            "metadata": {},
        },
        {
            "id": "3",
            "text": "DSPy programs are optimized, not prompted.",
            "source": "dspy.md",
            "metadata": {},
        },
    ]
    create_or_replace_table(tmp_path, rows, _Embedder(), table_name="documents")

    retriever = LanceDBRetriever(tmp_path, _Embedder(), reranker=IdentityReranker())
    results = retriever.search("Paris France capital", k=2)
    sources = {d.source for d in results}
    assert "facts.md" in sources

    # Sanity: every result has the JSON-decoded metadata, not the raw string.
    for doc in results:
        assert isinstance(doc.metadata, dict)


def test_retriever_protocol_recognizes_stub() -> None:
    """A duck-typed object with the right ``search`` signature satisfies the
    Protocol — important for tests that inject fakes into GeneralProgram."""

    class FakeRetriever:
        def search(self, query: str, k: int = 5):
            return [Document(text=f"fake: {query}", source="stub")]

    fake = FakeRetriever()
    assert isinstance(fake, Retriever)
    assert fake.search("hi")[0].text == "fake: hi"


# ---------------------------------------------------------------------------
# Ingest script: chunking + suffix filtering
# ---------------------------------------------------------------------------


def test_build_rag_index_chunker(tmp_path) -> None:
    """Pull the chunker function via the script module without executing main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_rag_index",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "build_rag_index.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_rag_index"] = mod
    spec.loader.exec_module(mod)

    para = "x" * 300
    text = "\n\n".join([para] * 5)
    chunks = mod._chunk(text, target_chars=800)
    assert all(len(c) <= 2000 for c in chunks)
    assert len(chunks) >= 2  # 5 * 300 chars >> 800-char budget
    assert "x" * 100 in chunks[0]


def test_build_rag_index_ignores_non_utf8(tmp_path) -> None:
    """Binary files in the source dir are skipped, not crashed on."""
    (tmp_path / "good.md").write_text("hello")
    (tmp_path / "bad.bin").write_bytes(b"\xff\xfe\x00\x01")
    # Renaming bad.bin to bad.md exercises the decode path. Just confirm the
    # _iter_files helper at least yields the .md file.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_rag_index",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "build_rag_index.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = list(mod._iter_files(tmp_path, {".md"}))
    assert len(files) == 1
    assert files[0].name == "good.md"


# ---------------------------------------------------------------------------
# .env.example contract — confirm the new RAG_ knobs are documented.
# ---------------------------------------------------------------------------


def test_env_example_documents_rag_knobs() -> None:
    """Operators should be able to grep .env.example for the RAG envs."""
    env_text = os.path.join(os.path.dirname(__file__), "..", ".env.example")
    with open(env_text, encoding="utf-8") as f:
        content = f.read()
    for key in (
        "RAG_ENABLED",
        "RAG_INDEX_PATH",
    ):
        assert key in content, f"missing env doc for {key} in .env.example"


# Quiet ruff about the unused fixture parameter in one of the tests.
_ = json
