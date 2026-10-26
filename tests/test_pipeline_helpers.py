
from unittest.mock import Mock

import pytest

from reading.ingestion import load_pdfs
from reading.pipeline import RAGEngine


def test_empty_question_is_rejected():
    engine = RAGEngine()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        engine.ask("   ", strategy="direct")


def test_missing_pdf_is_rejected(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdfs([missing_pdf])


def test_non_pdf_is_rejected(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Example", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdfs([text_file])


def test_ingestion_builds_other_indexes_on_demand(monkeypatch, tmp_path):
    import reading.pipeline as pipeline

    documents = [object()]
    direct_index = object()
    sentence_window_index = object()
    build_index = Mock(return_value=direct_index)

    monkeypatch.setattr(pipeline, "load_directory", lambda _: documents)
    monkeypatch.setattr(pipeline, "build_and_persist_index", build_index)
    monkeypatch.setattr(
        pipeline,
        "index_directory",
        lambda strategy: tmp_path / strategy,
    )

    for strategy in ("sentence_window", "auto_merging"):
        stale_index = tmp_path / strategy
        stale_index.mkdir()
        (stale_index / "docstore.json").touch()

    engine = RAGEngine()
    result = engine.ingest_directory(tmp_path)

    assert result["indexes"] == ["direct"]
    assert build_index.call_args.args == (documents, "direct")
    assert not (tmp_path / "sentence_window").exists()
    assert not (tmp_path / "auto_merging").exists()

    build_index.return_value = sentence_window_index
    assert engine.get_index("sentence_window") is sentence_window_index
    assert build_index.call_args.args == (documents, "sentence_window")
