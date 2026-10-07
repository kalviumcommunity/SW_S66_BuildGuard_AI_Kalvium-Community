from pathlib import Path

from src.ingestion_pipeline import (
    IngestionValidationError,
    ingest_corpus,
    validate_ingestion_report,
)


def make_corpus(root: Path) -> None:
    (root / "nested").mkdir()
    (root / "good.txt").write_text("A valid workplace procedure with enough content.", encoding="utf-8")
    (root / "nested" / "guide.md").write_text("# Guide\nUse the documented process.", encoding="utf-8")
    (root / "nested" / "contact.html").write_text("<p>Contact the support team.</p>", encoding="utf-8")
    (root / "unsupported.csv").write_text("not supported", encoding="utf-8")
    (root / "broken.txt").write_bytes(b"\xff\xfe invalid UTF-8")


def test_ingestion_reconciles_all_sources_and_chunks(tmp_path: Path) -> None:
    make_corpus(tmp_path)

    report = ingest_corpus(tmp_path)

    assert report["complete"] is True
    assert report["total_sources"] == 5
    assert report["ingested"] == 3
    assert report["skipped"] == 2
    assert report["total_sources"] == report["ingested"] + report["skipped"]
    assert {entry["source"] for entry in report["sources"]} == {
        "good.txt",
        "nested/guide.md",
        "nested/contact.html",
        "unsupported.csv",
        "broken.txt",
    }
    assert all(chunk["source"] in {"good.txt", "nested/guide.md", "nested/contact.html"} for chunk in report["chunks"])
    assert all(chunk["metadata"]["source"] == chunk["source"] for chunk in report["chunks"])

    reasons = {entry["source"]: entry["reason"] for entry in report["sources"] if entry["status"] == "skipped"}
    assert "unsupported" in reasons["unsupported.csv"]
    assert "UTF-8" in reasons["broken.txt"]


def test_dropped_source_is_rejected(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    report = ingest_corpus(tmp_path)
    report["chunks"] = [chunk for chunk in report["chunks"] if chunk["source"] != "good.txt"]
    report["total_chunks"] = len(report["chunks"])
    report["sources"][0]["chunk_count"] = 0

    try:
        validate_ingestion_report(report)
    except IngestionValidationError as error:
        assert "no chunks" in str(error)
    else:
        raise AssertionError("dropped source was not rejected")


def test_missing_or_empty_corpus_fails_without_fabricating_sources(tmp_path: Path) -> None:
    report = ingest_corpus(tmp_path / "missing")

    assert report["complete"] is False
    assert report["total_sources"] == 0
    assert report["sources"] == []
    assert "missing or empty" in report["validation_error"]
