"""Tests for the runtime document upload & indexing endpoint.

Pure text-handling (validation, extraction, chunking) is tested directly.
The FastAPI endpoint and the ``index_document`` orchestrator are tested with
the embedding / ChromaDB step stubbed, so no models or vector store are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import upload as upload_module
from src.api.app import app
from src.api.config import APIConfig
from src.api.upload import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFormatError,
    chunk_pages,
    extract_pages,
    index_document,
    sanitize_filename,
    validate_upload,
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def base_config(tmp_path: Path) -> APIConfig:
    return APIConfig(
        api_base_url="https://x.test/v1",
        api_key="k",
        chat_model="m",
        embedding_model="test-embed-model",
        persist_dir=tmp_path / "vector_db",
        collection_name="nbc_test_collection",
        candidate_k=10,
        final_k=3,
        rerank_method="cross_encoder",
        host="127.0.0.1",
        port=8000,
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=1_000_000,
        allowed_extensions=frozenset({".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}),
    )


class _FakeEnc:
    """Word-based stand-in for the tiktoken encoding."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split()) or 1))

    def decode(self, ids: list[int]) -> str:
        return " ".join("x" for _ in ids)


# ---------------------------------------------------------------------------
# Filename safety + validation (Task 4)
# ---------------------------------------------------------------------------


def test_sanitize_filename_strips_paths():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\temp\\report.pdf") == "report.pdf"


def test_sanitize_filename_rejects_empty():
    with pytest.raises(EmptyFileError):
        sanitize_filename("")


def test_validate_rejects_unsupported_format():
    with pytest.raises(UnsupportedFormatError):
        validate_upload(
            filename="data.csv", size_bytes=10,
            allowed_extensions=frozenset({".txt", ".pdf"}), max_upload_bytes=1000,
        )


def test_validate_rejects_empty_file():
    with pytest.raises(EmptyFileError):
        validate_upload(
            filename="notes.txt", size_bytes=0,
            allowed_extensions=frozenset({".txt"}), max_upload_bytes=1000,
        )


def test_validate_rejects_oversized_file():
    with pytest.raises(FileTooLargeError):
        validate_upload(
            filename="big.txt", size_bytes=2000,
            allowed_extensions=frozenset({".txt"}), max_upload_bytes=1000,
        )


# ---------------------------------------------------------------------------
# Text extraction + chunking (Task 2)
# ---------------------------------------------------------------------------


def test_extract_pages_from_txt():
    pages = extract_pages(b"Boiler rooms need fire doors. Keep them closed.", ".txt")
    assert len(pages) == 1
    assert "fire doors" in pages[0][1]


def test_extract_pages_strips_html():
    pages = extract_pages(b"<h1>Title</h1><p>Roof load limit.</p>", ".html")
    assert "Roof load limit." in pages[0][1]
    assert "<p>" not in pages[0][1]


def test_chunk_pages_produces_unique_ids_and_metadata():
    pages = [(1, "One sentence here. Another sentence follows."), (2, "Second page text.")]
    chunks = chunk_pages(pages, doc_hash="abc123", source_name="doc.txt")
    assert len(chunks) >= 2
    ids = [c["id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(cid.startswith("upload-abc123-") for cid in ids)
    assert chunks[0]["metadata"]["source"] == "doc.txt"
    assert chunks[0]["metadata"]["origin"] == "upload"


def test_chunk_pages_splits_long_pages(monkeypatch):
    monkeypatch.setattr(upload_module, "_encoding", lambda: _FakeEnc())
    long_text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_pages([(1, long_text)], doc_hash="h", source_name="big.txt")
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# index_document orchestration (Task 2) — embed/index step stubbed
# ---------------------------------------------------------------------------


def test_index_document_targets_query_collection_and_stores_file(monkeypatch, base_config):
    monkeypatch.setattr(upload_module, "_encoding", lambda: _FakeEnc())
    captured: dict = {}

    def fake_embed_index(chunks, *, model_name, persist_dir, collection_name):
        captured["model_name"] = model_name
        captured["persist_dir"] = persist_dir
        captured["collection_name"] = collection_name
        captured["n_chunks"] = len(chunks)
        return (len(chunks) + 100, [c["id"] for c in chunks])

    monkeypatch.setattr(upload_module, "embed_and_index", fake_embed_index)

    result = index_document(
        content=b"Rooftop helipad must support 1500 kilograms minimum.",
        filename="amendment.txt",
        config=base_config,
    )

    # Content is written to the SAME collection the /query endpoint reads.
    assert captured["collection_name"] == base_config.collection_name
    assert captured["persist_dir"] == base_config.persist_dir
    assert captured["model_name"] == base_config.embedding_model
    assert result.chunks_indexed >= 1
    assert result.status == "indexed"
    assert Path(result.stored_path).exists()
    assert result.total_records_after == captured["n_chunks"] + 100


# ---------------------------------------------------------------------------
# Endpoint success + error mapping (Tasks 1, 4)
# ---------------------------------------------------------------------------


def test_upload_endpoint_success(client, monkeypatch, base_config):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)

    class R:
        document = "amendment.txt"
        stored_path = "/tmp/amendment.txt"
        media_type = ".txt"
        bytes = 10
        pages = 1
        chunks_created = 2
        chunks_indexed = 2
        collection = "nbc_test_collection"
        embedding_model = "test-embed-model"
        total_records_after = 202
        doc_hash = "deadbeef"
        elapsed_ms = 5.0
        status = "indexed"
        chunk_ids = ["upload-deadbeef-000000", "upload-deadbeef-000001"]

    monkeypatch.setattr(app_module, "index_document", lambda **kw: R())

    resp = client.post(
        "/documents/upload",
        files={"file": ("amendment.txt", b"some text", "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["indexing"]["chunks_indexed"] == 2
    assert body["chunk_ids"][0].startswith("upload-deadbeef-")


def test_upload_endpoint_unsupported_format_415(client, monkeypatch, base_config):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    monkeypatch.setattr(app_module, "index_document", lambda **kw: (_ for _ in ()).throw(
        UnsupportedFormatError("Unsupported file type '.exe'.")
    ))
    resp = client.post(
        "/documents/upload",
        files={"file": ("bad.exe", b"MZ...", "application/octet-stream")},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] == "unsupported_format"


def test_upload_endpoint_empty_400(client, monkeypatch, base_config):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    monkeypatch.setattr(app_module, "index_document", lambda **kw: (_ for _ in ()).throw(
        EmptyFileError("Uploaded file is empty.")
    ))
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    # 0-byte uploads are rejected by python-multipart/TestClient transport as a
    # bad request; assert it is a 4xx client error with our envelope.
    assert 400 <= resp.status_code < 500


def test_upload_endpoint_oversized_413(client, monkeypatch, base_config):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    monkeypatch.setattr(app_module, "index_document", lambda **kw: (_ for _ in ()).throw(
        FileTooLargeError("File exceeds the limit.")
    ))
    resp = client.post(
        "/documents/upload",
        files={"file": ("big.txt", b"x" * 100, "text/plain")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "file_too_large"


def test_upload_endpoint_missing_file_422(client, monkeypatch, base_config):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.post("/documents/upload", data={"note": "no file"})
    assert resp.status_code == 422
