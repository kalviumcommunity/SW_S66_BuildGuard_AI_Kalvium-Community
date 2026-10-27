"""Unit tests for the RAG UI HTTP client (src.ui.client).

Validates request formation, response parsing, and error mapping (422, 503, 500,
connection failure, and timeout) using mocked HTTP responses without requiring
a live backend.
"""

from __future__ import annotations

import httpx
import pytest

from src.ui.client import (
    RAGClient,
    RAGClientError,
    RAGConnectionError,
    RAGInternalError,
    RAGServiceUnavailableError,
    RAGTimeoutError,
    RAGValidationError,
)


def test_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_get(url: str, **kwargs):
        return httpx.Response(
            status_code=200,
            json={
                "status": "ok",
                "config_loaded": True,
                "chat_configured": False,
                "missing_chat_settings": ["OPENAI_API_KEY"],
                "embedding_model": "test-model",
                "collection_name": "test-collection",
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: mock_get(url, **kw))

    info = client.health()
    assert info["status"] == "ok"
    assert info["embedding_model"] == "test-model"


def test_health_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_get(url: str, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: mock_get(url, **kw))

    with pytest.raises(RAGConnectionError) as exc_info:
        client.health()
    assert "Cannot connect" in str(exc_info.value)


def test_query_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    expected_payload = {
        "status": "success",
        "answer": "Boiler room walls require a 2-hour rating.",
        "query": "What is the boiler room rating?",
        "sources": [
            {
                "rank": 1,
                "id": "chunk-101",
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page": "314",
                "heading": "Fire Safety",
                "cosine_similarity": 0.85,
                "rerank_score": 4.1,
                "text_preview": "120 min fire resistance...",
            }
        ],
        "metadata": {
            "model": "dry-run",
            "source_count": 1,
            "pipeline_elapsed_ms": 120.5,
        },
    }

    def mock_post(url: str, **kwargs):
        assert kwargs.get("json")["question"] == "What is the boiler room rating?"
        assert kwargs.get("json")["candidate_k"] == 10
        assert kwargs.get("json")["dry_run"] is True
        return httpx.Response(
            status_code=200,
            json=expected_payload,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    res = client.query("What is the boiler room rating?", candidate_k=10, dry_run=True)
    assert res["status"] == "success"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["id"] == "chunk-101"


def test_query_validation_error_422(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        return httpx.Response(
            status_code=422,
            json={
                "status": "error",
                "error": "invalid_request",
                "detail": "Request body failed validation.",
                "fields": ["question"],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    with pytest.raises(RAGValidationError) as exc_info:
        client.query("")
    assert exc_info.value.status_code == 422
    assert "Fields: question" in str(exc_info.value)


def test_query_service_unavailable_503(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        return httpx.Response(
            status_code=503,
            json={
                "status": "error",
                "error": "service_unavailable",
                "detail": "Missing OPENAI_API_KEY",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    with pytest.raises(RAGServiceUnavailableError) as exc_info:
        client.query("test question", dry_run=False)
    assert exc_info.value.status_code == 503
    assert "Missing OPENAI_API_KEY" in str(exc_info.value)


def test_query_internal_error_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        return httpx.Response(
            status_code=500,
            json={
                "status": "error",
                "error": "internal_error",
                "detail": "Database lock timeout",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    with pytest.raises(RAGInternalError) as exc_info:
        client.query("test question")
    assert exc_info.value.status_code == 500
    assert "Database lock timeout" in str(exc_info.value)


def test_query_connection_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        raise httpx.ConnectError("Failed to connect")

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    with pytest.raises(RAGConnectionError):
        client.query("test question")


def test_query_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        raise httpx.TimeoutException("Timed out")

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    with pytest.raises(RAGTimeoutError):
        client.query("test question")


def test_upload_document_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RAGClient("http://127.0.0.1:8000")

    def mock_post(url: str, **kwargs):
        assert "files" in kwargs
        return httpx.Response(
            status_code=201,
            json={
                "status": "success",
                "message": "Indexed 4 chunks",
                "indexing_summary": {
                    "document": "circular.pdf",
                    "media_type": "application/pdf",
                    "raw_bytes": 1024,
                    "extracted_chars": 2000,
                    "chunks_created": 4,
                    "chunks_upserted": 4,
                    "collection": "test_collection",
                    "elapsed_ms": 230.0,
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: mock_post(url, **kw))

    res = client.upload_document(b"%PDF-test", "circular.pdf")
    assert res["status"] == "success"
    assert res["indexing_summary"]["chunks_created"] == 4
