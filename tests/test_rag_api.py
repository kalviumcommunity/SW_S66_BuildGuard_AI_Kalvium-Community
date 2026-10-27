"""Tests for the RAG backend HTTP API (src.api.app).

Uses FastAPI's TestClient and a stubbed ``run_pipeline`` so the request
validation, response serialization, and error-handling paths are exercised
without loading embedding models, ChromaDB, or calling a live LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.config import APIConfig
from src.api.app import app
from src.rag_pipeline import PipelineResult


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def base_config() -> APIConfig:
    return APIConfig(
        api_base_url="https://api.example.test/v1",
        api_key="test-key",
        chat_model="test-chat-model",
        embedding_model="test-embed-model",
        persist_dir=Path("outputs/vector_db"),
        collection_name="test_collection",
        candidate_k=10,
        final_k=3,
        rerank_method="cross_encoder",
        host="127.0.0.1",
        port=8000,
        default_dry_run=False,
    )


def _sample_result(query: str) -> PipelineResult:
    return PipelineResult(
        query=query,
        answer="Boiler room walls require a 2-hour fire rating [Volume 1, PDF p. 314].",
        sources=[
            {
                "rank": 1,
                "id": "nbc-v1-p314-001012",
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page": "314-315",
                "heading": "Part 4 — Fire and Life Safety",
                "cosine_similarity": 0.8264,
                "rerank_score": 4.0295,
                "text_preview": "120 min fire resistance rating wall.",
            }
        ],
        context_block="=== RETRIEVED BUILDING CODE CONTEXT === ...",
        stages={
            "embed": {"model": "test-embed-model", "vector_dim": 384, "elapsed_ms": 5.0},
            "retrieve": {"candidate_k": 10, "retrieved_count": 10, "hybrid": True, "elapsed_ms": 3.0},
            "rerank": {"method": "cross_encoder", "final_k": 3, "returned_count": 1, "elapsed_ms": 8.0},
            "assemble": {"chunk_count": 1, "context_chars": 42, "elapsed_ms": 0.0},
            "generate": {"model": "test-chat-model", "answer_chars": 60, "elapsed_ms": 12.0},
        },
        generated_at="2026-10-27T00:00:00Z",
        pipeline_elapsed_ms=28.0,
    )


# ---------------------------------------------------------------------------
# Task 1 & 2 — query endpoint returns grounded answer + sources + metadata
# ---------------------------------------------------------------------------


def test_query_returns_structured_json(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    monkeypatch.setattr(app_module, "run_pipeline", lambda *a, **k: _sample_result(a[0]))

    resp = client.post("/query", json={"question": "Boiler room fire rating?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "2-hour fire rating" in body["answer"]
    assert body["query"] == "Boiler room fire rating?"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["page"] == "314-315"
    assert body["metadata"]["candidate_k"] == 10
    assert body["metadata"]["final_k"] == 3
    assert body["metadata"]["dry_run"] is False
    assert body["metadata"]["source_count"] == 1
    assert body["metadata"]["model"] == "test-chat-model"


def test_query_dry_run_flag_propagates(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return _sample_result(args[0])

    monkeypatch.setattr(app_module, "run_pipeline", fake_run)
    resp = client.post("/query", json={"question": "x", "dry_run": True})
    assert resp.status_code == 200
    assert captured["dry_run"] is True
    assert resp.json()["metadata"]["dry_run"] is True


def test_query_uses_config_defaults_for_k(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return _sample_result(args[0])

    monkeypatch.setattr(app_module, "run_pipeline", fake_run)
    client.post("/query", json={"question": "x"})
    assert captured["candidate_k"] == 10
    assert captured["final_k"] == 3


def test_query_request_overrides_config_k(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return _sample_result(args[0])

    monkeypatch.setattr(app_module, "run_pipeline", fake_run)
    client.post("/query", json={"question": "x", "candidate_k": 7, "final_k": 2})
    assert captured["candidate_k"] == 7
    assert captured["final_k"] == 2


def test_query_no_sources_returns_no_context_status(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    empty = _sample_result("unanswerable?")
    empty.sources = []
    monkeypatch.setattr(app_module, "run_pipeline", lambda *a, **k: empty)

    resp = client.post("/query", json={"question": "unanswerable?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_context"
    assert resp.json()["sources"] == []


# ---------------------------------------------------------------------------
# Task 3 — validation + error status codes
# ---------------------------------------------------------------------------


def test_missing_question_returns_422(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.post("/query", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "error"
    assert "question" in body["fields"]


def test_blank_question_returns_422(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.post("/query", json={"question": "    "})
    assert resp.status_code == 422
    assert resp.json()["status"] == "error"


def test_unknown_field_returns_422(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.post("/query", json={"question": "x", "bogus": 1})
    assert resp.status_code == 422


def test_out_of_range_k_returns_422(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.post("/query", json={"question": "x", "final_k": 0})
    assert resp.status_code == 422


def test_live_generation_without_key_returns_503(client, monkeypatch):
    cfg = APIConfig(
        api_base_url="", api_key="", chat_model="",
        embedding_model="m", persist_dir=Path("outputs/vector_db"),
        collection_name="c", candidate_k=10, final_k=3,
        rerank_method="cross_encoder", host="127.0.0.1", port=8000,
        default_dry_run=False,
    )
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: cfg)
    resp = client.post("/query", json={"question": "x", "dry_run": False})
    assert resp.status_code == 503


def test_pipeline_error_returns_500(base_config, monkeypatch):
    # raise_server_exceptions=False lets the app's 500 handler produce a body.
    error_client = TestClient(app, raise_server_exceptions=False)
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)

    def boom(*args, **kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(app_module, "run_pipeline", boom)
    resp = error_client.post("/query", json={"question": "x"})
    assert resp.status_code == 500
    assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# Health endpoint (Task 4 — reflects env config)
# ---------------------------------------------------------------------------


def test_health_reports_configured(client, base_config, monkeypatch):
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: base_config)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chat_configured"] is True
    assert body["collection_name"] == "test_collection"


def test_health_reports_missing_chat_settings(client, monkeypatch):
    cfg = APIConfig(
        api_base_url="", api_key="", chat_model="",
        embedding_model="m", persist_dir=Path("outputs/vector_db"),
        collection_name="c", candidate_k=10, final_k=3,
        rerank_method="cross_encoder", host="127.0.0.1", port=8000,
    )
    monkeypatch.setattr(app_module, "get_config", lambda *a, **k: cfg)
    resp = client.get("/health")
    body = resp.json()
    assert body["chat_configured"] is False
    assert set(body["missing_chat_settings"]) == {"API_BASE_URL", "API_KEY", "CHAT_MODEL"}
