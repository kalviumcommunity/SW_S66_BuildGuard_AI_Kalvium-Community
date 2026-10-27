"""HTTP client for communicating with the BuildGuard RAG FastAPI backend.

Provides a clean interface for querying the RAG pipeline, checking service
health, and uploading new documents with typed exceptions and timeout handling.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RAGClientError(Exception):
    """Base exception for all RAG client errors."""

    def __init__(self, message: str, status_code: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class RAGConnectionError(RAGClientError):
    """Raised when the backend API server cannot be reached."""


class RAGTimeoutError(RAGClientError):
    """Raised when a request to the backend API times out."""


class RAGValidationError(RAGClientError):
    """Raised when the backend returns 422 Validation Error."""


class RAGServiceUnavailableError(RAGClientError):
    """Raised when the backend returns 503 (e.g. missing API keys for live LLM)."""


class RAGInternalError(RAGClientError):
    """Raised when the backend returns 500 Internal Server Error."""


class RAGClient:
    """Synchronous HTTP client for interacting with the RAG backend service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        """Check backend service liveness and configuration state.

        Returns:
            dict containing status, config_loaded, chat_configured, etc.
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise RAGConnectionError(
                f"Cannot connect to BuildGuard API at {self.base_url}. Ensure the FastAPI server is running.",
                details=str(exc),
            ) from exc
        except httpx.TimeoutException as exc:
            raise RAGTimeoutError(
                f"Health check timed out connecting to {self.base_url}.",
                details=str(exc),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RAGClientError(
                f"Health check failed with status {exc.response.status_code}.",
                status_code=exc.response.status_code,
                details=exc.response.text,
            ) from exc
        except Exception as exc:
            raise RAGClientError(f"Unexpected error checking health: {exc}", details=str(exc)) from exc

    def query(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
        filter_heading: str | None = None,
        hybrid: bool | None = None,
        dry_run: bool | None = None,
    ) -> dict[str, Any]:
        """Send a natural-language question to the RAG backend.

        Args:
            question: Natural language NBC question.
            candidate_k: Initial retrieval pool size.
            final_k: Number of re-ranked chunks to provide as context.
            filter_heading: Optional section heading filter.
            hybrid: Whether to use hybrid (dense + keyword) search.
            dry_run: If True, skip LLM call and return synthetic answer.

        Returns:
            dict containing status, answer, query, sources, metadata.
        """
        url = f"{self.base_url}/query"
        payload: dict[str, Any] = {"question": question}
        if candidate_k is not None:
            payload["candidate_k"] = candidate_k
        if final_k is not None:
            payload["final_k"] = final_k
        if filter_heading:
            payload["filter_heading"] = filter_heading
        if hybrid is not None:
            payload["hybrid"] = hybrid
        if dry_run is not None:
            payload["dry_run"] = dry_run

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise RAGConnectionError(
                f"Connection refused at {self.base_url}. Is the backend running? "
                f"Run `uvicorn src.api.app:app --port 8000` to start it.",
                details=str(exc),
            ) from exc
        except httpx.TimeoutException as exc:
            raise RAGTimeoutError(
                f"Request to {url} timed out after {self.timeout}s.",
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise RAGClientError(
                f"Error communicating with {self.base_url}: {exc}",
                details=str(exc),
            ) from exc

        if resp.status_code == 200:
            return resp.json()

        # Parse error envelope
        try:
            error_data = resp.json()
        except Exception:
            error_data = {"detail": resp.text}

        detail = error_data.get("detail") or error_data.get("error") or resp.text

        if resp.status_code == 422:
            fields = error_data.get("fields", [])
            msg = f"Validation Error (422): {detail}"
            if fields:
                msg += f" (Fields: {', '.join(fields)})"
            raise RAGValidationError(msg, status_code=422, details=error_data)

        if resp.status_code == 503:
            raise RAGServiceUnavailableError(
                f"Service Unavailable (503): {detail}",
                status_code=503,
                details=error_data,
            )

        if resp.status_code == 500:
            raise RAGInternalError(
                f"Internal Server Error (500): {detail}",
                status_code=500,
                details=error_data,
            )

        raise RAGClientError(
            f"API Error ({resp.status_code}): {detail}",
            status_code=resp.status_code,
            details=error_data,
        )

    def upload_document(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """Upload and index a document into the active RAG vector database.

        Args:
            file_bytes: Raw binary content of the file (.pdf, .txt, .md, .html).
            filename: Original file name.

        Returns:
            dict containing status, message, indexing_summary.
        """
        url = f"{self.base_url}/documents/upload"
        files = {"file": (filename, file_bytes)}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, files=files)
        except httpx.ConnectError as exc:
            raise RAGConnectionError(
                f"Connection refused at {self.base_url}.",
                details=str(exc),
            ) from exc
        except httpx.TimeoutException as exc:
            raise RAGTimeoutError(
                f"Document upload to {url} timed out after {self.timeout}s.",
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise RAGClientError(f"Upload error: {exc}", details=str(exc)) from exc

        if resp.status_code in (200, 201):
            return resp.json()

        try:
            error_data = resp.json()
        except Exception:
            error_data = {"detail": resp.text}

        detail = error_data.get("detail") or error_data.get("error") or resp.text
        raise RAGClientError(
            f"Upload failed ({resp.status_code}): {detail}",
            status_code=resp.status_code,
            details=error_data,
        )
