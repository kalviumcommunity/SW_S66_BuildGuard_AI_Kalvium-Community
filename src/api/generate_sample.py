"""Generate a committed sample request/response for the RAG backend API.

Drives the real FastAPI ``POST /query`` endpoint through a TestClient in
dry-run mode — so it runs the genuine embed → retrieve → rerank → assemble
stages against the local Chroma index and returns real citations, while
skipping only the LLM call (no API key required).

Usage
-----
    uv run python -m src.api.generate_sample
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs" / "rag_api"

SAMPLE_REQUEST = {
    "question": (
        "What is the minimum fire resistance rating required for walls "
        "separating a boiler room from other parts of a building?"
    ),
    "candidate_k": 10,
    "final_k": 3,
    "dry_run": True,
}


def main() -> int:
    client = TestClient(app)

    response = client.post("/query", json=SAMPLE_REQUEST)
    status_code = response.status_code
    body = response.json()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    request_doc = {
        "endpoint": "POST /query",
        "headers": {"Content-Type": "application/json"},
        "body": SAMPLE_REQUEST,
    }
    (OUTPUT_DIR / "sample_request.json").write_text(
        json.dumps(request_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    response_doc = {"http_status": status_code, "body": body}
    (OUTPUT_DIR / "sample_response.json").write_text(
        json.dumps(response_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"HTTP {status_code}")
    print(f"status field: {body.get('status')}  sources: {len(body.get('sources', []))}")
    print(f"Wrote -> {OUTPUT_DIR / 'sample_request.json'}")
    print(f"Wrote -> {OUTPUT_DIR / 'sample_response.json'}")
    return 0 if status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
