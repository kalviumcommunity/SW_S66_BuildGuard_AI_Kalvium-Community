"""Demonstrate runtime upload + searchability and save the committed sample.

Drives the SAME in-process FastAPI app for two calls, with no restart between
them, against a throwaway Chroma directory so the project's real corpus stays
untouched:

  1. POST /documents/upload  — ingest, embed, and index a new document.
  2. POST /query             — ask a question only the new document answers.

If the follow-up query cites the just-uploaded chunk, runtime searchability is
proven. Artifacts are written to ``outputs/rag_api/upload/``.

Usage
-----
    uv run python -m src.api.generate_upload_sample
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.config import get_config

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs" / "rag_api" / "upload"

DEMO_DOCUMENT_NAME = "buildguard_addendum_2026.txt"
DEMO_DOCUMENT_TEXT = (
    "BuildGuard Internal Addendum 2026. Section A12 - Rooftop Helipads. "
    "On Type-2 residential buildings, a rooftop helipad must support a minimum "
    "static load of 1500 kilograms per square meter and include a 3-metre "
    "unobstructed safety perimeter around the landing circle. These requirements "
    "supplement the National Building Code fire and life safety provisions."
)
DEMO_QUERY = (
    "What is the minimum static load a rooftop helipad must support on a "
    "Type-2 residential building?"
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bg_upload_demo_"))

    # Point config at an isolated vector DB + upload dir for the demo run.
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp / "vector_db")
    os.environ["CHROMA_COLLECTION"] = "buildguard_upload_demo"
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["RAG_DRY_RUN"] = "true"
    config = get_config(reload=True)

    client = TestClient(app)

    # 1) Upload + index the new document.
    upload_request = {
        "endpoint": "POST /documents/upload",
        "multipart_fields": {
            "file": {
                "filename": DEMO_DOCUMENT_NAME,
                "content_type": "text/plain",
                "body": DEMO_DOCUMENT_TEXT,
            }
        },
    }
    upload_response = client.post(
        "/documents/upload",
        files={"file": (DEMO_DOCUMENT_NAME, DEMO_DOCUMENT_TEXT.encode("utf-8"), "text/plain")},
    )

    # 2) Query the running app (no restart) for content only the new doc has.
    query_request = {"endpoint": "POST /query", "body": {"question": DEMO_QUERY, "dry_run": True}}
    query_response = client.post("/query", json={"question": DEMO_QUERY, "dry_run": True})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump(name: str, payload) -> None:
        (OUTPUT_DIR / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    dump("upload_request.json", upload_request)
    dump("upload_indexing_summary.json", {
        "http_status": upload_response.status_code,
        "body": upload_response.json(),
    })
    dump("followup_query_request.json", query_request)
    dump("followup_query_result.json", {
        "http_status": query_response.status_code,
        "body": query_response.json(),
    })

    qbody = query_response.json()
    sources = qbody.get("sources", []) if query_response.status_code == 200 else []
    cited_uploaded = any(
        src.get("id", "").startswith("upload-") or DEMO_DOCUMENT_NAME in str(src.get("source", ""))
        for src in sources
    )

    combined = {
        "upload": upload_request,
        "upload_summary": upload_response.json() if upload_response.status_code == 201 else upload_response.json(),
        "followup_query": query_request,
        "followup_result": qbody,
        "cited_uploaded_document": cited_uploaded,
        "collection": config.collection_name,
        "persist_dir": str(config.persist_dir),
    }
    dump("sample_upload_run.json", combined)

    print(f"[1] POST /documents/upload  -> HTTP {upload_response.status_code}")
    if upload_response.status_code == 201:
        s = upload_response.json()["indexing"]
        print(f"      indexed {s['chunks_indexed']} chunk(s) into collection '{s['collection']}'")
    print(f"[2] POST /query             -> HTTP {query_response.status_code}, sources={len(sources)}")
    print(f"      uploaded document cited at query time: {cited_uploaded}")
    print(f"      artifacts written to: {OUTPUT_DIR}")
    return 0 if (upload_response.status_code == 201 and cited_uploaded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
