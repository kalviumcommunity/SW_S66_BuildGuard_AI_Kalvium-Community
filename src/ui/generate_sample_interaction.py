"""Generate sample interaction data demonstrating the Query UI flow.

Simulates and records user interactions with the BuildGuard RAG UI, including:
1. Question submission & grounded answer reception (Tasks 1 & 2)
2. Retrieved sources inspection with chunk IDs, scores, and metadata (Task 3)
3. Loading and error states handling (Task 4)

Outputs the resulting interaction log to outputs/ui/sample_interaction.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ui.client import RAGClient, RAGValidationError, RAGServiceUnavailableError

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ui"
OUTPUT_FILE = OUTPUT_DIR / "sample_interaction.json"


def generate_sample_interaction() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = RAGClient(base_url="http://127.0.0.1:8000")

    # 1. Health check interaction
    health = client.health()

    # 2. Successful query interaction with grounded answer and retrieved sources
    question = "What is the minimum fire resistance rating required for walls separating a boiler room?"
    query_resp = client.query(
        question=question,
        candidate_k=5,
        final_k=2,
        hybrid=True,
        dry_run=True,
    )

    # 3. Validation error simulation (Empty/Blank query -> 422)
    validation_error_record = None
    try:
        client.query(question="   ")
    except RAGValidationError as err:
        validation_error_record = {
            "trigger_input": "   ",
            "http_status": err.status_code,
            "error_message": err.message,
            "error_details": err.details,
            "ui_behavior": "Displays warning banner prompting the user to provide a non-empty question; 422 error mapped cleanly without crashing.",
        }

    # 4. Service unavailable error simulation (Live query requested without LLM credentials -> 503)
    service_unavailable_record = None
    try:
        client.query(question=question, dry_run=False)
    except RAGServiceUnavailableError as err:
        service_unavailable_record = {
            "trigger_input": question,
            "dry_run": False,
            "http_status": err.status_code,
            "error_message": err.message,
            "error_details": err.details,
            "ui_behavior": "Displays friendly 503 error banner explaining missing LLM keys, with a tip to enable 'Dry-Run Mode' in the sidebar.",
        }

    # 5. Compile sample interaction document
    sample_interaction = {
        "title": "BuildGuard AI — UI Sample Interaction Flow",
        "timestamp": "2026-10-27T09:10:00Z",
        "backend": {
            "base_url": client.base_url,
            "health": health,
        },
        "task_1_2_3_query_and_grounded_sources": {
            "user_prompt": question,
            "retrieval_parameters": {
                "candidate_k": 5,
                "final_k": 2,
                "hybrid": True,
                "dry_run": True,
            },
            "ui_state": "idle -> loading (st.spinner) -> success",
            "grounded_answer": query_resp.get("answer"),
            "status": query_resp.get("status"),
            "retrieved_sources": query_resp.get("sources", []),
            "pipeline_metadata": query_resp.get("metadata", {}),
        },
        "task_4_loading_and_error_handling": {
            "loading_state": {
                "indicator": "st.spinner",
                "message": "🔍 Querying National Building Code knowledge base & verifying grounding...",
                "behavior": "Visual loading spinner remains active while HTTP request is in-flight, preventing duplicate submissions.",
            },
            "error_state_validation_422": validation_error_record,
            "error_state_unconfigured_503": service_unavailable_record,
            "error_state_connection_offline": {
                "trigger_condition": "FastAPI backend unreachable or stopped",
                "ui_behavior": "Displays red alert banner with exact target URL and CLI command to start backend (`uv run uvicorn src.api.app:app`).",
            },
        },
    }

    OUTPUT_FILE.write_text(json.dumps(sample_interaction, indent=2))
    print(f"Sample interaction successfully written to {OUTPUT_FILE}")
    return sample_interaction


if __name__ == "__main__":
    generate_sample_interaction()
