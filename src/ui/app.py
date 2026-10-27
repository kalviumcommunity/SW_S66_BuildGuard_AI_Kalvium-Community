"""BuildGuard AI Streamlit Application.

Interactive Q&A web interface for querying the National Building Code (NBC)
via the BuildGuard RAG API. Displays grounded answers, structured retrieved sources,
pipeline metadata, and robust loading/error handling.
"""

from __future__ import annotations

import logging
import os
import streamlit as st

from src.ui.client import (
    RAGClient,
    RAGClientError,
    RAGConnectionError,
    RAGInternalError,
    RAGServiceUnavailableError,
    RAGTimeoutError,
    RAGValidationError,
)
from src.ui.components import (
    apply_custom_styles,
    render_error_banner,
    render_header,
    render_metadata_drawer,
    render_sources_section,
)

logger = logging.getLogger(__name__)

# Sample prompt buttons to help users test immediately
SAMPLE_QUESTIONS = [
    "What is the minimum fire resistance rating required for walls separating a boiler room?",
    "What are the requirements for exit doorway widths in educational buildings?",
    "What are the ventilation and light requirements for basements in buildings?",
]


def init_session_state() -> None:
    """Initialize Streamlit session state keys if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None


def get_client() -> RAGClient:
    """Return an instance of RAGClient targeting the configured backend URL."""
    return RAGClient(base_url=st.session_state.api_url)


def run_query(
    question: str,
    candidate_k: int,
    final_k: int,
    filter_heading: str | None,
    hybrid: bool,
    dry_run: bool,
) -> None:
    """Send query to backend, display loading indicator, and record response or error."""
    client = get_client()
    clean_question = question.strip()

    if not clean_question:
        st.warning("Please enter a question to search the National Building Code.")
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": clean_question})

    # Task 4: Clear loading state while request is in flight
    with st.spinner("🔍 Querying National Building Code knowledge base & verifying grounding..."):
        try:
            # Task 2: Call the backend RAG API
            response = client.query(
                question=clean_question,
                candidate_k=candidate_k,
                final_k=final_k,
                filter_heading=filter_heading if filter_heading else None,
                hybrid=hybrid,
                dry_run=dry_run,
            )

            status = response.get("status", "success")
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            metadata = response.get("metadata", {})

            # Task 1, 2, 3: Store assistant response with answer, sources, metadata
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": status,
                    "content": answer,
                    "sources": sources,
                    "metadata": metadata,
                    "error": None,
                }
            )

        except RAGConnectionError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": (
                        "⚠️ **Connection Refused**: Could not reach the BuildGuard RAG API.\n\n"
                        f"Target URL: `{client.base_url}`\n\n"
                        "Please verify the backend server is running. You can start it with:\n"
                        "```bash\n"
                        "uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000\n"
                        "```"
                    ),
                    "sources": [],
                    "metadata": None,
                    "error": str(exc),
                }
            )

        except RAGValidationError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": f"⚠️ **Invalid Query Parameter (422)**: {exc.message}",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc.details),
                }
            )

        except RAGServiceUnavailableError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": (
                        f"⚠️ **Service Unavailable (503)**: {exc.message}\n\n"
                        "💡 *Tip:* Enable **'Dry-Run Mode'** in the sidebar to test retrieval and "
                        "grounding without requiring live OpenAI/Groq API keys."
                    ),
                    "sources": [],
                    "metadata": None,
                    "error": str(exc.details),
                }
            )

        except RAGInternalError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": f"⚠️ **Internal Server Error (500)**: {exc.message}",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc.details),
                }
            )

        except RAGTimeoutError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": f"⏱️ **Request Timed Out**: The backend pipeline took too long to respond ({exc.message}).",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc),
                }
            )

        except RAGClientError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "error",
                    "content": f"⚠️ **API Error**: {exc.message}",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc.details),
                }
            )


def main() -> None:
    st.set_page_config(
        page_title="BuildGuard AI — NBC Compliance & Safety Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_custom_styles()

    client = get_client()

    # Sidebar: configuration & controls
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")

        st.session_state.api_url = st.text_input(
            "Backend API URL",
            value=st.session_state.api_url,
            help="HTTP address where the FastAPI RAG service is running.",
        )

        # Health status check
        health_info = None
        health_err = None
        try:
            health_info = client.health()
        except Exception as exc:
            health_err = str(exc)

        if health_info and health_info.get("status") == "ok":
            st.success("✅ Backend Online")
            with st.expander("ℹ️ Backend Status", expanded=False):
                st.write(f"**Embedding Model:** `{health_info.get('embedding_model')}`")
                st.write(f"**Chroma Collection:** `{health_info.get('collection_name')}`")
                st.write(f"**Chat Configured:** `{'Yes' if health_info.get('chat_configured') else 'No'}`")
                if health_info.get("missing_chat_settings"):
                    st.caption(f"Missing keys: {', '.join(health_info.get('missing_chat_settings'))}")
        else:
            st.error("❌ Backend Offline")
            st.caption("FastAPI server is not responding at this URL.")

        st.markdown("---")
        st.markdown("### 🎛️ Retrieval Parameters")

        dry_run = st.toggle(
            "Dry-Run Mode",
            value=True,
            help="Skip LLM generation and return synthetic grounded answers (ideal for local testing without API keys).",
        )

        candidate_k = st.slider(
            "Candidate Pool (`candidate_k`)",
            min_value=1,
            max_value=30,
            value=10,
            help="Number of initial vector chunks to retrieve before re-ranking.",
        )

        final_k = st.slider(
            "Final Context Chunks (`final_k`)",
            min_value=1,
            max_value=10,
            value=3,
            help="Top-ranked chunks passed to the LLM for grounded answer generation.",
        )

        hybrid_search = st.toggle(
            "Hybrid Search (Dense + Keyword)",
            value=True,
            help="Boost retrieval accuracy by combining vector embeddings with BM25 lexical keyword matching.",
        )

        filter_heading = st.text_input(
            "Section Filter (Optional)",
            value="",
            placeholder="e.g. Fire and Life Safety",
            help="Scope retrieval strictly to chunks matching this section heading.",
        )

        st.markdown("---")

        # Document upload expander (POST /documents/upload)
        with st.expander("📄 Upload Document to Knowledge Base", expanded=False):
            uploaded_file = st.file_uploader(
                "Add PDF/TXT/MD to vector DB",
                type=["pdf", "txt", "md", "html"],
                help="Uploaded documents are chunked, embedded, and indexed immediately.",
            )
            if uploaded_file and st.button("Upload & Index", use_container_width=True):
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        up_resp = client.upload_document(file_bytes, uploaded_file.name)
                        st.success(f"Successfully indexed {uploaded_file.name}!")
                        st.json(up_resp.get("indexing_summary", {}))
                    except Exception as err:
                        st.error(f"Upload failed: {err}")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main view Header
    render_header(health_info, health_err)

    # Sample query quick pills
    st.markdown("**Quick Test Questions:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, sample_q in enumerate(SAMPLE_QUESTIONS):
        with cols[idx]:
            if st.button(sample_q, key=f"sample_q_{idx}", use_container_width=True):
                st.session_state.selected_sample = sample_q

    st.markdown("---")

    # Render Conversation Feed (Task 1 & Task 3)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                # Assistant grounded response
                if msg.get("status") == "error":
                    st.markdown(msg["content"])
                    if msg.get("error"):
                        with st.expander("Diagnostic Details", expanded=False):
                            st.code(msg["error"], language="text")
                else:
                    if msg.get("status") == "no_context":
                        st.warning("⚠️ No relevant provisions were found in the National Building Code for this query.")

                    # Task 1: Display area for grounded answers
                    st.markdown(msg["content"])

                    # Task 3: Display retrieved sources alongside the answer
                    sources = msg.get("sources", [])
                    if sources:
                        render_sources_section(sources)

                    # Metadata drawer
                    if msg.get("metadata"):
                        render_metadata_drawer(msg["metadata"])

    # Query Input (Task 1)
    prompt_to_send = None

    # Handle quick sample question click
    if st.session_state.selected_sample:
        prompt_to_send = st.session_state.selected_sample
        st.session_state.selected_sample = None

    # Chat input bar
    chat_prompt = st.chat_input("Ask a question about the National Building Code (NBC)...")
    if chat_prompt:
        prompt_to_send = chat_prompt

    if prompt_to_send:
        run_query(
            question=prompt_to_send,
            candidate_k=candidate_k,
            final_k=final_k,
            filter_heading=filter_heading.strip() if filter_heading else None,
            hybrid=hybrid_search,
            dry_run=dry_run,
        )
        st.rerun()


if __name__ == "__main__":
    main()
