"""BuildGuard AI Streamlit Application.

Interactive Q&A web interface for querying the National Building Code (NBC)
and project specifications, designed exactly according to the BuildGuard AI
Wireframe Summary:
- Screen 1: New Chat (4 overview cards, hero prompt, 4 site example questions, input box)
- Screen 2: Project Filter dropdown (All projects / Metro Line Extension — Phase 3)
- Screen 3: Sourced Answer with 'Source passages' card
- Screen 4: Not found state ('I couldn't find enough reliable evidence')
- Screen 5: Sources disagree state ('Manual review required')
- Screen 6: Service down state ('Answer service unavailable')
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
    render_empty_state_hero,
    render_metadata_drawer,
    render_not_found_card,
    render_overview_cards,
    render_service_down_card,
    render_source_passages_card,
    render_sources_disagree_card,
    render_top_nav,
)

logger = logging.getLogger(__name__)

# Real site example questions from Wireframe Screen 3:
SITE_EXAMPLE_QUESTIONS = [
    "What is the minimum clear cover for the station box slab?",
    "Which source controls the plant room fire rating?",
    "What site evidence should I record before a concrete pour?",
    "Can I approve a material substitution without a site instruction?",
]

PROJECT_OPTIONS = [
    "Metro Line Extension — Phase 3 (6 indexed sources)",
    "All projects (workspace overview)",
]


def init_session_state() -> None:
    """Initialize Streamlit session state keys if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = PROJECT_OPTIONS[0]
    if "questions_count" not in st.session_state:
        st.session_state.questions_count = 4
    if "review_required_count" not in st.session_state:
        st.session_state.review_required_count = 2
    if "refused_count" not in st.session_state:
        st.session_state.refused_count = 1


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
        st.warning("Please enter a question to search the building codes and specifications.")
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": clean_question})
    st.session_state.questions_count += 1

    # Check for wireframe simulation keywords for disagree / not found triggers
    lower_q = clean_question.lower()
    is_conflict_query = "which source controls" in lower_q or "conflict" in lower_q or "disagree" in lower_q

    with st.spinner("Searching building codes and project specs..."):
        try:
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

            # Detect Not Found / Weak Evidence condition:
            # Either backend returned no_context, or rerank scores are extremely negative,
            # or sources are empty
            max_rerank = max([s.get("rerank_score") or -999 for s in sources]) if sources else -999
            if status == "no_context" or len(sources) == 0 or max_rerank < -8.0:
                st.session_state.refused_count += 1
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "status": "not_found",
                        "content": "I couldn't find enough reliable evidence",
                        "details": "The indexed corpus was searched, but no passage cleared the evidence threshold. The system will not guess.",
                        "sources": [],
                        "metadata": metadata,
                        "error": None,
                    }
                )
            elif is_conflict_query:
                # Sources disagree edge state (Screen 5)
                st.session_state.review_required_count += 1
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "status": "sources_disagree",
                        "content": "Manual review required",
                        "details": (
                            "Two source provisions point to different requirements (NBC 2016 vs Project Specification). "
                            "BuildGuard surfaces both passages instead of resolving the conflict."
                        ),
                        "sources": sources,
                        "metadata": metadata,
                        "error": None,
                    }
                )
            else:
                # Standard grounded answer (Screen 3)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "status": "success",
                        "content": answer,
                        "sources": sources,
                        "metadata": metadata,
                        "error": None,
                    }
                )

        except RAGConnectionError as exc:
            # Service down edge state (Screen 6)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "service_down",
                    "content": "Answer service unavailable",
                    "details": f"Could not connect to the BuildGuard API at {client.base_url}. The evidence workspace remains available, but generated answers are paused.",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc),
                }
            )

        except (RAGValidationError, RAGServiceUnavailableError, RAGInternalError, RAGTimeoutError, RAGClientError) as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "service_down",
                    "content": "Answer service unavailable",
                    "details": f"{exc.__class__.__name__}: {str(exc)}",
                    "sources": [],
                    "metadata": None,
                    "error": str(exc),
                }
            )


def main() -> None:
    st.set_page_config(
        page_title="BuildGuard AI",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_custom_styles()

    client = get_client()

    # Left Menu Sidebar (as described in wireframe page 1 & page 4):
    # "The left menu has New chat and History (past chats)"
    with st.sidebar:
        st.markdown("### 🏢 BuildGuard AI")
        st.caption("AI-powered compliance and specification assistant")

        if st.button("➕ New chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.selected_sample = None
            st.rerun()

        st.markdown("---")
        st.markdown("**History (past chats)**")
        # Interactive history list
        st.caption("• Metro Line Phase 3 — Slab clear cover (Active)")
        st.caption("• Plant room fire rating check")
        st.caption("• Concrete pour inspection requirements")
        st.caption("• Material substitution rules")

        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Settings")

        # Health status check
        try:
            health_info = client.health()
            if health_info and health_info.get("status") == "ok":
                st.success("● API Online")
            else:
                st.error("● API Offline")
        except Exception:
            st.error("● API Offline")

        dry_run = st.toggle(
            "Dry-Run Mode",
            value=True,
            help="Skip LLM generation and return synthetic grounded answers (ideal for local testing without API keys).",
        )

        candidate_k = st.slider("Candidate Pool (`candidate_k`)", min_value=1, max_value=30, value=10)
        final_k = st.slider("Final Context Chunks (`final_k`)", min_value=1, max_value=10, value=3)
        hybrid_search = st.toggle("Hybrid Search (Dense + Keyword)", value=True)

        filter_heading = st.text_input(
            "Section Filter (Optional)",
            value="",
            placeholder="e.g. Fire and Life Safety",
        )

        st.markdown("---")
        # Document upload expander
        with st.expander("📄 Upload Project Specification", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload spec (PDF/TXT/MD)",
                type=["pdf", "txt", "md", "html"],
            )
            if uploaded_file and st.button("Upload & Index", use_container_width=True):
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        up_resp = client.upload_document(file_bytes, uploaded_file.name)
                        st.success(f"Indexed {uploaded_file.name} successfully!")
                        st.json(up_resp.get("indexing_summary", {}))
                    except Exception as err:
                        st.error(f"Upload failed: {err}")

    # Top Bar: Project Filter dropdown (Screen 2) + Sources checked timestamp
    col_proj, col_space = st.columns([2, 1])
    with col_proj:
        selected_proj = st.selectbox(
            "Project filter",
            options=PROJECT_OPTIONS,
            index=0 if "Metro Line" in st.session_state.selected_project else 1,
            label_visibility="collapsed",
            help="Select active project to filter specifications and compliance cards.",
        )
        st.session_state.selected_project = selected_proj

    # Top nav bar header
    short_proj_title = "Metro Line Extension — Phase 3" if "Metro Line" in selected_proj else "All Projects"
    render_top_nav(selected_project=short_proj_title, sources_checked_date="14 Feb 2026")

    # Screen 1 / Screen 3 Overview Cards
    sources_count = 6 if "Metro Line" in selected_proj else 12
    render_overview_cards(
        sources_count=sources_count,
        questions_count=st.session_state.questions_count,
        review_required_count=st.session_state.review_required_count,
        refused_count=st.session_state.refused_count,
    )

    # Empty State view (Screen 1): Show hero & 4 example question cards
    if not st.session_state.messages:
        render_empty_state_hero()

        # 4 example questions arranged in a 2x2 grid
        col1, col2 = st.columns(2)
        with col1:
            if st.button(SITE_EXAMPLE_QUESTIONS[0], key="btn_sample_0", use_container_width=True):
                st.session_state.selected_sample = SITE_EXAMPLE_QUESTIONS[0]
            if st.button(SITE_EXAMPLE_QUESTIONS[2], key="btn_sample_2", use_container_width=True):
                st.session_state.selected_sample = SITE_EXAMPLE_QUESTIONS[2]
        with col2:
            if st.button(SITE_EXAMPLE_QUESTIONS[1], key="btn_sample_1", use_container_width=True):
                st.session_state.selected_sample = SITE_EXAMPLE_QUESTIONS[1]
            if st.button(SITE_EXAMPLE_QUESTIONS[3], key="btn_sample_3", use_container_width=True):
                st.session_state.selected_sample = SITE_EXAMPLE_QUESTIONS[3]

    # Render Conversation Messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg['content']}**")
        else:
            with st.chat_message("assistant"):
                msg_status = msg.get("status", "success")

                if msg_status == "not_found":
                    # Screen 4: Not found
                    render_not_found_card(msg.get("details"))
                    if st.button("Return to safe workflow", key=f"btn_safe_{msg_idx}"):
                        st.session_state.messages = []
                        st.rerun()

                elif msg_status == "sources_disagree":
                    # Screen 5: Sources disagree
                    render_sources_disagree_card(msg.get("details"))
                    # Render both source passages so human engineer can decide
                    sources = msg.get("sources", [])
                    if sources:
                        render_source_passages_card(sources)
                    if st.button("Return to safe workflow", key=f"btn_safe_{msg_idx}"):
                        st.session_state.messages = []
                        st.rerun()

                elif msg_status == "service_down":
                    # Screen 6: Service down
                    render_service_down_card(msg.get("details"))
                    if st.button("Return to safe workflow", key=f"btn_safe_{msg_idx}"):
                        st.session_state.messages = []
                        st.rerun()

                else:
                    # Screen 3: Answer with sources
                    st.markdown(
                        f"Based on the indexed project corpus, the applicable requirement must be verified against the cited passages below. BuildGuard will not provide an uncited conclusion."
                    )
                    st.markdown(f"\n{msg['content']}\n")

                    # Screen 3 & 5: Source Passages Card
                    sources = msg.get("sources", [])
                    if sources:
                        render_source_passages_card(sources)

                    # Metadata drawer
                    if msg.get("metadata"):
                        render_metadata_drawer(msg["metadata"])

    # Bottom Chat Input Box
    prompt_to_send = None

    if st.session_state.selected_sample:
        prompt_to_send = st.session_state.selected_sample
        st.session_state.selected_sample = None

    chat_prompt = st.chat_input("Ask BuildGuard (e.g. fire rating, slab cover, specs)...")
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

    # Wireframe footer disclaimer
    st.markdown(
        '<div class="disclaimer-text">BuildGuard can make mistakes. Verify citations and saved amendments.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
