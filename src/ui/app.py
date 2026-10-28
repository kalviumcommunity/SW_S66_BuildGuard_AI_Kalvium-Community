"""BuildGuard AI Streamlit Application.

Interactive Q&A web interface for site engineers querying building codes & project specs.
Fulfills the complete specification of the BuildGuard AI Wireframe:
- Screen 1: New chat (Overview KPI cards + empty state hero with 4 site questions)
- Screen 2: Project filter (Top bar selector + left sidebar history)
- Screen 3: Answer with sources ('Source passages' card naming documents)
- Screen 4: Not found ('I couldn't find enough reliable evidence', US-03, red stop)
- Screen 5: Sources disagree ('Manual review required', Safe answers, neutral grey)
- Screen 6: Service down ('Answer service unavailable', Safe answers, red stop)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
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
    render_chat_disclaimer,
    render_edge_state_not_found,
    render_edge_state_service_down,
    render_edge_state_sources_disagree,
    render_empty_state_hero,
    render_metadata_drawer,
    render_overview_cards,
    render_project_bar,
    render_source_passages_card,
)

logger = logging.getLogger(__name__)

# Wireframe Screen 1: The 4 site engineering example questions
WIREFRAME_SAMPLE_QUESTIONS = [
    "What is the minimum fire resistance rating for basement exit staircases?",
    "Which source controls the plant room fire rating?",
    "What evidence should I record before a concrete pour?",
    "What is the maximum permissible length for a dead-end corridor?",
]

# Wireframe Project options (Screen 2)
PROJECT_OPTIONS = [
    "Metro Line Extension — Phase 3",
    "All projects (workspace overview)",
]

# Sample history sessions to populate "History (past chats)" in left menu (Page 1 & 4)
INITIAL_HISTORY_SESSIONS = [
    {
        "id": "hist_1",
        "title": "Plant room fire rating: NBC vs Project Spec",
        "project": "Metro Line Extension — Phase 3",
        "date": "14 Feb 2026",
        "messages": [
            {
                "role": "user",
                "content": "Which source controls the plant room fire rating?",
            },
            {
                "role": "assistant",
                "status": "sources_disagree",
                "content": "Manual review required",
                "sources": [],
                "provisions": [
                    {
                        "title": "NBC 2016, Part 4, Clause 6.1.1.2",
                        "text": "Walls separating a boiler room or transformer enclosure from remainder of building must provide a minimum fire resistance rating of 120 minutes (2 hours).",
                    },
                    {
                        "title": "Project Specification, Section 08-200 (Rev B)",
                        "text": "Underground metro station plant room perimeter walls adjoining public egress paths specify 180 minutes (3 hours) fire separation.",
                    },
                ],
            },
        ],
    },
    {
        "id": "hist_2",
        "title": "Station box slab clear cover requirements",
        "project": "Metro Line Extension — Phase 3",
        "date": "13 Feb 2026",
        "messages": [
            {
                "role": "user",
                "content": "What is the minimum clear cover for the station box slab?",
            },
            {
                "role": "assistant",
                "status": "success",
                "content": (
                    "Based on the indexed project corpus, the applicable requirement must be verified "
                    "against the cited passages below. BuildGuard will not provide an uncited conclusion.\n\n"
                    "Under IS 456:2000 Clause 26.4.2 and Project Specification 03-300, structural slabs in "
                    "contact with ground or severe metro environments require a minimum nominal cover of 45 mm (50 mm if cast directly against earth)."
                ),
                "sources": [
                    {
                        "source": "IS 456:2000",
                        "heading": "Clause 26.4.2 — Concrete Cover to Reinforcement",
                        "page": "47",
                        "text_preview": "Nominal cover to meet durability requirements for severe exposure conditions shall not be less than 45 mm for structural foundations and underground retaining slabs.",
                    },
                    {
                        "source": "Project Structural Specification, Rev B",
                        "heading": "Section 03-300 — Cast-in-Place Concrete",
                        "page": "18",
                        "text_preview": "Station box bottom slab and perimeter walls: 50 mm clear cover to primary longitudinal reinforcement bars.",
                    },
                ],
                "cited_docs": ["IS 456:2000", "Project Structural Specification, Rev B"],
                "metadata": {
                    "model": "configured-via-env",
                    "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
                    "candidate_k": 10,
                    "final_k": 2,
                    "rerank_method": "cross_encoder",
                    "hybrid": True,
                    "dry_run": False,
                    "source_count": 2,
                    "context_chars": 512,
                    "pipeline_elapsed_ms": 310.4,
                    "generated_at": "2026-02-13T14:20:00Z",
                },
            },
        ],
    },
    {
        "id": "hist_3",
        "title": "Material substitution without formal submittal",
        "project": "Metro Line Extension — Phase 3",
        "date": "12 Feb 2026",
        "messages": [
            {
                "role": "user",
                "content": "Can I approve a material substitution without a formal submittal?",
            },
            {
                "role": "assistant",
                "status": "not_found",
                "content": "I couldn't find enough reliable evidence",
                "sources": [],
            },
        ],
    },
]


def init_session_state() -> None:
    """Initialize Streamlit session state keys if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = PROJECT_OPTIONS[0]
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
    if "history_sessions" not in st.session_state:
        st.session_state.history_sessions = list(INITIAL_HISTORY_SESSIONS)
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    # Pipeline settings stored in session_state so they persist regardless of sidebar state
    if "pipeline_dry_run" not in st.session_state:
        st.session_state.pipeline_dry_run = False
    if "pipeline_candidate_k" not in st.session_state:
        st.session_state.pipeline_candidate_k = 10
    if "pipeline_final_k" not in st.session_state:
        st.session_state.pipeline_final_k = 3
    if "pipeline_hybrid" not in st.session_state:
        st.session_state.pipeline_hybrid = True
    if "pipeline_filter_heading" not in st.session_state:
        st.session_state.pipeline_filter_heading = ""


def get_client() -> RAGClient:
    """Return an instance of RAGClient targeting the configured backend URL."""
    return RAGClient(base_url=st.session_state.api_url)


def start_new_chat() -> None:
    """Save current chat to history if non-empty, and reset to clean empty state (Screen 1)."""
    if st.session_state.messages:
        first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
        if first_user_msg and len(first_user_msg) > 45:
            title = first_user_msg[:45] + "..."
        else:
            title = first_user_msg or "Compliance Query"
        session_id = f"hist_{len(st.session_state.history_sessions) + 1}"
        new_entry = {
            "id": session_id,
            "title": title,
            "project": st.session_state.selected_project,
            "date": datetime.now().strftime("%d %b %Y"),
            "messages": list(st.session_state.messages),
        }
        st.session_state.history_sessions.insert(0, new_entry)

    st.session_state.messages = []
    st.session_state.active_session_id = None
    st.session_state.pending_query = None


def load_history_session(session_id: str) -> None:
    """Load a past chat session from history."""
    for session in st.session_state.history_sessions:
        if session["id"] == session_id:
            st.session_state.messages = list(session["messages"])
            st.session_state.selected_project = session.get("project", PROJECT_OPTIONS[0])
            st.session_state.active_session_id = session_id
            st.session_state.pending_query = None
            return


def execute_query(question: str) -> None:
    """Send natural language query to the RAG backend.

    All pipeline settings are read from session_state so they are always
    available regardless of whether the sidebar expander is open.
    """
    clean_question = question.strip()
    if not clean_question:
        return

    st.session_state.messages.append({"role": "user", "content": clean_question})

    # Read pipeline settings from session state (always defined in init_session_state)
    dry_run: bool = st.session_state.pipeline_dry_run
    candidate_k: int = st.session_state.pipeline_candidate_k
    final_k: int = st.session_state.pipeline_final_k
    hybrid: bool = st.session_state.pipeline_hybrid
    filter_heading: str | None = st.session_state.pipeline_filter_heading.strip() or None

    client = get_client()
    with st.spinner("🔍 Grounding answer against indexed project sources & building codes..."):
        try:
            response = client.query(
                question=clean_question,
                candidate_k=candidate_k,
                final_k=final_k,
                filter_heading=filter_heading,
                hybrid=hybrid,
                dry_run=dry_run,
            )

            status = response.get("status", "success")
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            metadata = response.get("metadata", {})

            # Wireframe Rule: "No answer is ever shown without a source."
            if status == "no_context" or not sources:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "status": "not_found",
                        "content": "I couldn't find enough reliable evidence",
                        "sources": [],
                        "metadata": metadata,
                    }
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "status": "success",
                        "content": answer,
                        "sources": sources,
                        "metadata": metadata,
                    }
                )

        except (RAGConnectionError, RAGServiceUnavailableError, RAGTimeoutError) as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "service_down",
                    "content": "Answer service unavailable",
                    "sources": [],
                    "error_detail": str(exc),
                }
            )

        except RAGValidationError as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "service_down",
                    "content": "Answer service unavailable",
                    "sources": [],
                    "error_detail": f"Validation error: {exc}",
                }
            )

        except Exception as exc:
            logger.exception("Unexpected query error: %s", exc)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "status": "service_down",
                    "content": "Answer service unavailable",
                    "sources": [],
                    "error_detail": str(exc),
                }
            )


def main() -> None:
    st.set_page_config(
        page_title="BuildGuard AI — Compliance Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_custom_styles()

    client = get_client()

    # Check backend connectivity
    is_backend_online = False
    health_info: dict[str, Any] = {}
    try:
        health_info = client.health()
        if health_info and health_info.get("status") == "ok":
            is_backend_online = True
    except Exception:
        is_backend_online = False

    chat_configured: bool = health_info.get("chat_configured", False)
    missing_chat: list[str] = health_info.get("missing_chat_settings", [])

    # -------------------------------------------------------------
    # Left Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="brand-accent-line"></div>', unsafe_allow_html=True)
        st.markdown("## 🛡️ BuildGuard AI")
        st.caption("Ask a building-code question. Get a grounded answer — or a clear 'not found'.")

        st.divider()

        if st.button("➕  New chat", use_container_width=True, type="primary", key="btn_new_chat"):
            start_new_chat()
            st.rerun()

        st.divider()

        st.markdown("#### 🕒 History")
        if not st.session_state.history_sessions:
            st.caption("No past chats yet.")
        else:
            for s in st.session_state.history_sessions:
                is_active = s["id"] == st.session_state.active_session_id
                label = f"{'▶ ' if is_active else ''}{s['title']}"
                date_str = s.get("date", "")
                if st.button(
                    label,
                    key=f"hist_btn_{s['id']}",
                    use_container_width=True,
                    help=f"{s.get('project', '')} · {date_str}",
                ):
                    load_history_session(s["id"])
                    st.rerun()

        st.divider()

        # Pipeline Settings — always rendered so session_state values are always current
        with st.expander("⚙️ Pipeline Settings", expanded=False):
            new_url = st.text_input(
                "Backend URL",
                value=st.session_state.api_url,
                key="input_api_url",
                help="FastAPI server URL (e.g. http://127.0.0.1:8000)",
            )
            if new_url != st.session_state.api_url:
                st.session_state.api_url = new_url

            st.session_state.pipeline_dry_run = st.toggle(
                "Dry-Run Mode",
                value=st.session_state.pipeline_dry_run,
                key="toggle_dry_run",
                help="When ON, retrieval runs but the LLM call is skipped — no API key needed.",
            )
            st.session_state.pipeline_candidate_k = st.slider(
                "Candidate Pool (candidate_k)",
                min_value=1, max_value=30,
                value=st.session_state.pipeline_candidate_k,
                key="slider_candidate_k",
            )
            st.session_state.pipeline_final_k = st.slider(
                "Final Chunks (final_k)",
                min_value=1, max_value=10,
                value=st.session_state.pipeline_final_k,
                key="slider_final_k",
            )
            st.session_state.pipeline_hybrid = st.toggle(
                "Hybrid Search (Dense + BM25)",
                value=st.session_state.pipeline_hybrid,
                key="toggle_hybrid",
            )
            st.session_state.pipeline_filter_heading = st.text_input(
                "Section Filter (optional)",
                value=st.session_state.pipeline_filter_heading,
                key="input_filter_heading",
                placeholder="e.g. Part 4 — Fire Safety",
            )

            conn_icon = "✅" if is_backend_online else "❌"
            chat_icon = "✅" if chat_configured else "⚠️"
            dry_label = "DRY-RUN" if st.session_state.pipeline_dry_run else "LIVE LLM"
            missing_str = ", ".join(missing_chat) if missing_chat else ""
            st.caption(
                f"**Backend:** {conn_icon} {'Online' if is_backend_online else 'Offline'}  \n"
                f"**Chat Model:** {chat_icon} {'Configured' if chat_configured else 'Not configured (' + missing_str + ')' if missing_str else 'Unknown'}  \n"
                f"**Mode:** `{dry_label}`"
            )

        # Document Upload
        with st.expander("📄 Document Upload", expanded=False):
            st.caption("Index a PDF, TXT, MD, or HTML file into the active vector database.")
            uploaded_file = st.file_uploader(
                "Upload file",
                type=["pdf", "txt", "md", "html"],
                key="file_uploader",
                label_visibility="collapsed",
            )
            if uploaded_file:
                if st.button("📥  Upload & Index", use_container_width=True, key="btn_upload"):
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        try:
                            file_bytes = uploaded_file.read()
                            client.upload_document(file_bytes, uploaded_file.name)
                            st.success(f"✅ Indexed **{uploaded_file.name}** successfully!")
                        except Exception as exc:
                            st.error(f"Upload failed: {exc}")

        # Demo Screens
        with st.expander("📐 Demo Screens", expanded=False):
            st.caption("Load pre-built example conversations:")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("1. New chat", use_container_width=True, key="demo_1"):
                    start_new_chat()
                    st.rerun()
                if st.button("3. Sourced ans.", use_container_width=True, key="demo_3"):
                    load_history_session("hist_2")
                    st.rerun()
                if st.button("5. Disagree", use_container_width=True, key="demo_5"):
                    load_history_session("hist_1")
                    st.rerun()
            with col_b:
                if st.button("2. Filter demo", use_container_width=True, key="demo_2"):
                    st.session_state.selected_project = PROJECT_OPTIONS[1]
                    st.rerun()
                if st.button("4. Not found", use_container_width=True, key="demo_4"):
                    load_history_session("hist_3")
                    st.rerun()
                if st.button("6. Down state", use_container_width=True, key="demo_6"):
                    st.session_state.messages = [
                        {"role": "user", "content": "Verify transformer enclosure wall thickness"},
                        {
                            "role": "assistant",
                            "status": "service_down",
                            "content": "Answer service unavailable",
                            "sources": [],
                        },
                    ]
                    st.rerun()

    # -------------------------------------------------------------
    # Main Content Area
    # -------------------------------------------------------------

    # Top bar: Project filter + backend status
    col_proj, col_status = st.columns([3, 1])
    with col_proj:
        selected_proj = st.selectbox(
            "Active Project",
            options=PROJECT_OPTIONS,
            index=PROJECT_OPTIONS.index(st.session_state.selected_project)
            if st.session_state.selected_project in PROJECT_OPTIONS
            else 0,
            key="project_selectbox",
            label_visibility="collapsed",
        )
        st.session_state.selected_project = selected_proj

    with col_status:
        dot_color = "#10b981" if is_backend_online else "#ef4444"
        status_text = "API Online" if is_backend_online else "API Offline"
        st.markdown(
            f'<div style="text-align: right; font-size: 0.8125rem; color: #94a3b8; padding-top: 0.55rem; font-weight: 500;">'
            f'<span style="color: {dot_color}; font-size: 0.95rem; margin-right: 0.25rem;">●</span> {status_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-bottom: 0.85rem;"></div>', unsafe_allow_html=True)

    # Show warning if live mode is selected but chat model isn't configured
    if not st.session_state.pipeline_dry_run and not chat_configured and is_backend_online:
        st.warning(
            f"⚠️ **Live LLM mode is active but the chat model is not configured.** "
            f"Missing: `{', '.join(missing_chat)}`. "
            f"Set these in your `.env` file or enable **Dry-Run Mode** in Pipeline Settings.",
            icon="⚠️",
        )

    # Overview KPI Cards
    is_metro = "Metro" in st.session_state.selected_project
    render_overview_cards(
        indexed_sources=6 if is_metro else 18,
        compliance_reviews=4 if is_metro else 14,
        review_required=2 if is_metro else 5,
        refused_answers=1 if is_metro else 3,
        standards_text="NBC 2016 + IS 456:2000" if is_metro else "All 12 Active Project Specs",
    )

    # -------------------------------------------------------------
    # Screen 1: Empty State Hero OR Conversation Feed
    # -----------------------------------------------------------
    if not st.session_state.messages:
        render_empty_state_hero()

        # 4 Wireframe example questions in a 2x2 grid
        q_col1, q_col2 = st.columns(2)
        for idx, question_text in enumerate(WIREFRAME_SAMPLE_QUESTIONS):
            target_col = q_col1 if idx % 2 == 0 else q_col2
            with target_col:
                if st.button(question_text, key=f"empty_q_{idx}", use_container_width=True):
                    st.session_state.pending_query = question_text
                    st.rerun()

    else:
        # Render Conversation Feed
        for msg_idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**{msg['content']}**")
            else:
                with st.chat_message("assistant", avatar="🛡️"):
                    status = msg.get("status", "success")

                    # Screen 4: Not found (Edge state: US-03, red stop)
                    if status == "not_found":
                        reset_clicked = render_edge_state_not_found(return_button_key=f"reset_not_found_{msg_idx}")
                        if reset_clicked:
                            start_new_chat()
                            st.rerun()

                    # Screen 5: Sources disagree (Edge state: Safe answers, neutral grey)
                    elif status == "sources_disagree":
                        reset_clicked = render_edge_state_sources_disagree(
                            provisions=msg.get("provisions"),
                            return_button_key=f"reset_disagree_{msg_idx}",
                        )
                        if reset_clicked:
                            start_new_chat()
                            st.rerun()

                    # Screen 6: Service down (Edge state: Safe answers, red stop)
                    elif status == "service_down":
                        error_detail = msg.get("error_detail", "")
                        reset_clicked = render_edge_state_service_down(
                            return_button_key=f"reset_down_{msg_idx}",
                            error_detail=error_detail,
                        )
                        if reset_clicked:
                            start_new_chat()
                            st.rerun()

                    # Screen 3: Answer with sources
                    else:
                        st.markdown(msg["content"])
                        sources = msg.get("sources", [])
                        cited_docs = msg.get("cited_docs")
                        if sources:
                            render_source_passages_card(sources, cited_docs=cited_docs)
                        if msg.get("metadata"):
                            render_metadata_drawer(msg["metadata"])

    # -------------------------------------------------------------
    # Chat Input
    # -------------------------------------------------------------
    chat_prompt = st.chat_input("Ask BuildGuard a compliance question...")

    prompt_to_run = None
    if st.session_state.pending_query:
        prompt_to_run = st.session_state.pending_query
        st.session_state.pending_query = None
    elif chat_prompt:
        prompt_to_run = chat_prompt

    if prompt_to_run:
        execute_query(prompt_to_run)
        st.rerun()

    # Footer note from wireframe
    render_chat_disclaimer()


if __name__ == "__main__":
    main()
