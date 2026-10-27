"""UI components and visual formatting for the BuildGuard Streamlit application.

Handles custom CSS styling, source cards, operational metadata widgets,
and styled error banners for clear UX across all states.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


CUSTOM_CSS = """
<style>
/* Main container styling */
.main-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}

.brand-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0ea5e9;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.brand-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 0.8rem;
}

/* Badge styling */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

.badge-connected {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.badge-disconnected {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.badge-source {
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

.badge-page {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
}

.badge-heading {
    background-color: #f3e8ff;
    color: #7e22ce;
    border: 1px solid #e9d5ff;
}

.badge-score {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

.badge-chunk {
    font-family: monospace;
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

/* Source Card Container */
.source-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: rgba(255, 255, 255, 0.03);
    transition: all 0.2s ease-in-out;
}

.source-card:hover {
    border-color: #0ea5e9;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.source-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-snippet {
    font-size: 0.85rem;
    line-height: 1.45;
    color: #4b5563;
    background: rgba(0, 0, 0, 0.02);
    border-left: 3px solid #0ea5e9;
    padding: 0.5rem 0.75rem;
    margin-top: 0.5rem;
    border-radius: 0 0.25rem 0.25rem 0;
    font-family: inherit;
}

/* Dark mode adjustment for text */
@media (prefers-color-scheme: dark) {
    .source-snippet {
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.04);
    }
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(health_status: dict[str, Any] | None, error_message: str | None = None) -> None:
    """Render the top banner with application title and connection status."""
    st.markdown(
        """
        <div class="main-header">
            <div class="brand-title">🛡️ BuildGuard AI — NBC Compliance Assistant</div>
            <div class="brand-subtitle">
                Grounded Question Answering & Source Verification for the National Building Code of India
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if health_status and health_status.get("status") == "ok":
            chat_status = "Live LLM Ready" if health_status.get("chat_configured") else "Dry-Run Ready (No LLM Key)"
            st.markdown(
                f'<div style="text-align: right;"><span class="badge badge-connected">● API Online</span> '
                f'<span class="badge badge-source">{chat_status}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: right;"><span class="badge badge-disconnected">● API Offline</span></div>',
                unsafe_allow_html=True,
            )


def render_source_card(source: dict[str, Any], index: int) -> None:
    """Render a single retrieved source card with chunk metadata and snippet."""
    rank = source.get("rank", index + 1)
    chunk_id = source.get("id", "unknown")
    source_file = source.get("source", "NBC Document")
    vol = source.get("volume", "")
    page = source.get("page", "?")
    heading = source.get("heading") or "General Provision"
    cosine = source.get("cosine_similarity")
    rerank_score = source.get("rerank_score")
    snippet = source.get("text_preview", "")

    # Format badges
    vol_page_str = f"Vol {vol}, Page {page}" if vol else f"Page {page}"
    score_str = f"Score: {rerank_score:.2f}" if rerank_score is not None else (
        f"Sim: {cosine:.2f}" if cosine is not None else ""
    )

    with st.container():
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">
                    <span><strong>Source #{rank}:</strong> {source_file}</span>
                    <span>{f'<span class="badge badge-score">{score_str}</span>' if score_str else ''}</span>
                </div>
                <div>
                    <span class="badge badge-source">{source_file}</span>
                    <span class="badge badge-page">📄 {vol_page_str}</span>
                    <span class="badge badge-heading">🔖 {heading}</span>
                    <span class="badge badge-chunk">ID: {chunk_id}</span>
                </div>
                <div class="source-snippet">
                    "{snippet}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources_section(sources: list[dict[str, Any]]) -> None:
    """Render the list of retrieved sources inside an expander or section."""
    if not sources:
        st.info("No sources were cited for this response.")
        return

    st.markdown(f"#### 📚 Retrieved Grounding Sources ({len(sources)} cited)")
    for i, source in enumerate(sources):
        render_source_card(source, i)


def render_metadata_drawer(metadata: dict[str, Any] | None) -> None:
    """Render pipeline operational metrics in a compact accordion."""
    if not metadata:
        return

    with st.expander("⚙️ Pipeline Execution Metadata", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Latency", f"{metadata.get('pipeline_elapsed_ms', 0):.1f} ms")
        with col2:
            st.metric("Retrieved Sources", str(metadata.get("source_count", 0)))
        with col3:
            st.metric("Context Size", f"{metadata.get('context_chars', 0)} chars")
        with col4:
            mode = "Dry-Run" if metadata.get("dry_run") else "Live LLM"
            st.metric("Execution Mode", mode)

        st.markdown("---")
        st.markdown(
            f"- **Generation Model:** `{metadata.get('model', 'unknown')}`\n"
            f"- **Embedding Model:** `{metadata.get('embedding_model', 'unknown')}`\n"
            f"- **Re-rank Method:** `{metadata.get('rerank_method', 'none')}`\n"
            f"- **Hybrid Search:** `{'Enabled' if metadata.get('hybrid') else 'Disabled'}`\n"
            f"- **Pool Size:** `candidate_k={metadata.get('candidate_k')}` ➔ `final_k={metadata.get('final_k')}`\n"
            f"- **Timestamp:** `{metadata.get('generated_at', '')}`"
        )


def render_error_banner(error_type: str, message: str, details: Any = None) -> None:
    """Render a prominent, helpful error banner with troubleshooting guidance."""
    st.error(f"**{error_type}**: {message}")
    if details:
        with st.expander("🔍 Error Technical Details", expanded=False):
            st.code(str(details), language="text")
