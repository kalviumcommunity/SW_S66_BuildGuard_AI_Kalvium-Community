"""UI components and visual formatting for the BuildGuard Streamlit application.

Implements the design specification from the BuildGuard AI Wireframe summary:
- Clean, modern layout matching the wireframe design
- Top bar with project filter dropdown (Step 1-4) & "Sources checked" timestamp
- 4 Overview Metric Cards:
    1. Sources loaded (6) [NBC 2016 + IS 456:2000]
    2. Compliance reviews (4) [Recorded questions]
    3. Review required (2) [Manual verification]
    4. Refused answers (1) [Insufficient evidence]
- Central "What needs verifying on site?" empty-state heading with 4 site prompt buttons
- Structured message rendering with "Source passages" card
- Safe workflow alert cards with red/neutral tones and "Return to safe workflow" buttons:
    * Not found: "I couldn't find enough reliable evidence"
    * Sources disagree: "Manual review required"
    * Service down: "Answer service unavailable"
- Operational metadata drawer
"""

from __future__ import annotations

from typing import Any
import streamlit as st

WIRE_FRAME_CSS = """
<style>
/* Font and base theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
}

/* App container padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* Top bar navigation */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #e2e8f0;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.sources-checked-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: #64748b;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-weight: 500;
}

.sources-checked-pill::before {
    content: "●";
    color: #10b981;
    font-size: 0.75rem;
}

/* Overview cards grid */
.overview-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}

.overview-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 105px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.overview-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
}

.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.5rem;
}

.card-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.card-icon {
    font-size: 0.9rem;
    color: #94a3b8;
}

.card-number {
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

.card-desc {
    font-size: 0.72rem;
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Empty state / Welcome prompt */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}

.hero-avatar {
    width: 48px;
    height: 48px;
    background: #0f172a;
    color: white;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15);
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    max-width: 580px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.5;
}

/* Question sample cards */
.sample-btn-card {
    text-align: left;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #334155;
    cursor: pointer;
    transition: all 0.15s ease;
}

.sample-btn-card:hover {
    border-color: #94a3b8;
    background: #f8fafc;
}

/* Source Passages Card (Wireframe Page 5) */
.source-passages-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.9rem;
    margin-bottom: 0.5rem;
}

.source-passages-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
}

.source-passages-list {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.6;
}

.source-tag {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: #1e293b;
    margin-right: 0.4rem;
    margin-bottom: 0.25rem;
}

.source-tag-page {
    background: #f1f5f9;
    color: #475569;
}

.source-snippet-box {
    margin-top: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: #ffffff;
    border-left: 3px solid #64748b;
    border-radius: 0 4px 4px 0;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.45;
}

/* Safe Alert State Cards (Wireframe Page 6 & 7) */
.safe-state-card {
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin: 1.25rem 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.safe-state-red {
    background: #fef2f2;
    border: 1px solid #fecaca;
}

.safe-state-grey {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.safe-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.safe-badge-red {
    color: #dc2626;
}

.safe-badge-grey {
    color: #64748b;
}

.safe-state-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.4rem;
}

.safe-state-desc {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}

/* Disclaimer text at bottom */
.disclaimer-text {
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 1.5rem;
}

/* Badges for legacy compat */
.badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.badge-connected { background-color: #dcfce7; color: #15803d; }
.badge-disconnected { background-color: #fee2e2; color: #b91c1c; }
.badge-source { background-color: #e0f2fe; color: #0369a1; }
.badge-page { background-color: #fef3c7; color: #b45309; }
.badge-heading { background-color: #f3e8ff; color: #7e22ce; }
.badge-score { background-color: #ecfdf5; color: #047857; }
.badge-chunk { font-family: monospace; background-color: #f1f5f9; color: #475569; }
</style>
"""


def apply_custom_styles() -> None:
    """Inject wireframe CSS into the Streamlit app."""
    st.markdown(WIRE_FRAME_CSS, unsafe_allow_html=True)


def render_top_nav(selected_project: str, sources_checked_date: str = "14 Feb 2026") -> None:
    """Render the top bar with project name, status indicator, and last verification date."""
    st.markdown(
        f"""
        <div class="top-nav">
            <div class="brand-wrapper">
                <span style="font-size: 1.2rem;">🏢</span>
                <span class="brand-title">BuildGuard AI</span>
                <span style="color: #cbd5e1; margin: 0 4px;">|</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: #334155;">{selected_project}</span>
            </div>
            <div>
                <span class="sources-checked-pill">Sources checked {sources_checked_date}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_cards(
    sources_count: int = 6,
    questions_count: int = 4,
    review_required_count: int = 2,
    refused_count: int = 1,
) -> None:
    """Render the 4 overview cards as defined in Screen 1 & Screen 3 of the wireframe.

    1. Sources loaded (e.g. 6) — NBC 2016 + IS 456:2000
    2. Compliance reviews (e.g. 4) — Recorded questions
    3. Review required (e.g. 2) — Manual verification
    4. Refused answers (e.g. 1) — Insufficient evidence
    """
    st.markdown(
        f"""
        <div class="overview-grid">
            <div class="overview-card">
                <div class="card-top">
                    <span class="card-label">Sources loaded</span>
                    <span class="card-icon">📚</span>
                </div>
                <div class="card-number">{sources_count}</div>
                <div class="card-desc">NBC 2016 + IS 456:2000</div>
            </div>
            <div class="overview-card">
                <div class="card-top">
                    <span class="card-label">Compliance reviews</span>
                    <span class="card-icon">📋</span>
                </div>
                <div class="card-number">{questions_count}</div>
                <div class="card-desc">Recorded questions</div>
            </div>
            <div class="overview-card">
                <div class="card-top">
                    <span class="card-label">Review required</span>
                    <span class="card-icon">⚠️</span>
                </div>
                <div class="card-number">{review_required_count}</div>
                <div class="card-desc">Manual verification</div>
            </div>
            <div class="overview-card">
                <div class="card-top">
                    <span class="card-label">Refused answers</span>
                    <span class="card-icon">🛡️</span>
                </div>
                <div class="card-number">{refused_count}</div>
                <div class="card-desc">Insufficient evidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_hero() -> None:
    """Render the central empty state from Screen 1 before questions are asked."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-avatar">🛡️</div>
            <div class="hero-title">What needs verifying on site?</div>
            <div class="hero-subtitle">
                Ask a construction-compliance question. Every conclusion is grounded in your indexed project sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_passages_card(sources: list[dict[str, Any]]) -> None:
    """Render the 'Source passages' card matching Screen 3 and Screen 5.

    Lists documents used (NBC 2016, IS 456:2000, project specifications) with
    page references, headings, and quoted excerpts.
    """
    if not sources:
        return

    tags_html = []
    snippets_html = []

    for s in sources:
        src_name = s.get("source", "NBC 2016")
        page = s.get("page", "?")
        vol = s.get("volume")
        vol_str = f"Vol {vol}, " if vol else ""
        heading = s.get("heading") or "General Specification"
        preview = s.get("text_preview", "").strip()

        tags_html.append(
            f'<span class="source-tag">📄 {src_name}</span>'
            f'<span class="source-tag source-tag-page">p. {page}</span>'
        )

        if preview:
            snippets_html.append(
                f'<div class="source-snippet-box">'
                f'<strong>{src_name} ({vol_str}p. {page}) — {heading}</strong><br/>'
                f'"{preview}"'
                f'</div>'
            )

    tags_joined = " ".join(tags_html)
    snippets_joined = "".join(snippets_html[:3])  # show top 3 snippets max

    st.markdown(
        f"""
        <div class="source-passages-card">
            <div class="source-passages-header">
                <span>📑</span>
                <span>Source passages</span>
            </div>
            <div class="source-passages-list">
                {tags_joined}
            </div>
            {snippets_joined}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_not_found_card(detail_text: str | None = None) -> None:
    """Screen 4: Not found — 'I couldn't find enough reliable evidence'."""
    desc = (
        detail_text
        if detail_text
        else "The indexed corpus was searched, but no passage cleared the evidence threshold. The system will not guess."
    )
    st.markdown(
        f"""
        <div class="safe-state-card safe-state-red">
            <div style="font-size: 1.2rem; margin-bottom: 0.4rem;">🚫</div>
            <div class="safe-badge safe-badge-red">Annotated Edge State · Not Found</div>
            <div class="safe-state-title">I couldn't find enough reliable evidence</div>
            <div class="safe-state-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources_disagree_card(detail_text: str | None = None) -> None:
    """Screen 5: Sources disagree — 'Manual review required'."""
    desc = (
        detail_text
        if detail_text
        else "Two source provisions point to different requirements. BuildGuard surfaces both passages instead of resolving the conflict."
    )
    st.markdown(
        f"""
        <div class="safe-state-card safe-state-grey">
            <div style="font-size: 1.2rem; margin-bottom: 0.4rem;">⚖️</div>
            <div class="safe-badge safe-badge-grey">Annotated Edge State · Conflict</div>
            <div class="safe-state-title">Manual review required</div>
            <div class="safe-state-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_service_down_card(detail_text: str | None = None) -> None:
    """Screen 6: Service down — 'Answer service unavailable'."""
    desc = (
        detail_text
        if detail_text
        else "The evidence workspace remains available, but generated answers are paused. Retry instead of using an unverified conclusion."
    )
    st.markdown(
        f"""
        <div class="safe-state-card safe-state-red">
            <div style="font-size: 1.2rem; margin-bottom: 0.4rem;">⚠️</div>
            <div class="safe-badge safe-badge-red">Annotated Edge State · Service Down</div>
            <div class="safe-state-title">Answer service unavailable</div>
            <div class="safe-state-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_drawer(metadata: dict[str, Any] | None) -> None:
    """Render technical pipeline metrics in a collapsible accordion."""
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


def render_source_card(source: dict[str, Any], index: int) -> None:
    """Legacy helper maintained for backward compatibility."""
    render_source_passages_card([source])


def render_sources_section(sources: list[dict[str, Any]]) -> None:
    """Legacy helper maintained for backward compatibility."""
    render_source_passages_card(sources)


def render_header(health_status: dict[str, Any] | None, error_message: str | None = None) -> None:
    """Legacy helper maintained for backward compatibility."""
    render_top_nav(selected_project="Metro Line Extension — Phase 3")


def render_error_banner(error_type: str, message: str, details: Any = None) -> None:
    """Legacy helper maintained for backward compatibility."""
    render_service_down_card(f"{error_type}: {message}")
