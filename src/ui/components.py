"""UI components and visual formatting for the BuildGuard Streamlit application.

Implements the BuildGuard AI wireframe specifications:
- Project filter & top status bar (Screen 2)
- Overview KPI metric cards (Screen 1)
- Empty-state hero with 4 site engineering question pills (Screen 1)
- Source passages card with cited documents list & snippets (Screen 3)
- Annotated edge-state cards for 'Not found', 'Sources disagree', and 'Service down' (Screens 4, 5, 6)
- Disclaimer footer and pipeline metadata diagnostics
"""

from __future__ import annotations

from typing import Any
import streamlit as st


CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------
 * Google Fonts
 * ----------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* -------------------------------------------------------------
 * Global Layout & Typography
 * ----------------------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #f8fafc;
}

/* Main app background */
.stApp {
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
}

/* Prevent Streamlit top header from blocking dropdowns and top components */
header[data-testid="stHeader"] {
    background: transparent !important;
    pointer-events: none !important;
    z-index: 100 !important;
}

header[data-testid="stHeader"] * {
    pointer-events: auto !important;
}

/* Main content panel — ample clearance so top dropdown never collides with header */
.block-container {
    padding-top: 4.75rem !important;
    padding-bottom: 5rem !important;
    max-width: 1040px !important;
    background-color: transparent !important;
}

/* =============================================================
 * DARK SIDEBAR
 * Target all container layers to guarantee uniform dark navy styling
 * ============================================================= */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}

/* Sidebar text hierarchy */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: #94a3b8 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong {
    color: #f8fafc !important;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* Sidebar expanders */
section[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #162032 !important;
    border: 1px solid #27354a !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #f59e0b !important;
}

/* Sidebar inputs & dropzones */
section[data-testid="stSidebar"] input,
[data-testid="stSidebar"] input {
    background-color: #0b0f19 !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 6px !important;
}

section[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] input::placeholder {
    color: #64748b !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #0b0f19 !important;
    border: 1px dashed #334155 !important;
    border-radius: 8px !important;
}

/* Sidebar history & action buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button,
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: #162032 !important;
    border: 1px solid #27354a !important;
    color: #cbd5e1 !important;
    font-size: 0.8125rem !important;
    text-align: left !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.85rem !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #1e293b !important;
    border-color: #f59e0b !important;
    color: #ffffff !important;
}

/* Sidebar primary button (New Chat) — vibrant amber */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    border: none !important;
    color: #0b0f19 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25) !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    box-shadow: 0 6px 18px rgba(245, 158, 11, 0.40) !important;
    color: #0b0f19 !important;
}

/* Brand gold accent line */
.brand-accent-line {
    width: 36px;
    height: 3px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
    border-radius: 2px;
    margin-bottom: 0.5rem;
}

/* =============================================================
 * Main Area Dropdowns / Selectboxes (Active Project Filter)
 * ============================================================= */
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] {
    background-color: #162032 !important;
    border: 1px solid #27354a !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] * {
    color: #f8fafc !important;
    background-color: transparent !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    z-index: 9999999 !important;
    background-color: #162032 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
}

li[role="option"] {
    background-color: #162032 !important;
    color: #cbd5e1 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #1e293b !important;
    color: #f59e0b !important;
}

/* =============================================================
 * Overview Metric Cards (Screen 1)
 * ============================================================= */
.overview-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.875rem;
    margin-top: 0.5rem;
    margin-bottom: 1.75rem;
}

.overview-card {
    background: #162032;
    border: 1px solid #27354a;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
}

.overview-card:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.card-top-tag {
    font-size: 0.6875rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}

.card-number {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1;
    letter-spacing: -0.025em;
}

.card-icon-pill {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
}

.icon-green { background: rgba(34, 197, 94, 0.18); border: 1px solid rgba(34, 197, 94, 0.35); }
.icon-blue  { background: rgba(59, 130, 246, 0.18); border: 1px solid rgba(59, 130, 246, 0.35); }
.icon-amber { background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(245, 158, 11, 0.35); }
.icon-rose  { background: rgba(244, 63, 94, 0.18); border: 1px solid rgba(244, 63, 94, 0.35); }

.card-label {
    font-size: 0.8125rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 0.2rem;
}

.card-subtext {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* =============================================================
 * Empty State Hero (Screen 1)
 * ============================================================= */
.empty-state-hero {
    text-align: center;
    margin: 2.25rem auto 1.75rem auto;
    max-width: 600px;
}

.hero-avatar-circle {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid #334155;
    color: #f59e0b;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.5rem;
    line-height: 1.25;
}

.hero-subtitle {
    font-size: 0.9375rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* =============================================================
 * Site Engineering Question Buttons (2x2 Grid)
 * ============================================================= */
.main div[data-testid="stButton"] > button,
section[data-testid="stMain"] div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    transition: all 0.2s ease-in-out !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    background: #162032 !important;
    border: 1px solid #27354a !important;
    color: #e2e8f0 !important;
    padding: 0.75rem 1rem !important;
    min-height: 52px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
}

.main div[data-testid="stButton"] > button:hover,
section[data-testid="stMain"] div[data-testid="stButton"] > button:hover {
    background: #1e2b42 !important;
    border-color: #f59e0b !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.20) !important;
    transform: translateY(-1px);
}

/* =============================================================
 * Chat Messages & Dialogue
 * ============================================================= */
[data-testid="stChatMessage"] {
    background-color: #162032 !important;
    border: 1px solid #27354a !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 1rem !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #e2e8f0 !important;
}

[data-testid="stChatMessage"] strong {
    color: #f8fafc !important;
}

/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #1a2538 !important;
    border-color: #2e3e56 !important;
}

/* =============================================================
 * Chat Input
 * ============================================================= */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    background-color: #162032 !important;
    border: 1px solid #27354a !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.25) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #f8fafc !important;
    font-size: 0.9375rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

[data-testid="stChatInput"] button {
    color: #f59e0b !important;
}

/* =============================================================
 * Source Passages Card (Screen 3)
 * ============================================================= */
.source-passages-container {
    background: #162032;
    border: 1px solid #27354a;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.875rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.source-passages-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.65rem;
}

.source-tags-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px dashed #27354a;
}

.source-tag {
    font-size: 0.75rem;
    font-weight: 500;
    background: #0f172a;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
}

.source-tag-accent {
    background: #1e3a5f;
    color: #93c5fd;
    border-color: #2563eb;
}

.passage-snippet-box {
    background: #0b0f19;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.8125rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.passage-meta-line {
    font-size: 0.75rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
}

/* =============================================================
 * Edge State Cards (Screens 4, 5, 6)
 * ============================================================= */
.edge-state-card {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}

.edge-state-red {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 4px solid #ef4444;
}

.edge-state-grey {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155;
    border-left: 4px solid #64748b;
}

.edge-eyebrow-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.edge-eyebrow-text {
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #94a3b8;
    text-transform: uppercase;
}

.edge-pill {
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}

.edge-pill-dark {
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
}

.edge-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}

.edge-description {
    font-size: 0.875rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.conflict-provisions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.conflict-box {
    background: #0b0f19;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.85rem;
    font-size: 0.8125rem;
    color: #cbd5e1;
}

.conflict-box-title {
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}

.error-detail-box {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    padding: 0.6rem 0.85rem;
    font-size: 0.75rem;
    color: #fca5a5;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    margin-bottom: 1rem;
    word-break: break-word;
}

/* Return to safe workflow action button */
.safe-workflow-action div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: #0b0f19 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 0.8125rem !important;
    padding: 0.55rem 1.1rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25) !important;
}

.safe-workflow-action div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    box-shadow: 0 6px 18px rgba(245, 158, 11, 0.40) !important;
    color: #0b0f19 !important;
}

/* Chat disclaimer footer */
.chat-footer-disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
</style>
"""


def apply_custom_styles() -> None:
    """Inject custom CSS into the Streamlit app to match BuildGuard wireframe."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_project_bar(
    selected_project: str,
    sources_checked_date: str = "14 Feb 2026",
    is_connected: bool = True,
) -> None:
    """Render top bar with project name and sources checked timestamp (Screen 2)."""
    status_dot_class = "status-dot-green" if is_connected else "status-dot-red"
    status_text = f"Sources checked {sources_checked_date}" if is_connected else "API Offline"

    st.markdown(
        f"""
        <div class="top-bar-container">
            <div class="top-bar-project">
                <span>📁</span>
                <span><strong>{selected_project}</strong></span>
            </div>
            <div class="top-bar-status">
                <span class="{status_dot_class}"></span>
                <span>{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_cards(
    indexed_sources: int = 6,
    compliance_reviews: int = 4,
    review_required: int = 2,
    refused_answers: int = 1,
    standards_text: str = "NBC 2016 + IS 456:2000",
) -> None:
    """Render 4 overview cards above the chat matching Wireframe Screen 1."""
    st.markdown(
        f"""
        <div class="overview-grid">
            <div class="overview-card">
                <div class="card-top-tag">Compliance overview</div>
                <div class="card-top-row">
                    <div class="card-number">{indexed_sources}</div>
                    <div class="card-icon-pill icon-green">📚</div>
                </div>
                <div class="card-label">Indexed sources</div>
                <div class="card-subtext">{standards_text}</div>
            </div>
            <div class="overview-card">
                <div class="card-top-tag">&nbsp;</div>
                <div class="card-top-row">
                    <div class="card-number">{compliance_reviews}</div>
                    <div class="card-icon-pill icon-blue">📋</div>
                </div>
                <div class="card-label">Compliance reviews</div>
                <div class="card-subtext">Recorded questions</div>
            </div>
            <div class="overview-card">
                <div class="card-top-tag">&nbsp;</div>
                <div class="card-top-row">
                    <div class="card-number">{review_required}</div>
                    <div class="card-icon-pill icon-amber">⚠️</div>
                </div>
                <div class="card-label">Review required</div>
                <div class="card-subtext">Manual verification</div>
            </div>
            <div class="overview-card">
                <div class="card-top-tag">MLS P3 · Sample data</div>
                <div class="card-top-row">
                    <div class="card-number">{refused_answers}</div>
                    <div class="card-icon-pill icon-rose">🛑</div>
                </div>
                <div class="card-label">Refused answers</div>
                <div class="card-subtext">Insufficient evidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_hero() -> None:
    """Render centered greeting and prompt for Screen 1."""
    st.markdown(
        """
        <div class="empty-state-hero">
            <div class="hero-avatar-circle">🛡️</div>
            <div class="hero-title">What needs verifying on site?</div>
            <div class="hero-subtitle">
                Ask a construction-compliance question. Every conclusion is grounded in your indexed project sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_passages_card(
    sources: list[dict[str, Any]],
    cited_docs: list[str] | None = None,
) -> None:
    """Render 'Source passages' card matching Wireframe Screen 3."""
    if not sources:
        return

    # Extract distinct document names for the top badge list
    if not cited_docs:
        seen = set()
        cited_docs = []
        for s in sources:
            src_name = s.get("source", "NBC Document")
            if src_name not in seen:
                seen.add(src_name)
                # Map technical file names to readable doc titles if needed
                if "nationalbuilding01" in src_name.lower():
                    cited_docs.append("NBC 2016, Part 4")
                elif "nationalbuilding02" in src_name.lower():
                    cited_docs.append("NBC 2016, Vol 2")
                else:
                    cited_docs.append(src_name)

        if not cited_docs:
            cited_docs = ["NBC 2016, Part 4", "IS 456:2000", "Project Structural Specification, Rev B"]

    tags_html = "".join([f'<span class="source-tag source-tag-accent">{doc}</span>' for doc in cited_docs])

    snippets_html = ""
    for idx, s in enumerate(sources[:3]):
        heading = s.get("heading") or "General Provision"
        page = s.get("page", "?")
        vol = s.get("volume", "")
        loc_str = f"Vol {vol}, Page {page}" if vol else f"Page {page}"
        snippet = s.get("text_preview", "")
        snippets_html += f"""
        <div class="passage-snippet-box">
            <div class="passage-meta-line">
                <span>{heading}</span>
                <span>{loc_str}</span>
            </div>
            <div>"{snippet}"</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="source-passages-container">
            <div class="source-passages-header">
                <span>📄 Source passages</span>
                <span style="font-size: 0.75rem; color: #10b981; font-weight: 500;">✓ Verified Grounding</span>
            </div>
            <div class="source-tags-bar">
                {tags_html}
            </div>
            {snippets_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_edge_state_not_found(return_button_key: str = "btn_return_not_found") -> bool:
    """Render Wireframe Screen 4: 'Not found' (US-03) with red stop accent."""
    st.markdown(
        """
        <div class="edge-state-card edge-state-red">
            <div class="edge-eyebrow-row">
                <span class="edge-eyebrow-text">ANNOTATED EDGE STATE</span>
                <span class="edge-pill edge-pill-dark">US-03</span>
            </div>
            <div class="edge-heading">I couldn't find enough reliable evidence</div>
            <div class="edge-description">
                The indexed corpus was searched, but no passage cleared the evidence threshold.
                The system will not guess.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, _ = st.columns([1, 3])
    with col1:
        st.markdown('<div class="safe-workflow-action">', unsafe_allow_html=True)
        clicked = st.button("Return to safe workflow", key=return_button_key, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return clicked


def render_edge_state_sources_disagree(
    provisions: list[dict[str, str]] | None = None,
    return_button_key: str = "btn_return_disagree",
) -> bool:
    """Render Wireframe Screen 5: 'Sources disagree' (Safe answers) with neutral grey accent."""
    if not provisions:
        provisions = [
            {
                "title": "NBC 2016, Part 4, Clause 6.1.1.2",
                "text": "Walls separating a boiler room or transformer enclosure must provide a minimum fire resistance rating of 120 minutes (2 hours).",
            },
            {
                "title": "Project Specification, Section 08-200 (Rev B)",
                "text": "Underground metro station plant room perimeter walls adjoining public egress paths specify 180 minutes (3 hours) fire separation.",
            },
        ]

    prov_html = "".join([
        f"""
        <div class="conflict-box">
            <div class="conflict-box-title">📌 {p['title']}</div>
            <div style="color: #475569;">"{p['text']}"</div>
        </div>
        """
        for p in provisions
    ])

    st.markdown(
        f"""
        <div class="edge-state-card edge-state-grey">
            <div class="edge-eyebrow-row">
                <span class="edge-eyebrow-text">ANNOTATED EDGE STATE</span>
                <span class="edge-pill edge-pill-dark">Safe answers</span>
            </div>
            <div class="edge-heading">Manual review required</div>
            <div class="edge-description">
                Two source provisions point to different requirements. BuildGuard surfaces both passages instead of resolving the conflict.
            </div>
            <div class="conflict-provisions-grid">
                {prov_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, _ = st.columns([1, 3])
    with col1:
        st.markdown('<div class="safe-workflow-action">', unsafe_allow_html=True)
        clicked = st.button("Return to safe workflow", key=return_button_key, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return clicked


def render_edge_state_service_down(
    return_button_key: str = "btn_return_service_down",
    error_detail: str = "",
) -> bool:
    """Render Wireframe Screen 6: 'Service down' (Safe answers) with red stop accent."""
    error_html = ""
    if error_detail:
        safe_detail = error_detail.replace("<", "&lt;").replace(">", "&gt;")[:300]
        error_html = f'<div class="error-detail-box">{safe_detail}</div>'

    st.markdown(
        f"""
        <div class="edge-state-card edge-state-red">
            <div class="edge-eyebrow-row">
                <span class="edge-eyebrow-text">ANNOTATED EDGE STATE</span>
                <span class="edge-pill edge-pill-dark">Safe answers</span>
            </div>
            <div class="edge-heading">Answer service unavailable</div>
            <div class="edge-description">
                The evidence workspace remains available, but generated answers are paused. Retry instead of using an unverified conclusion.
            </div>
            {error_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, _ = st.columns([1, 3])
    with col1:
        st.markdown('<div class="safe-workflow-action">', unsafe_allow_html=True)
        clicked = st.button("Return to safe workflow", key=return_button_key, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return clicked


def render_chat_disclaimer() -> None:
    """Render the wireframe disclaimer below the chat input."""
    st.markdown(
        '<div class="chat-footer-disclaimer">'
        "BuildGuard can make mistakes. Verify citations and project amendments."
        "</div>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# Backwards Compatible Component Wrappers
# -----------------------------------------------------------
def render_header(health_status: dict[str, Any] | None, error_message: str | None = None) -> None:
    """Render legacy header if needed."""
    is_connected = bool(health_status and health_status.get("status") == "ok")
    render_project_bar(
        selected_project="Metro Line Extension — Phase 3",
        sources_checked_date="14 Feb 2026",
        is_connected=is_connected,
    )


def render_source_card(source: dict[str, Any], index: int) -> None:
    """Render individual source card."""
    render_source_passages_card([source])


def render_sources_section(sources: list[dict[str, Any]]) -> None:
    """Render full source passages section."""
    render_source_passages_card(sources)


def render_metadata_drawer(metadata: dict[str, Any] | None) -> None:
    """Render pipeline operational metrics in a compact accordion."""
    if not metadata:
        return

    with st.expander("⚙️ Pipeline Execution Diagnostics", expanded=False):
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

        st.markdown(
            f"- **Model:** `{metadata.get('model', 'unknown')}` | "
            f"**Embedding:** `{metadata.get('embedding_model', 'unknown')}`\n"
            f"- **Rerank:** `{metadata.get('rerank_method', 'none')}` | "
            f"**Hybrid:** `{'Yes' if metadata.get('hybrid') else 'No'}`\n"
            f"- **Timestamp:** `{metadata.get('generated_at', '')}`"
        )


def render_error_banner(error_type: str, message: str, details: Any = None) -> None:
    """Render a prominent error banner."""
    st.error(f"**{error_type}**: {message}")
    if details:
        with st.expander("🔍 Technical Details", expanded=False):
            st.code(str(details), language="text")
