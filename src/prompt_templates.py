"""Top-level prompt templates re-export for backwards compatibility."""

from __future__ import annotations

from src.assistant.prompt_templates import TEMPLATE_PATH, render_staff_question
from src.conversational_rag import CONVERSATIONAL_SYSTEM_PROMPT, QUERY_REWRITER_SYSTEM_PROMPT

__all__ = [
    "TEMPLATE_PATH",
    "render_staff_question",
    "CONVERSATIONAL_SYSTEM_PROMPT",
    "QUERY_REWRITER_SYSTEM_PROMPT",
]
