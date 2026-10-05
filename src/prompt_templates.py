"""Shared prompt templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "staff_question_template.txt"
)


def render_staff_question(*, context: str, question: str) -> str:
    """Render the shared staff question template."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(context=context, question=question)
