"""Top-level prompt templates re-export for backwards compatibility."""

from __future__ import annotations

from src.assistant.prompt_templates import TEMPLATE_PATH, render_staff_question

__all__ = ["TEMPLATE_PATH", "render_staff_question"]
