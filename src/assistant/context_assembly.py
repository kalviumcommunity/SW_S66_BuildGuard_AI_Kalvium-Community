"""Context injection and prompt augmentation re-export for assistant package."""

from __future__ import annotations

from src.context_assembly import (
    AugmentedPromptResult,
    InjectedChunk,
    TokenBudget,
    assemble_context_with_budget,
    build_augmented_prompt,
    count_tokens,
    format_chunk_with_marker,
    generate_markdown_report,
    get_synthetic_test_chunks,
    print_prompt_summary,
    save_augmented_prompt,
    DEFAULT_GROUNDING_INSTRUCTIONS,
    INSUFFICIENT_CONTEXT_FALLBACK,
)

__all__ = [
    "AugmentedPromptResult",
    "InjectedChunk",
    "TokenBudget",
    "assemble_context_with_budget",
    "build_augmented_prompt",
    "count_tokens",
    "format_chunk_with_marker",
    "generate_markdown_report",
    "get_synthetic_test_chunks",
    "print_prompt_summary",
    "save_augmented_prompt",
    "DEFAULT_GROUNDING_INSTRUCTIONS",
    "INSUFFICIENT_CONTEXT_FALLBACK",
]
