"""Context Injection & Prompt Augmentation.

Assembles retrieved text chunks into a grounded, token-budgeted prompt for
the National Building Code of India (NBC) assistant.

Responsibilities:
  Task 1 — Format retrieved chunks and inject them into the prompt context.
  Task 2 — Enforce a token budget, leaving guaranteed room for system instructions,
            user question, and generated answer.
  Task 3 — Include unambiguous source markers ([1], [2], [Volume X, PDF p. Y])
            for reliable in-text citations.
  Task 4 — Provide explicit grounding instructions compelling the model to
            answer only from provided context and state when information is insufficient.
  Task 5 — Output and commit sample augmented prompts showing injected chunks,
            source markers, and token budget analytics.

Usage:
  # Run CLI with default NBC query against ChromaDB:
  uv run python -m src.context_assembly

  # Custom query with tighter token budget:
  uv run python -m src.context_assembly \\
      --query "What is the fire resistance rating for boiler room walls?" \\
      --max-tokens 2048 \\
      --reserved-answer-tokens 384

  # Dry-run with synthetic chunks (no database or model download needed):
  uv run python -m src.context_assembly --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Re-use project defaults where applicable
from src.embeddings.embed_text_chunks import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
    _safe_collection_name,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "prompts"
DEFAULT_ENCODING = "cl100k_base"

# Budget defaults
DEFAULT_MAX_TOTAL_TOKENS = 4096       # Total context window budget
DEFAULT_RESERVED_ANSWER_TOKENS = 512  # Guaranteed tokens reserved for LLM generation
DEFAULT_MIN_CHUNK_TOKENS = 25         # Minimum text tokens needed to justify partial truncation

DEFAULT_SAMPLE_QUERY = (
    "What is the minimum fire resistance rating required for walls separating "
    "a boiler room from other parts of a building?"
)

# ---------------------------------------------------------------------------
# Grounding Instructions & Fallback
# ---------------------------------------------------------------------------

INSUFFICIENT_CONTEXT_FALLBACK = (
    "I don't have enough information to answer that reliably based on the "
    "provided National Building Code context."
)

DEFAULT_GROUNDING_INSTRUCTIONS = (
    "You are an expert staff-support assistant answering technical compliance questions "
    "about the National Building Code of India (NBC).\n"
    "Follow these strict operational rules:\n"
    "1. Answer the question using ONLY the factual evidence directly stated in the "
    "RETRIEVED BUILDING CODE CONTEXT below.\n"
    "2. Do NOT extrapolate, speculate, assume, or introduce outside knowledge not "
    "corroborated by the provided context.\n"
    "3. Attribute every factual claim, requirement, rating, dimension, or specification "
    "using the bracketed source markers (e.g. [1], [2]) or explicit citations "
    "such as [Volume 1, PDF p. 314-315].\n"
    "4. If the provided context does NOT contain sufficient factual evidence to answer "
    "the question reliably, reply with this exact fallback message:\n"
    f'   "{INSUFFICIENT_CONTEXT_FALLBACK}"\n'
    "5. Keep the response professional, precise, and concise (typically 2 to 4 sentences)."
)

_USER_PROMPT_TEMPLATE = """=== RETRIEVED BUILDING CODE CONTEXT ===
{context_block}
=======================================

GROUNDING INSTRUCTIONS:
- Answer using ONLY the retrieved context above.
- Cite your sources for every factual assertion using markers like [1], [2] or [Volume X, PDF p. Y].
- If the context does not provide sufficient evidence, output: "{fallback_phrase}".

USER QUESTION:
{query}

COMPLIANCE ANSWER:"""

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TokenBudget:
    """Detailed token budget accounting for prompt assembly."""
    max_total_tokens: int
    reserved_answer_tokens: int
    system_tokens: int
    query_template_overhead_tokens: int
    available_context_tokens: int
    used_context_tokens: int
    total_prompt_tokens: int
    remaining_tokens: int
    chunks_considered: int
    chunks_injected: int
    chunks_truncated: int
    chunks_skipped: int
    utilization_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InjectedChunk:
    """Individual chunk formatted and packed into the context."""
    index: int
    marker: str                   # e.g., "[1]"
    chunk_id: str
    source_file: str
    volume: str | int
    page: str | int
    clause: str
    heading: str
    cosine_similarity: float | None
    rerank_score: float | None
    token_count: int
    truncated: bool
    formatted_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AugmentedPromptResult:
    """Assembled augmented prompt with full token accounting and source markers."""
    query: str
    system_prompt: str
    user_prompt: str
    combined_prompt: str
    context_block: str
    token_budget: TokenBudget
    injected_chunks: list[InjectedChunk]
    grounding_instructions: str
    fallback_phrase: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context_block": self.context_block,
            "token_budget": self.token_budget.to_dict(),
            "injected_chunks": [c.to_dict() for c in self.injected_chunks],
            "grounding_instructions": self.grounding_instructions,
            "fallback_phrase": self.fallback_phrase,
            "created_at": self.created_at,
        }

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Return standard OpenAI-format messages list."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


# ---------------------------------------------------------------------------
# Token Counting
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Any] = {}


def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> Any:
    """Retrieve or cache the tiktoken encoding."""
    if encoding_name not in _TOKENIZER_CACHE:
        try:
            import tiktoken
            _TOKENIZER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning("Could not initialize tiktoken encoding %s (%s). Using fallback counter.", encoding_name, exc)
            _TOKENIZER_CACHE[encoding_name] = None
    return _TOKENIZER_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in *text* using tiktoken with fallback estimation."""
    if not text:
        return 0
    enc = get_tokenizer(encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback heuristic: approx 4 characters per token
    return max(1, math.ceil(len(text) / 4.0))


# ---------------------------------------------------------------------------
# Source Marker Formatting
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 60


def _extract_page_repr(meta: dict[str, Any]) -> str:
    """Extract page or page range string from chunk metadata."""
    p_start = meta.get("page_start")
    p_end = meta.get("page_end")
    if p_start is not None and p_end is not None and p_start != p_end:
        return f"{p_start}-{p_end}"
    if p_start is not None:
        return str(p_start)
    return str(meta.get("page", meta.get("page_number", "?")))


def format_chunk_with_marker(
    chunk: dict[str, Any],
    index: int,
    *,
    marker_style: str = "numeric",
    text_override: str | None = None,
) -> tuple[str, InjectedChunk]:
    """Format a single chunk with unambiguous source markers and metadata tags.

    Parameters:
      chunk: Raw dictionary from ChromaDB / retrieval stage.
      index: 1-based chunk sequence index.
      marker_style: 'numeric' produces '[1]', 'verbose' produces '[Source 1]'.
      text_override: Explicit text to use (e.g., if truncated).

    Returns:
      (formatted_string, InjectedChunk_record)
    """
    meta = chunk.get("metadata", {})
    text = (text_override if text_override is not None else chunk.get("text", "")).strip()
    chunk_id = chunk.get("id", f"chunk-{index}")
    source_file = meta.get("source", "nationalbuilding01.pdf")
    volume = meta.get("volume", 1)
    page = _extract_page_repr(meta)
    clause = meta.get("clause", "")
    heading = meta.get("heading", "")
    sim = chunk.get("cosine_similarity", chunk.get("score"))
    rerank = chunk.get("rerank_score")

    marker_tag = f"[{index}]" if marker_style == "numeric" else f"[Source {index}]"

    # Build header metadata line
    meta_parts = [
        marker_tag,
        f"Doc: {source_file}",
        f"Volume: {volume}",
        f"Page: {page}",
    ]
    if clause:
        meta_parts.append(f"Clause: {clause}")
    if heading:
        meta_parts.append(f"Section: {heading}")

    header_line = " | ".join(meta_parts)

    score_parts: list[str] = []
    if sim is not None:
        try:
            score_parts.append(f"Similarity: {float(sim):.4f}")
        except (ValueError, TypeError):
            pass
    if rerank is not None:
        try:
            score_parts.append(f"Re-rank Score: {float(rerank):.4f}")
        except (ValueError, TypeError):
            pass

    score_line = ("Score: " + " | ".join(score_parts)) if score_parts else ""

    lines = [header_line]
    if score_line:
        lines.append(score_line)
    lines.extend([
        _DIVIDER,
        text,
        _DIVIDER,
        "",
    ])
    formatted_block = "\n".join(lines)
    token_cnt = count_tokens(formatted_block)

    record = InjectedChunk(
        index=index,
        marker=marker_tag,
        chunk_id=chunk_id,
        source_file=source_file,
        volume=volume,
        page=page,
        clause=clause,
        heading=heading,
        cosine_similarity=float(sim) if sim is not None else None,
        rerank_score=float(rerank) if rerank is not None else None,
        token_count=token_cnt,
        truncated=(text_override is not None),
        formatted_text=formatted_block,
    )
    return formatted_block, record


# ---------------------------------------------------------------------------
# Context Assembly & Token Budget Enforcement
# ---------------------------------------------------------------------------


def assemble_context_with_budget(
    chunks: list[dict[str, Any]],
    *,
    available_context_tokens: int,
    marker_style: str = "numeric",
    allow_partial_truncation: bool = True,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[str, list[InjectedChunk], int, int, int]:
    """Assemble candidate chunks into context while strictly respecting token budget.

    Greedily includes chunks in order of priority (relevance) until
    available_context_tokens is exhausted.

    Parameters:
      chunks: Candidate chunks sorted in order of relevance.
      available_context_tokens: Hard ceiling for context block tokens.
      marker_style: Citation marker formatting style.
      allow_partial_truncation: If True, partially truncates a chunk if it
                                exceeds the remaining budget but has >= min_chunk_tokens.
      min_chunk_tokens: Minimum tokens required to justify partial truncation.

    Returns:
      (context_block, injected_chunk_records, used_tokens, skipped_count, truncated_count)
    """
    if not chunks or available_context_tokens <= 0:
        empty_text = "[No relevant context available within token budget]"
        return empty_text, [], count_tokens(empty_text, encoding_name), len(chunks), 0

    injected: list[InjectedChunk] = []
    formatted_blocks: list[str] = []
    used_tokens = 0
    skipped_count = 0
    truncated_count = 0

    for i, chunk in enumerate(chunks, start=1):
        formatted_full, record_full = format_chunk_with_marker(
            chunk, len(injected) + 1, marker_style=marker_style
        )
        chunk_cost = record_full.token_count
        remaining = available_context_tokens - used_tokens

        if chunk_cost <= remaining:
            # Full chunk fits
            formatted_blocks.append(formatted_full)
            injected.append(record_full)
            used_tokens += chunk_cost
        elif allow_partial_truncation:
            # Chunk partially fits: calculate exact headroom for text
            notice = "\n[... truncated to fit token budget ...]"
            notice_cost = count_tokens(notice, encoding_name)
            _, r_empty = format_chunk_with_marker(
                chunk, len(injected) + 1, marker_style=marker_style, text_override=""
            )
            base_overhead = r_empty.token_count
            available_for_text = remaining - base_overhead - notice_cost

            if available_for_text >= min_chunk_tokens:
                text = chunk.get("text", "").strip()
                enc = get_tokenizer(encoding_name)
                if enc is not None:
                    tokens = enc.encode(text)
                    truncated_text = enc.decode(tokens[:available_for_text]) + notice
                else:
                    char_budget = available_for_text * 4
                    truncated_text = text[:char_budget].rsplit(" ", 1)[0] + notice

                formatted_trunc, record_trunc = format_chunk_with_marker(
                    chunk, len(injected) + 1, marker_style=marker_style, text_override=truncated_text
                )
                actual_cost = record_trunc.token_count
                if actual_cost <= remaining:
                    formatted_blocks.append(formatted_trunc)
                    injected.append(record_trunc)
                    used_tokens += actual_cost
                    truncated_count += 1
                else:
                    skipped_count += 1
                # Stop after filling remaining capacity
                break
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    final_context = "\n".join(formatted_blocks).strip()
    return final_context, injected, used_tokens, skipped_count, truncated_count


# ---------------------------------------------------------------------------
# Augmented Prompt Builder
# ---------------------------------------------------------------------------


def build_augmented_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    reserved_answer_tokens: int = DEFAULT_RESERVED_ANSWER_TOKENS,
    grounding_instructions: str = DEFAULT_GROUNDING_INSTRUCTIONS,
    fallback_phrase: str = INSUFFICIENT_CONTEXT_FALLBACK,
    marker_style: str = "numeric",
    allow_partial_truncation: bool = True,
    encoding_name: str = DEFAULT_ENCODING,
) -> AugmentedPromptResult:
    """Construct an augmented, grounded prompt within strict token limits.

    Budget Allocation
    -----------------
    max_total_tokens
      - reserved_answer_tokens (guaranteed generation window)
      - system_tokens (grounding rules and system persona)
      - query_template_overhead_tokens (user question and framing)
      = available_context_tokens (allocated for injected chunks)

    Returns:
      AugmentedPromptResult containing system_prompt, user_prompt,
      context_block, InjectedChunk records, and TokenBudget analytics.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    system_prompt = grounding_instructions.strip()
    system_tokens = count_tokens(system_prompt, encoding_name)

    # Measure fixed overhead of user prompt template (excluding context)
    empty_user_prompt = _USER_PROMPT_TEMPLATE.format(
        context_block="",
        fallback_phrase=fallback_phrase,
        query=query.strip(),
    )
    query_template_overhead_tokens = count_tokens(empty_user_prompt, encoding_name)

    # Compute available token budget for context chunks
    available_context_tokens = (
        max_total_tokens
        - reserved_answer_tokens
        - system_tokens
        - query_template_overhead_tokens
    )

    if available_context_tokens < 0:
        raise ValueError(
            f"Token budget exhausted by fixed prompts alone: max_total_tokens={max_total_tokens}, "
            f"reserved_answer={reserved_answer_tokens}, system={system_tokens}, "
            f"query_overhead={query_template_overhead_tokens}."
        )

    # Assemble context chunks respecting budget
    context_block, injected_chunks, used_context_tokens, skipped, truncated = assemble_context_with_budget(
        chunks,
        available_context_tokens=available_context_tokens,
        marker_style=marker_style,
        allow_partial_truncation=allow_partial_truncation,
        encoding_name=encoding_name,
    )

    # Render full user prompt with injected context
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        context_block=context_block,
        fallback_phrase=fallback_phrase,
        query=query.strip(),
    )

    total_prompt_tokens = system_tokens + count_tokens(user_prompt, encoding_name)
    remaining_tokens = max_total_tokens - total_prompt_tokens - reserved_answer_tokens
    utilization_pct = round((total_prompt_tokens + reserved_answer_tokens) / max_total_tokens * 100.0, 1)

    budget = TokenBudget(
        max_total_tokens=max_total_tokens,
        reserved_answer_tokens=reserved_answer_tokens,
        system_tokens=system_tokens,
        query_template_overhead_tokens=query_template_overhead_tokens,
        available_context_tokens=available_context_tokens,
        used_context_tokens=used_context_tokens,
        total_prompt_tokens=total_prompt_tokens,
        remaining_tokens=max(0, remaining_tokens),
        chunks_considered=len(chunks),
        chunks_injected=len(injected_chunks),
        chunks_truncated=truncated,
        chunks_skipped=skipped,
        utilization_pct=utilization_pct,
    )

    combined_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "AUGMENT_PROMPT  injected=%d/%d  context_tokens=%d/%d  total_prompt=%d  util=%.1f%%",
        len(injected_chunks), len(chunks), used_context_tokens, available_context_tokens,
        total_prompt_tokens, utilization_pct,
    )

    return AugmentedPromptResult(
        query=query.strip(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        combined_prompt=combined_prompt,
        context_block=context_block,
        token_budget=budget,
        injected_chunks=injected_chunks,
        grounding_instructions=grounding_instructions,
        fallback_phrase=fallback_phrase,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Persistence & Markdown Report Generation
# ---------------------------------------------------------------------------


def generate_markdown_report(result: AugmentedPromptResult) -> str:
    """Generate human-readable Markdown report for inspection and documentation."""
    tb = result.token_budget
    lines = [
        "# Augmented Prompt & Context Assembly Report",
        "",
        f"**Generated:** `{result.created_at}`  ",
        f"**Query:** *{result.query}*",
        "",
        "## 1. Token Budget Allocation",
        "",
        "| Budget Metric | Tokens | % of Max Window | Description |",
        "|:---|---:|---:|:---|",
        f"| **Max Window Budget** | **{tb.max_total_tokens:,}** | 100.0% | Total model context window limit |",
        f"| **Reserved for Generation** | **{tb.reserved_answer_tokens:,}** | {tb.reserved_answer_tokens / tb.max_total_tokens * 100:.1f}% | Guaranteed headroom for model output |",
        f"| System Instructions | {tb.system_tokens:,} | {tb.system_tokens / tb.max_total_tokens * 100:.1f}% | Grounding constraints and persona rules |",
        f"| Query + Framing Overhead | {tb.query_template_overhead_tokens:,} | {tb.query_template_overhead_tokens / tb.max_total_tokens * 100:.1f}% | User question and instruction templates |",
        f"| **Available Context Budget** | **{tb.available_context_tokens:,}** | {tb.available_context_tokens / tb.max_total_tokens * 100:.1f}% | Maximum allowed for injected chunks |",
        f"| **Used Context Tokens** | **{tb.used_context_tokens:,}** | {tb.used_context_tokens / tb.max_total_tokens * 100:.1f}% | Actual tokens consumed by chunks |",
        f"| Total Prompt Tokens | {tb.total_prompt_tokens:,} | {tb.total_prompt_tokens / tb.max_total_tokens * 100:.1f}% | Input tokens sent to API (system + user) |",
        f"| Remaining Safety Headroom | {tb.remaining_tokens:,} | {tb.remaining_tokens / tb.max_total_tokens * 100:.1f}% | Unused token capacity |",
        f"| **Overall Budget Utilization** | **—** | **{tb.utilization_pct:.1f}%** | (Input prompt + reserved answer) / Max |",
        "",
        f"- **Chunks Considered:** {tb.chunks_considered}",
        f"- **Chunks Injected:** {tb.chunks_injected}",
        f"- **Chunks Truncated:** {tb.chunks_truncated}",
        f"- **Chunks Skipped:** {tb.chunks_skipped}",
        "",
        "## 2. Injected Sources & Markers",
        "",
        "| Marker | Chunk ID | Source Doc | Vol | Page | Clause | Section | Sim Score | Re-rank Score | Tokens | Status |",
        "|:---|:---|:---|:---:|:---:|:---|:---|---:|---:|---:|:---:|",
    ]

    for c in result.injected_chunks:
        sim_str = f"{c.cosine_similarity:.4f}" if c.cosine_similarity is not None else "—"
        rr_str = f"{c.rerank_score:.4f}" if c.rerank_score is not None else "—"
        clause_str = c.clause or "—"
        status_str = "Truncated" if c.truncated else "Full"
        lines.append(
            f"| `{c.marker}` | `{c.chunk_id}` | {c.source_file} | {c.volume} | {c.page} | "
            f"{clause_str} | {c.heading[:30]} | {sim_str} | {rr_str} | {c.token_count} | {status_str} |"
        )

    lines.extend([
        "",
        "## 3. Grounding Instructions (System Message)",
        "",
        "```text",
        result.system_prompt,
        "```",
        "",
        "## 4. Rendered User Prompt with Injected Context",
        "",
        "```text",
        result.user_prompt,
        "```",
    ])
    return "\n".join(lines)


def save_augmented_prompt(
    result: AugmentedPromptResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    label: str = "sample",
) -> tuple[Path, Path]:
    """Persist augmented prompt JSON and Markdown documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}_augmented_prompt.json"
    md_path = output_dir / f"{label}_augmented_prompt.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    md_content = generate_markdown_report(result)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved prompt artifacts: %s and %s", json_path, md_path)
    return json_path, md_path


def print_prompt_summary(result: AugmentedPromptResult) -> None:
    """Print high-level budget and chunk summary to stdout."""
    tb = result.token_budget
    bar = "=" * 72
    print(f"\n{bar}")
    print("CONTEXT INJECTION & PROMPT AUGMENTATION SUMMARY")
    print(bar)
    print(f"Query: {result.query}")
    print(f"\nTOKEN BUDGET:")
    print(f"  Max Total Window:      {tb.max_total_tokens:>6} tokens")
    print(f"  Reserved for Answer:   {tb.reserved_answer_tokens:>6} tokens")
    print(f"  System Instructions:   {tb.system_tokens:>6} tokens")
    print(f"  Query Template:        {tb.query_template_overhead_tokens:>6} tokens")
    print(f"  Context Allowed:       {tb.available_context_tokens:>6} tokens")
    print(f"  Context Injected:      {tb.used_context_tokens:>6} tokens ({tb.chunks_injected}/{tb.chunks_considered} chunks)")
    print(f"  Total Prompt Tokens:   {tb.total_prompt_tokens:>6} tokens")
    print(f"  Remaining Headroom:    {tb.remaining_tokens:>6} tokens")
    print(f"  Window Utilization:    {tb.utilization_pct:>6.1f} %")
    print("\nINJECTED CHUNKS:")
    for c in result.injected_chunks:
        sim = f"sim={c.cosine_similarity:.4f}" if c.cosine_similarity else ""
        rr = f"re-rank={c.rerank_score:.4f}" if c.rerank_score else ""
        scores = f"  {sim}  {rr}".strip()
        print(f"  {c.marker} {c.chunk_id:<22} Page={c.page:<8} tokens={c.token_count:<4} {scores}")
        if c.heading:
            print(f"     Section: {c.heading}")
    print(bar)


# ---------------------------------------------------------------------------
# Synthetic Fallback Chunks (for testing / dry-runs without ChromaDB)
# ---------------------------------------------------------------------------


def get_synthetic_test_chunks() -> list[dict[str, Any]]:
    """Return realistic NBC fire safety test chunks for standalone execution."""
    return [
        {
            "id": "nbc-v1-p314-001012",
            "text": (
                "120 min fire resistance rating wall. Such areas shall be provided with fire doors, "
                "to be kept closed and shall be posted with a sign on each side. All exterior openings "
                "in a boiler room or rooms containing central heating equipment, if located below an "
                "opening in another storey or if less than 3 m from other doors or windows of the same "
                "building, shall be protected by a fire assembly of not less than 2 hours rating."
            ),
            "cosine_similarity": 0.8842,
            "rerank_score": 4.0295,
            "metadata": {
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page_start": 314,
                "page_end": 315,
                "clause": "6.1.1.2",
                "heading": "Part 4 — Fire and Life Safety",
            },
        },
        {
            "id": "nbc-v1-p279-000915",
            "text": (
                "3.4.8.1 Air conditioning and mechanical ventilation requirements of different rooms "
                "or areas in any occupancy shall be as given in Part 8. The boilers shall be installed in "
                "a fire resisting room of 180 min fire resistance rating when installed in basement or "
                "hazardous locations, and 120 min fire resistance rating in other locations. The door "
                "shall be provided with 120 min fire door."
            ),
            "cosine_similarity": 0.8120,
            "rerank_score": 3.7279,
            "metadata": {
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page_start": 279,
                "page_end": 280,
                "clause": "3.4.8.1",
                "heading": "Part 4 — Fire and Life Safety",
            },
        },
        {
            "id": "nbc-v1-p320-001031",
            "text": (
                "The areas having storage, manufacturing, hazardous activities such as paint store, "
                "boiler room, fuel storage, spray booths shall be separated from non-hazardous areas "
                "like administrative offices, staff canteen by fire rated walls and fire doors of "
                "120 min fire resistance rating. Moderate and high hazard areas to have two fire doors."
            ),
            "cosine_similarity": 0.7715,
            "rerank_score": 3.2248,
            "metadata": {
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page_start": 320,
                "page_end": 321,
                "clause": "6.7.2",
                "heading": "Part 4 — Fire and Life Safety",
            },
        },
        {
            "id": "nbc-v1-p290-000940",
            "text": (
                "Shafts, chutes and vertical service ducts shall have fire resistance walls of not less "
                "than 120 min. Electrical service shafts shall be provided with inspection doors of 60 min "
                "fire resistance rating. Gas pipeline installations shall not pass through electrical shafts."
            ),
            "cosine_similarity": 0.6920,
            "rerank_score": 1.8402,
            "metadata": {
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page_start": 290,
                "page_end": 291,
                "clause": "3.4.10",
                "heading": "Part 4 — Fire and Life Safety",
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BuildGuard RAG — Context Injection & Prompt Augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--query", "-q",
        default=DEFAULT_SAMPLE_QUERY,
        help="Compliance question to assemble context for.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOTAL_TOKENS,
        help=f"Maximum total context window budget (default: {DEFAULT_MAX_TOTAL_TOKENS}).",
    )
    p.add_argument(
        "--reserved-answer-tokens", type=int, default=DEFAULT_RESERVED_ANSWER_TOKENS,
        help=f"Tokens reserved for model generation (default: {DEFAULT_RESERVED_ANSWER_TOKENS}).",
    )
    p.add_argument(
        "--candidate-k", type=int, default=6,
        help="Number of retrieved candidates to consider before budgeting (default: 6).",
    )
    p.add_argument(
        "--marker-style", default="numeric", choices=["numeric", "verbose"],
        help="Marker style: 'numeric' -> [1], 'verbose' -> [Source 1].",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Use realistic synthetic chunks instead of querying ChromaDB.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to persist sample augmented prompt outputs.",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save output artifacts to disk.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    chunks: list[dict[str, Any]] = []

    if args.dry_run:
        logger.info("Using synthetic test chunks (dry-run mode).")
        chunks = get_synthetic_test_chunks()
    else:
        try:
            logger.info("Retrieving candidates from ChromaDB for query: %r", args.query[:60])
            from src.embeddings.filtered_retrieval import retrieve_filtered
            col_name = _safe_collection_name(DEFAULT_COLLECTION)
            raw = retrieve_filtered(
                query=args.query,
                top_k=args.candidate_k,
                hybrid=True,
                persist_dir=DEFAULT_PERSIST_DIR,
                collection_name=col_name,
                model_name=DEFAULT_MODEL,
            )
            chunks = raw.get("results", [])
            logger.info("Retrieved %d candidates from vector store.", len(chunks))
        except Exception as exc:
            logger.warning("ChromaDB retrieval failed (%s); falling back to synthetic test chunks.", exc)
            chunks = get_synthetic_test_chunks()

    result = build_augmented_prompt(
        args.query,
        chunks,
        max_total_tokens=args.max_tokens,
        reserved_answer_tokens=args.reserved_answer_tokens,
        marker_style=args.marker_style,
    )

    print_prompt_summary(result)

    if not args.no_save:
        json_path, md_path = save_augmented_prompt(result, output_dir=args.output_dir, label="sample")
        print(f"\nSample augmented prompt persisted to:\n  - {json_path}\n  - {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
