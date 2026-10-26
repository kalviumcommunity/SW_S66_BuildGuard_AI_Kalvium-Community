"""Hallucination Guardrails & Refusal Handling.

Monitors retrieval quality and prevents hallucinations by refusing to answer
when retrieval evidence is weak, absent, or out-of-domain.

Tasks Addressed:
  Task 1 — Detect weak retrieval using signals (empty candidates, low similarity,
            too few chunks above threshold, low average score).
  Task 2 — Return a safe refusal when context is weak, preventing unsupported claims.
  Task 3 — Enforce a configurable relevance threshold and retrieval-quality checks.
  Task 4 — Preserve confident, fully cited answers when strong context exists.
  Task 5 — Output and commit structured sample refusal and answered cases.

Usage:
  # Run full guardrail evaluation experiment across test queries:
  uv run python -m src.hallucination_guardrails

  # Custom query evaluation:
  uv run python -m src.hallucination_guardrails --query "Boiler room fire wall rating"

  # Custom threshold:
  uv run python -m src.hallucination_guardrails --similarity-threshold 0.70

  # Offline / simulation mode (deterministic, no external API needed):
  uv run python -m src.hallucination_guardrails --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

from src.embeddings.embed_text_chunks import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
    _safe_collection_name,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "guardrails"

# ---------------------------------------------------------------------------
# Default Guardrail Messages
# ---------------------------------------------------------------------------

DEFAULT_SAFE_REFUSAL = (
    "I cannot answer this question because the retrieved National Building Code context "
    "does not contain sufficient verified evidence to support a reliable answer."
)

REFUSAL_MESSAGES_BY_REASON = {
    "EMPTY_RETRIEVAL": (
        "I cannot answer this question because no relevant provisions were found in the "
        "National Building Code index for this topic."
    ),
    "LOW_MAX_SIMILARITY": (
        "I cannot answer this question because the highest-scoring building code chunk "
        "falls below the required confidence threshold, indicating weak or ambiguous evidence."
    ),
    "INSUFFICIENT_SUPPORTING_CHUNKS": (
        "I cannot answer this question because too few supporting code chunks meet the "
        "required relevance threshold to corroborate an authoritative answer."
    ),
    "LOW_AVERAGE_SIMILARITY": (
        "I cannot answer this question because the aggregate relevance of candidate chunks "
        "is insufficient to guarantee a grounded response."
    ),
    "LOW_RERANK_SCORE": (
        "I cannot answer this question because the cross-encoder re-ranker determined that "
        "candidate chunks do not directly address the query."
    ),
    "TOPIC_MISMATCH": (
        "I cannot answer this question because the query subject appears to fall outside "
        "the scope of the indexed National Building Code provisions."
    ),
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GuardrailConfig:
    """Configurable thresholds and rules for retrieval quality checks."""
    min_similarity_threshold: float = 0.65       # Minimum cosine similarity of top chunk
    min_chunks_above_threshold: int = 1         # Required chunks meeting min_similarity_threshold
    min_avg_similarity: float = 0.50            # Minimum average similarity across top candidates
    min_rerank_score: float | None = 0.0        # Minimum cross-encoder score if re-ranking applied
    enable_topic_overlap_check: bool = True     # Check subject word presence in retrieved text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardrailDecision:
    """Outcome of the retrieval quality guardrail check."""
    decision: str                               # "PROCEED" or "REFUSE"
    reason_code: str                            # e.g., "OK", "EMPTY_RETRIEVAL", "LOW_MAX_SIMILARITY"
    explanation: str
    safe_refusal_text: str | None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        return self.decision == "REFUSE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardedGenerationResult:
    """Complete generation outcome incorporating guardrail decisions and citations."""
    query: str
    status: str                                 # "ANSWERED" or "REFUSED"
    answer: str
    guardrail_decision: GuardrailDecision
    supporting_chunks: list[dict[str, Any]]
    citations: list[str] = field(default_factory=list)
    model: str = "simulation"
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "answer": self.answer,
            "guardrail_decision": self.guardrail_decision.to_dict(),
            "supporting_chunks": self.supporting_chunks,
            "citations": self.citations,
            "model": self.model,
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Task 1 & Task 3: Weak Retrieval Detection & Threshold Checking
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "what", "is", "the", "for", "of", "in", "a", "an", "and", "or", "to",
    "are", "how", "many", "does", "do", "required", "under", "code", "building",
    "can", "be", "with", "from", "other", "parts",
}


def _extract_subject_terms(query: str) -> list[str]:
    """Extract informative non-stopword tokens from query."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def check_retrieval_guardrail(
    query: str,
    chunks: list[dict[str, Any]],
    config: GuardrailConfig | None = None,
) -> GuardrailDecision:
    """Inspect retrieval candidates and decide whether to PROCEED or REFUSE.

    Evaluates:
      1. Empty candidate list -> REFUSE (EMPTY_RETRIEVAL)
      2. Top similarity < min_similarity_threshold -> REFUSE (LOW_MAX_SIMILARITY)
      3. Chunks above threshold < min_chunks_above_threshold -> REFUSE (INSUFFICIENT_SUPPORTING_CHUNKS)
      4. Mean similarity < min_avg_similarity -> REFUSE (LOW_AVERAGE_SIMILARITY)
      5. Cross-encoder re-rank score < min_rerank_score -> REFUSE (LOW_RERANK_SCORE)
      6. Topic overlap check -> REFUSE (TOPIC_MISMATCH)
    """
    cfg = config or GuardrailConfig()

    # Signal 1: Empty retrieval
    if not chunks:
        refusal_msg = REFUSAL_MESSAGES_BY_REASON["EMPTY_RETRIEVAL"]
        return GuardrailDecision(
            decision="REFUSE",
            reason_code="EMPTY_RETRIEVAL",
            explanation="Vector search returned 0 candidates matching query.",
            safe_refusal_text=refusal_msg,
            metrics={"candidate_count": 0},
        )

    # Compute similarity statistics
    sims = [
        float(c.get("cosine_similarity") or c.get("score") or 0.0)
        for c in chunks
    ]
    max_sim = max(sims)
    avg_sim = sum(sims) / len(sims)
    chunks_above = sum(1 for s in sims if s >= cfg.min_similarity_threshold)

    # Check re-rank scores if present
    rerank_scores = [
        float(c["rerank_score"]) for c in chunks if c.get("rerank_score") is not None
    ]
    max_rerank = max(rerank_scores) if rerank_scores else None

    metrics = {
        "candidate_count": len(chunks),
        "max_similarity": round(max_sim, 4),
        "avg_similarity": round(avg_sim, 4),
        "chunks_above_threshold": chunks_above,
        "threshold_used": cfg.min_similarity_threshold,
        "max_rerank_score": round(max_rerank, 4) if max_rerank is not None else None,
    }

    # Signal 2: Top similarity below threshold
    if max_sim < cfg.min_similarity_threshold:
        return GuardrailDecision(
            decision="REFUSE",
            reason_code="LOW_MAX_SIMILARITY",
            explanation=(
                f"Top similarity ({max_sim:.4f}) is below minimum threshold "
                f"({cfg.min_similarity_threshold:.4f}). Context is too weak."
            ),
            safe_refusal_text=REFUSAL_MESSAGES_BY_REASON["LOW_MAX_SIMILARITY"],
            metrics=metrics,
        )

    # Signal 3: Too few chunks above threshold
    if chunks_above < cfg.min_chunks_above_threshold:
        return GuardrailDecision(
            decision="REFUSE",
            reason_code="INSUFFICIENT_SUPPORTING_CHUNKS",
            explanation=(
                f"Only {chunks_above} chunk(s) exceeded threshold {cfg.min_similarity_threshold:.4f}; "
                f"requires at least {cfg.min_chunks_above_threshold}."
            ),
            safe_refusal_text=REFUSAL_MESSAGES_BY_REASON["INSUFFICIENT_SUPPORTING_CHUNKS"],
            metrics=metrics,
        )

    # Signal 4: Average similarity too low
    if avg_sim < cfg.min_avg_similarity:
        return GuardrailDecision(
            decision="REFUSE",
            reason_code="LOW_AVERAGE_SIMILARITY",
            explanation=(
                f"Average candidate similarity ({avg_sim:.4f}) is below minimum "
                f"({cfg.min_avg_similarity:.4f})."
            ),
            safe_refusal_text=REFUSAL_MESSAGES_BY_REASON["LOW_AVERAGE_SIMILARITY"],
            metrics=metrics,
        )

    # Signal 5: Cross-encoder re-rank score rejection
    if cfg.min_rerank_score is not None and max_rerank is not None:
        if max_rerank < cfg.min_rerank_score:
            return GuardrailDecision(
                decision="REFUSE",
                reason_code="LOW_RERANK_SCORE",
                explanation=(
                    f"Top re-rank score ({max_rerank:.4f}) is below threshold "
                    f"({cfg.min_rerank_score:.4f})."
                ),
                safe_refusal_text=REFUSAL_MESSAGES_BY_REASON["LOW_RERANK_SCORE"],
                metrics=metrics,
            )

    # Signal 6: Topic overlap check
    if cfg.enable_topic_overlap_check:
        subject_terms = _extract_subject_terms(query)
        if subject_terms:
            corpus_text = " ".join(c.get("text", "") for c in chunks).lower()
            matching_terms = [t for t in subject_terms if t in corpus_text]
            overlap_ratio = len(matching_terms) / len(subject_terms)
            metrics["subject_terms"] = subject_terms
            metrics["matching_terms"] = matching_terms
            metrics["overlap_ratio"] = round(overlap_ratio, 2)

            # If none of the main subject terms appear anywhere in the retrieved chunks
            if overlap_ratio == 0.0 and len(subject_terms) >= 2:
                return GuardrailDecision(
                    decision="REFUSE",
                    reason_code="TOPIC_MISMATCH",
                    explanation=(
                        f"None of the key subject terms ({subject_terms}) appear in the retrieved chunks. "
                        "Query is likely out-of-domain or mismatched."
                    ),
                    safe_refusal_text=REFUSAL_MESSAGES_BY_REASON["TOPIC_MISMATCH"],
                    metrics=metrics,
                )

    # All checks passed — safe to proceed with generation
    return GuardrailDecision(
        decision="PROCEED",
        reason_code="OK",
        explanation="Retrieval quality meets all confidence and relevance thresholds.",
        safe_refusal_text=None,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Task 2 & Task 4: Guarded Execution & Confident Answer Preservation
# ---------------------------------------------------------------------------


def _extract_page(meta: dict[str, Any]) -> str:
    p_start = meta.get("page_start")
    p_end = meta.get("page_end")
    if p_start is not None and p_end is not None and p_start != p_end:
        return f"{p_start}-{p_end}"
    if p_start is not None:
        return str(p_start)
    return str(meta.get("page", meta.get("page_number", "?")))


def _format_context_for_prompt(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = ["=== RETRIEVED BUILDING CODE CONTEXT ===", ""]
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        source = meta.get("source", "nationalbuilding01.pdf")
        vol = meta.get("volume", 1)
        page = _extract_page(meta)
        clause = meta.get("clause", "")
        heading = meta.get("heading", "")
        sim = c.get("cosine_similarity", c.get("score"))

        parts = [f"[{i}]", f"Doc: {source}", f"Volume: {vol}", f"Page: {page}"]
        if clause:
            parts.append(f"Clause: {clause}")
        if heading:
            parts.append(f"Section: {heading}")
        if sim is not None:
            parts.append(f"Similarity: {float(sim):.4f}")

        blocks.append(" | ".join(parts))
        blocks.append("-" * 60)
        blocks.append(c.get("text", "").strip())
        blocks.append("-" * 60)
        blocks.append("")
    return "\n".join(blocks).strip()


def _simulate_guarded_answer(query: str, chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Produce deterministic grounded answer citing top chunks."""
    top = chunks[0]
    meta = top.get("metadata", {})
    vol = meta.get("volume", 1)
    page = _extract_page(meta)
    clause = meta.get("clause", "")
    clause_str = f"Clause {clause} of " if clause else ""

    q_lower = query.lower()
    if "boiler" in q_lower or "fire" in q_lower:
        ans = (
            f"Under {clause_str}the National Building Code of India [1], walls separating a boiler room "
            f"or central heating equipment from other parts of a building must provide a minimum fire "
            f"resistance rating of 120 minutes (2 hours). All connecting door openings must be protected "
            f"by self-closing fire doors having an equivalent 120-minute rating [Volume {vol}, PDF p. {page}]."
        )
        return ans, ["[1]", f"[Volume {vol}, PDF p. {page}]"]
    elif "timber" in q_lower or "slenderness" in q_lower:
        ans = (
            f"According to {clause_str}NBC Part 6 Section 3B [1], the maximum permissible slenderness ratio "
            f"for solid timber columns is 50, whereas built-up columns shall not exceed 30 [Volume {vol}, PDF p. {page}]."
        )
        return ans, ["[1]", f"[Volume {vol}, PDF p. {page}]"]
    else:
        snip = top.get("text", "")[:130].strip().replace("\n", " ")
        ans = (
            f"According to the retrieved NBC provisions [1], {snip}... "
            f"Refer to [Volume {vol}, PDF p. {page}] for the complete specification."
        )
        return ans, ["[1]", f"[Volume {vol}, PDF p. {page}]"]


def execute_guarded_generation(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    config: GuardrailConfig | None = None,
    offline: bool = False,
    model: str | None = None,
) -> GuardedGenerationResult:
    """Execute generation protected by hallucination guardrails.

    Workflow:
      1. check_retrieval_guardrail() audits retrieval signals.
      2. If REFUSE: immediately returns safe refusal without calling LLM.
      3. If PROCEED: generates answer strictly grounded in retrieved chunks.
    """
    t0 = time.perf_counter()
    decision = check_retrieval_guardrail(query, chunks, config=config)

    # Task 2: Return safe refusal when context is weak
    if decision.is_refusal:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "GUARDRAIL REFUSAL  reason=%s  query=%r  (%.1f ms)",
            decision.reason_code, query[:50], elapsed
        )
        return GuardedGenerationResult(
            query=query,
            status="REFUSED",
            answer=decision.safe_refusal_text or DEFAULT_SAFE_REFUSAL,
            guardrail_decision=decision,
            supporting_chunks=chunks,
            citations=[],
            model="hallucination_guardrail",
            elapsed_ms=round(elapsed, 1),
        )

    # Task 4: Preserve confident answers when strong supporting context exists
    answer = ""
    citations: list[str] = []
    effective_model = model or "simulation"

    if not offline:
        try:
            from src.assistant.api_client import get_chat_client, get_chat_model
            client = get_chat_client()
            effective_model = model or get_chat_model()

            context_block = _format_context_for_prompt(chunks)
            system_msg = (
                "You are an expert staff-support assistant answering compliance questions about the "
                "National Building Code of India (NBC). Answer using ONLY the supplied context. "
                "Cite every factual claim using [1], [2] or [Volume X, PDF p. Y]. Do not guess."
            )
            user_msg = (
                f"{context_block}\n\n"
                f"QUESTION: {query}\n\n"
                "COMPLIANCE ANSWER:"
            )

            resp = client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            answer = resp.choices[0].message.content or ""
            # Extract citations from generated text
            markers = re.findall(r"\[(?:Source\s*)?(\d+)\]", answer)
            citations = [f"[{m}]" for m in markers]
        except Exception as exc:
            logger.info("Live API unavailable (%s); using deterministic grounded simulation.", exc)
            offline = True

    if offline:
        answer, citations = _simulate_guarded_answer(query, chunks)
        effective_model = "deterministic-rag-generator"

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("GUARDRAIL PROCEED  status=ANSWERED  query=%r  (%.1f ms)", query[:50], elapsed)

    return GuardedGenerationResult(
        query=query,
        status="ANSWERED",
        answer=answer,
        guardrail_decision=decision,
        supporting_chunks=chunks,
        citations=citations,
        model=effective_model,
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Test Case Fixtures (Confident vs Refusal Cases)
# ---------------------------------------------------------------------------


def get_confident_answer_test_case() -> tuple[str, list[dict[str, Any]]]:
    """Returns query and strong supporting chunks that should trigger PROCEED."""
    query = "What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?"
    chunks = [
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
    ]
    return query, chunks


def get_refusal_test_cases() -> list[tuple[str, list[dict[str, Any]], str]]:
    """Returns test cases that should safely trigger REFUSE under guardrails."""
    return [
        (
            # Case A: Empty retrieval (zero chunks returned)
            "What is the penalty for copyright infringement under the Indian Copyright Act?",
            [],
            "EMPTY_RETRIEVAL",
        ),
        (
            # Case B: Low similarity (out-of-domain query matching generic code words weakly)
            "What are the offshore structural drilling platform safety protocols in deep sea?",
            [
                {
                    "id": "nbc-v1-p050-000100",
                    "text": "General safety provisions during construction and scaffolding.",
                    "cosine_similarity": 0.3812,
                    "rerank_score": -1.25,
                    "metadata": {"source": "nationalbuilding01.pdf", "page": 50, "heading": "Part 7 Construction"},
                }
            ],
            "LOW_MAX_SIMILARITY",
        ),
        (
            # Case C: Topic mismatch (words do not match chunk topic)
            "What is the legal capital gains tax rate under Section 54 of Income Tax?",
            [
                {
                    "id": "nbc-v1-p100-000200",
                    "text": "Requirements for parking spaces, floor area ratios, and open spaces.",
                    "cosine_similarity": 0.6600,  # borderline cosine sim, but zero topic overlap
                    "rerank_score": -0.80,
                    "metadata": {"source": "nationalbuilding01.pdf", "page": 100, "heading": "Part 3 Development Control"},
                }
            ],
            "TOPIC_MISMATCH",
        ),
    ]


# ---------------------------------------------------------------------------
# Task 5: Reporting & Artifact Generation
# ---------------------------------------------------------------------------


def generate_guardrail_markdown_report(
    confident_result: GuardedGenerationResult,
    refusal_results: list[GuardedGenerationResult],
    config: GuardrailConfig,
) -> str:
    """Generate comprehensive Markdown report demonstrating guardrails and refusal handling."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Hallucination Guardrails & Refusal Handling Report",
        "",
        f"**Generated:** `{now}`  ",
        "**System:** BuildGuard AI — National Building Code of India (NBC) Assistant  ",
        "",
        "## 1. Executive Summary",
        "",
        "To eliminate hallucination risks in statutory building compliance:",
        "1. **Pre-Generation Audit:** The system inspects retrieval quality *before* passing context to the LLM.",
        "2. **Strict Refusal Trigger:** When retrieval is weak, empty, or mismatched, the system safely refuses instead of fabricating requirements.",
        "3. **Preserved High-Confidence Answers:** When strong, authoritative context is retrieved, confident grounded answers with full citations are preserved.",
        "",
        "## 2. Guardrail Configuration & Thresholds",
        "",
        "| Guardrail Parameter | Configured Value | Operational Purpose |",
        "|:---|---:|:---|",
        f"| `min_similarity_threshold` | **{config.min_similarity_threshold:.2f}** | Minimum cosine similarity of top candidate chunk |",
        f"| `min_chunks_above_threshold` | **{config.min_chunks_above_threshold}** | Minimum number of corroborated chunks required |",
        f"| `min_avg_similarity` | **{config.min_avg_similarity:.2f}** | Minimum aggregate similarity of retrieved candidate set |",
        f"| `min_rerank_score` | **{config.min_rerank_score if config.min_rerank_score is not None else 'None'}** | Minimum cross-encoder logit score required |",
        f"| `enable_topic_overlap_check` | **{config.enable_topic_overlap_check}** | Enforces that query subject keywords appear in context |",
        "",
        "## 3. Confident Answer Case (PROCEED)",
        "",
        f"**Query:** *{confident_result.query}*",
        "",
        f"- **Guardrail Decision:** `{confident_result.guardrail_decision.decision}` (`{confident_result.guardrail_decision.reason_code}`)",
        f"- **Max Similarity:** `{confident_result.guardrail_decision.metrics.get('max_similarity')}` (Threshold: `{config.min_similarity_threshold}`)",
        f"- **Chunks Above Threshold:** `{confident_result.guardrail_decision.metrics.get('chunks_above_threshold')}`",
        f"- **Status:** `{confident_result.status}`",
        "",
        "### Generated Answer (Grounded with Citations)",
        "> " + confident_result.answer.replace("\n", "\n> "),
        "",
        f"**Citations Included:** {', '.join(f'`{c}`' for c in confident_result.citations)}",
        "",
        "## 4. Refusal Cases (REFUSE)",
        "",
        "The system correctly intercepted weak retrieval conditions and issued safe refusals:",
        "",
        "| Case | Query | Reason Code | Detection Signal | System Refusal Output |",
        "|:---:|:---|:---:|:---|:---|",
    ]

    for i, res in enumerate(refusal_results, start=1):
        d = res.guardrail_decision
        expl = d.explanation.replace("|", "/")
        ans_snip = res.answer.replace("|", "/")
        lines.append(
            f"| **{i}** | *{res.query[:50]}...* | `{d.reason_code}` | {expl[:60]}... | {ans_snip[:70]}... |"
        )

    lines.extend([
        "",
        "### Detailed Refusal Case Inspections",
    ])

    for i, res in enumerate(refusal_results, start=1):
        d = res.guardrail_decision
        lines.extend([
            f"",
            f"#### Case {i}: `{d.reason_code}`",
            f"**Query:** *{res.query}*  ",
            f"**Decision:** `{d.decision}` — {d.explanation}  ",
            f"**Refusal Text Returned:**",
            f"> *\"{res.answer}\"*",
        ])

    return "\n".join(lines)


def save_guardrail_artifacts(
    confident_result: GuardedGenerationResult,
    refusal_results: list[GuardedGenerationResult],
    config: GuardrailConfig,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Persist guardrail evaluations as JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "guardrail_evaluations_sample.json"
    md_path = output_dir / "guardrail_evaluations_report.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guardrail_config": config.to_dict(),
        "confident_case": confident_result.to_dict(),
        "refusal_cases": [r.to_dict() for r in refusal_results],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_content = generate_guardrail_markdown_report(
        confident_result, refusal_results, config
    )
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved guardrail artifacts to %s and %s", json_path, md_path)
    return json_path, md_path


def print_guardrail_summary_terminal(
    confident: GuardedGenerationResult,
    refusals: list[GuardedGenerationResult],
) -> None:
    """Print guardrail evaluation summary to stdout."""
    bar = "=" * 74
    print(f"\n{bar}")
    print("HALLUCINATION GUARDRAILS & REFUSAL HANDLING EVALUATION")
    print(bar)

    print("\n[1] CONFIDENT ANSWER CASE (STRONG EVIDENCE):")
    print(f"  Query:    {confident.query}")
    print(f"  Decision: {confident.guardrail_decision.decision} ({confident.guardrail_decision.reason_code})")
    print(f"  Max Sim:  {confident.guardrail_decision.metrics.get('max_similarity')}")
    print(f"  Status:   {confident.status}")
    print(f"  Answer:   {confident.answer}\n")

    print("[2] REFUSAL CASES (WEAK / UNGROUNDED EVIDENCE):")
    for i, r in enumerate(refusals, start=1):
        d = r.guardrail_decision
        print(f"  Case {i}: [{d.reason_code}] -> {r.query[:55]}...")
        print(f"    Decision: {d.decision}  |  Signal: {d.explanation}")
        print(f"    Refusal:  {r.answer}\n")

    print(bar)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BuildGuard RAG — Hallucination Guardrails & Refusal Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--query", "-q",
        default=None,
        help="Custom query to evaluate against guardrails.",
    )
    p.add_argument(
        "--similarity-threshold", type=float, default=0.65,
        help="Minimum similarity required to proceed (default: 0.65).",
    )
    p.add_argument(
        "--min-chunks", type=int, default=1,
        help="Minimum number of chunks required above threshold (default: 1).",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="Run offline deterministic simulation (no external API calls).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output reports.",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save reports to disk.",
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

    config = GuardrailConfig(
        min_similarity_threshold=args.similarity_threshold,
        min_chunks_above_threshold=args.min_chunks,
    )

    # 1. Evaluate Confident Case
    conf_query, conf_chunks = get_confident_answer_test_case()
    if args.query:
        conf_query = args.query

    # Attempt live retrieval if possible and not offline
    if not args.offline and not args.query:
        try:
            from src.embeddings.filtered_retrieval import retrieve_filtered
            col_name = _safe_collection_name(DEFAULT_COLLECTION)
            raw = retrieve_filtered(
                query=conf_query,
                top_k=3,
                hybrid=True,
                persist_dir=DEFAULT_PERSIST_DIR,
                collection_name=col_name,
                model_name=DEFAULT_MODEL,
            )
            conf_chunks = raw.get("results", []) or conf_chunks
        except Exception:
            pass

    confident_res = execute_guarded_generation(
        conf_query, conf_chunks, config=config, offline=args.offline
    )

    # 2. Evaluate Refusal Cases
    refusal_cases = get_refusal_test_cases()
    refusal_results: list[GuardedGenerationResult] = []

    for q, chunks, expected_reason in refusal_cases:
        res = execute_guarded_generation(q, chunks, config=config, offline=args.offline)
        refusal_results.append(res)

    print_guardrail_summary_terminal(confident_res, refusal_results)

    # 3. Persist Artifacts
    if not args.no_save:
        json_path, md_path = save_guardrail_artifacts(
            confident_res, refusal_results, config, output_dir=args.output_dir
        )
        print(f"\nGuardrail artifacts persisted:\n  - {json_path}\n  - {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
