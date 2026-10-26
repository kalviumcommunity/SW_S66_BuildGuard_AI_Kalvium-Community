"""RAG Pipeline Architecture & Flow.

Connects every stage of the BuildGuard RAG system into a single coherent
query-to-answer pipeline:

    User Query
        │
        ▼
    Stage 1 — EMBED
        Embed the query using the same SentenceTransformer model that was used
        to index the corpus, producing a normalised dense vector.
        │
        ▼
    Stage 2 — RETRIEVE
        Run hybrid retrieval (vector ANN search + optional lexical keyword
        boosting) against the persistent ChromaDB collection, optionally
        scoped by a metadata heading/section filter.  Returns a broad
        candidate set (candidate_k > final_k).
        │
        ▼
    Stage 3 — RERANK
        Re-score and reorder the candidate set using a CrossEncoder
        (cross-attention, full text pair scoring) to promote the most
        directly relevant chunks to the top.  Keeps only the final top-k.
        │
        ▼
    Stage 4 — ASSEMBLE
        Concatenate the re-ranked chunk texts in order, annotate each with
        its source (PDF filename, page, heading, rank), and produce a
        structured grounded-context block for the LLM prompt.
        │
        ▼
    Stage 5 — GENERATE
        Call the OpenAI-compatible Chat API with the system prompt, the
        assembled context, and the original user query.  The model is
        instructed to answer only from the supplied NBC context and cite
        volume + PDF page for every factual claim.
        │
        ▼
    Pipeline Result
        answer      — LLM-generated, grounded answer string
        sources     — list of source dicts (id, page, heading, similarity…)
        context     — assembled context block fed to the model
        stages      — timing + metadata for every pipeline stage

Tasks Addressed:
  Task 1 — Full query-to-answer flow documented above and in RAG_FLOW.md.
  Task 2 — Clear code stages: embed(), retrieve(), rerank(), assemble(), generate().
  Task 3 — end-to-end run via __main__ / run_pipeline() with sample output saved.
  Task 4 — Separated, independently testable functions; no monolithic block.
  Task 5 — Committed with flow description (RAG_FLOW.md) and sample JSON output.

Usage
-----
# Run with the default sample query and save sample output:
uv run python -m src.rag_pipeline

# Custom query:
uv run python -m src.rag_pipeline --query "What is the fire rating for boiler room walls?"

# Dry-run (no LLM call, no model download — for CI / unit-testing):
uv run python -m src.rag_pipeline --dry-run

# Adjust retrieval parameters:
uv run python -m src.rag_pipeline \\
    --query "Timber column slenderness ratio" \\
    --filter-heading "Part 6 — Structural Design, Section 3A — Timber" \\
    --candidate-k 10 \\
    --final-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Re-use defaults from the existing embedding layer so settings stay in sync.
from src.embeddings.embed_text_chunks import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
    _safe_collection_name,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "outputs" / "rag_pipeline"

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATE_K = 10   # broad retrieval pool sent to re-ranker
DEFAULT_FINAL_K = 3        # chunks kept after re-ranking for context
DEFAULT_RERANK_METHOD = "cross_encoder"
DEFAULT_SAMPLE_QUERY = (
    "What is the minimum fire resistance rating required for walls separating "
    "a boiler room from other parts of a building?"
)
_RAG_SYSTEM_PROMPT = (
    "You are a staff-support assistant answering questions about the National "
    "Building Code of India (NBC). Use ONLY the retrieved NBC chunks supplied "
    "in the CONTEXT block; do not invent requirements, exceptions, deadlines, "
    "contacts, or facts. Cite every factual NBC claim using the volume and PDF "
    "page number supplied with each chunk, for example [Volume 1, PDF p. 42]. "
    "If the retrieved chunks do not support an answer, say: "
    "'I don't have enough information to answer that reliably.' "
    "Respond professionally and clearly in no more than three short sentences."
)

# ---------------------------------------------------------------------------
# Stage Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmbedResult:
    """Output of Stage 1 — EMBED."""
    query: str
    vector: list[float]
    model_name: str
    elapsed_ms: float


@dataclass
class RetrieveResult:
    """Output of Stage 2 — RETRIEVE."""
    query: str
    candidates: list[dict[str, Any]]   # raw chunk dicts from the vector store
    candidate_k: int
    metadata_filter: dict[str, Any] | None
    hybrid: bool
    elapsed_ms: float


@dataclass
class RerankResult:
    """Output of Stage 3 — RERANK."""
    chunks: list[dict[str, Any]]       # re-ranked, trimmed to final_k
    method: str
    final_k: int
    elapsed_ms: float


@dataclass
class AssembleResult:
    """Output of Stage 4 — ASSEMBLE."""
    context_block: str                 # formatted text sent to the LLM
    sources: list[dict[str, Any]]      # structured source citations
    chunk_count: int
    elapsed_ms: float


@dataclass
class GenerateResult:
    """Output of Stage 5 — GENERATE."""
    answer: str
    model: str
    elapsed_ms: float
    raw_response: str = ""


@dataclass
class PipelineResult:
    """Complete end-to-end pipeline result."""
    query: str
    answer: str
    sources: list[dict[str, Any]]
    context_block: str
    stages: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    pipeline_elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1 — EMBED
# ---------------------------------------------------------------------------


def embed(
    query: str,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> EmbedResult:
    """Embed *query* using the same SentenceTransformer used to index chunks.

    The resulting dense vector is L2-normalised so cosine similarity equals
    dot product — matching the indexing convention in embed_text_chunks.py.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    t0 = time.perf_counter()
    from src.embeddings.retrieve import _embed_query  # local: keeps startup fast

    vector = _embed_query(query, model_name=model_name, device=device)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info("EMBED  query=%r  model=%s  dim=%d  (%.0f ms)", query[:60], model_name, len(vector), elapsed)
    return EmbedResult(query=query, vector=vector, model_name=model_name, elapsed_ms=round(elapsed, 1))


# ---------------------------------------------------------------------------
# Stage 2 — RETRIEVE
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    metadata_filter: dict[str, Any] | None = None,
    hybrid: bool = True,
    keywords: list[str] | None = None,
    score_threshold: float | None = None,
    model_name: str = DEFAULT_MODEL,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    device: str | None = None,
) -> RetrieveResult:
    """Retrieve *candidate_k* chunks from ChromaDB using hybrid search.

    Uses filtered_retrieval.retrieve_filtered() which supports:
    - Dense vector search (always on)
    - Metadata heading/section filter (when metadata_filter is given)
    - Lexical keyword boosting (when hybrid=True and keywords are given)

    Returns a broad candidate set; the re-ranker will narrow it to final_k.
    """
    t0 = time.perf_counter()
    col_name = _safe_collection_name(collection_name)

    from src.embeddings.filtered_retrieval import retrieve_filtered  # local import

    raw = retrieve_filtered(
        query=query,
        where=metadata_filter,
        top_k=candidate_k,
        hybrid=hybrid,
        exact_keywords=keywords,
        model_name=model_name,
        persist_dir=persist_dir,
        collection_name=col_name,
    )
    candidates = raw.get("results", [])
    if score_threshold is not None:
        candidates = [
            c for c in candidates
            if (c.get("cosine_similarity") or 0.0) >= score_threshold
        ]
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "RETRIEVE  candidates=%d/%d  filter=%s  hybrid=%s  (%.0f ms)",
        len(candidates), candidate_k, bool(metadata_filter), hybrid, elapsed,
    )
    return RetrieveResult(
        query=query,
        candidates=candidates,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        hybrid=hybrid,
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Stage 3 — RERANK
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    final_k: int = DEFAULT_FINAL_K,
    method: str = DEFAULT_RERANK_METHOD,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str | None = None,
) -> RerankResult:
    """Re-rank *candidates* and return the top *final_k* most relevant chunks.

    Delegates to rerank.rerank_candidates() which supports:
    - 'cross_encoder': Full cross-attention scoring (CrossEncoder model)
    - 'custom_precision': Multi-factor scoring (coverage + phrase + heading)

    The re-ranker takes (query, candidate_text) pairs and assigns scores
    independent of the original cosine distance, placing the most
    directly relevant chunk at rank 1.
    """
    if not candidates:
        return RerankResult(chunks=[], method=method, final_k=final_k, elapsed_ms=0.0)

    t0 = time.perf_counter()
    from src.embeddings.rerank import rerank_candidates  # local import

    reranked = rerank_candidates(
        query=query,
        candidates=candidates,
        top_k=final_k,
        method=method,
        model_name=model_name,
        device=device,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "RERANK  method=%s  candidates=%d → final=%d  (%.0f ms)",
        method, len(candidates), len(reranked), elapsed,
    )
    return RerankResult(chunks=reranked, method=method, final_k=final_k, elapsed_ms=round(elapsed, 1))


# ---------------------------------------------------------------------------
# Stage 4 — ASSEMBLE
# ---------------------------------------------------------------------------

_CONTEXT_HEADER = "=== RETRIEVED BUILDING CODE CONTEXT ==="
_CHUNK_DIVIDER = "-" * 60


def assemble(chunks: list[dict[str, Any]]) -> AssembleResult:
    """Assemble re-ranked chunks into a grounded-context block for the LLM.

    Each chunk is annotated with its rank, source (PDF filename + page),
    section heading, and similarity / re-rank scores so the model can
    ground its citations precisely.

    The assembled block follows the format:
        === RETRIEVED BUILDING CODE CONTEXT ===
        [Chunk 1 of N]
        Source: nationalbuilding01.pdf | Page: 314 | Heading: Part 4 ...
        Similarity: 0.892 | Re-rank score: 0.9871
        ----
        <chunk text>
        ----
        ...
    """
    t0 = time.perf_counter()

    if not chunks:
        elapsed = (time.perf_counter() - t0) * 1000
        return AssembleResult(
            context_block="[No relevant chunks retrieved]",
            sources=[],
            chunk_count=0,
            elapsed_ms=round(elapsed, 1),
        )

    n = len(chunks)
    parts: list[str] = [_CONTEXT_HEADER, ""]
    sources: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "").strip()
        chunk_id = chunk.get("id", f"chunk-{i}")
        p_start = meta.get("page_start")
        p_end = meta.get("page_end")
        if p_start is not None and p_end is not None and p_start != p_end:
            page = f"{p_start}-{p_end}"
        elif p_start is not None:
            page = p_start
        else:
            page = meta.get("page", meta.get("page_number", "?"))
        clause = meta.get("clause", "")
        heading = meta.get("heading", "")
        source_file = meta.get("source", "nationalbuilding01.pdf")
        volume = meta.get("volume", "")
        sim = chunk.get("cosine_similarity", chunk.get("score", 0.0))
        rerank_score = chunk.get("rerank_score", None)
        rerank_rank = chunk.get("rerank_rank", i)

        # Build annotation line
        score_info = f"Similarity: {sim:.4f}"
        if rerank_score is not None:
            score_info += f" | Re-rank score: {rerank_score:.4f}"

        heading_str = f" | Heading: {heading}" if heading else ""
        volume_str = f"Volume {volume}, " if volume else ""

        parts += [
            f"[Chunk {i} of {n}]",
            f"Source: {source_file} | {volume_str}Page: {page}{heading_str}",
            score_info,
            _CHUNK_DIVIDER,
            text,
            _CHUNK_DIVIDER,
            "",
        ]

        sources.append({
            "rank": rerank_rank,
            "id": chunk_id,
            "source": source_file,
            "volume": volume,
            "page": page,
            "heading": heading,
            "cosine_similarity": sim,
            "rerank_score": rerank_score,
            "text_preview": text[:200] + ("…" if len(text) > 200 else ""),
        })

    context_block = "\n".join(parts).strip()
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info("ASSEMBLE  chunks=%d  context_chars=%d  (%.0f ms)", n, len(context_block), elapsed)
    return AssembleResult(
        context_block=context_block,
        sources=sources,
        chunk_count=n,
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Stage 5 — GENERATE
# ---------------------------------------------------------------------------


def generate(
    query: str,
    context_block: str,
    *,
    system_prompt: str = _RAG_SYSTEM_PROMPT,
    model: str | None = None,
) -> GenerateResult:
    """Generate a grounded answer from the LLM using the assembled context.

    Sends:
      - system_prompt: instructs the model to answer only from context
      - user message: the context block + the original query

    The user message is structured so the model sees the context before the
    question, reducing hallucination risk.
    """
    t0 = time.perf_counter()

    from src.assistant.api_client import get_chat_client, get_chat_model  # local import

    client = get_chat_client()
    effective_model = model or get_chat_model()

    user_message = (
        f"{context_block}\n\n"
        "---\n"
        f"QUESTION: {query}\n\n"
        "Answer using ONLY the context above. Cite sources as [Volume X, PDF p. Y]."
    )

    response = client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content or ""
    raw = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.info("GENERATE  model=%s  answer_chars=%d  (%.0f ms)", effective_model, len(answer), elapsed)
    return GenerateResult(answer=answer, model=effective_model, elapsed_ms=round(elapsed, 1), raw_response=raw)


# ---------------------------------------------------------------------------
# Dry-run GENERATE (no API key required)
# ---------------------------------------------------------------------------


def _generate_dry_run(query: str, context_block: str, sources: list[dict[str, Any]]) -> GenerateResult:
    """Produce a deterministic synthetic answer for testing without an LLM."""
    pages = [str(s.get("page", "?")) for s in sources[:3]]
    page_refs = ", ".join(f"PDF p. {p}" for p in pages) if pages else "PDF p. ?"
    answer = (
        f"[DRY-RUN] Based on the retrieved NBC chunks, the answer relates to: "
        f"{query[:100]}. "
        f"Relevant provisions are found at {page_refs}. "
        f"[This is a synthetic response — no LLM was called.]"
    )
    return GenerateResult(answer=answer, model="dry-run", elapsed_ms=0.0)


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    query: str,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    metadata_filter: dict[str, Any] | None = None,
    hybrid: bool = True,
    keywords: list[str] | None = None,
    score_threshold: float | None = None,
    rerank_method: str = DEFAULT_RERANK_METHOD,
    model_name: str = DEFAULT_MODEL,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    device: str | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """Run the complete query-to-answer RAG pipeline.

    Stages
    ------
    1. embed   — dense vector for the query
    2. retrieve — hybrid ANN search (vector + optional keyword)
    3. rerank  — CrossEncoder precision re-ranking
    4. assemble — grounded context block construction
    5. generate — LLM answer grounded in retrieved context

    Parameters
    ----------
    query         : Natural-language question.
    candidate_k   : Candidate pool size for retrieval (before re-ranking).
    final_k       : Chunks kept after re-ranking for context assembly.
    metadata_filter : Optional ChromaDB ``where`` filter dict.
    hybrid        : Enable lexical keyword boosting alongside vector search.
    keywords      : Explicit keyword list for hybrid scoring (inferred if None).
    score_threshold : Minimum cosine similarity to include a chunk.
    rerank_method : 'cross_encoder' or 'custom_precision'.
    model_name    : SentenceTransformer embedding model identifier.
    persist_dir   : ChromaDB persistence directory.
    collection_name : ChromaDB collection name prefix.
    device        : Torch device override (e.g. 'cpu', 'cuda').
    dry_run       : Skip LLM call; use synthetic answer (for CI / tests).

    Returns
    -------
    PipelineResult with answer, sources, context_block, stage timings.
    """
    pipeline_start = time.perf_counter()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("=== RAG PIPELINE START ===  query=%r", query[:80])

    # Stage 1 — EMBED
    embed_result = embed(query, model_name=model_name, device=device)

    # Stage 2 — RETRIEVE
    retrieve_result = retrieve(
        query,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        hybrid=hybrid,
        keywords=keywords,
        score_threshold=score_threshold,
        model_name=model_name,
        persist_dir=persist_dir,
        collection_name=collection_name,
        device=device,
    )

    # Stage 3 — RERANK
    rerank_result = rerank(
        query,
        retrieve_result.candidates,
        final_k=final_k,
        method=rerank_method,
        device=device,
    )

    # Stage 4 — ASSEMBLE
    assemble_result = assemble(rerank_result.chunks)

    # Stage 5 — GENERATE
    if dry_run:
        generate_result = _generate_dry_run(query, assemble_result.context_block, assemble_result.sources)
    else:
        generate_result = generate(query, assemble_result.context_block)

    pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
    logger.info("=== RAG PIPELINE END ===  total=%.0f ms", pipeline_elapsed)

    return PipelineResult(
        query=query,
        answer=generate_result.answer,
        sources=assemble_result.sources,
        context_block=assemble_result.context_block,
        generated_at=generated_at,
        pipeline_elapsed_ms=round(pipeline_elapsed, 1),
        stages={
            "embed": {
                "model": embed_result.model_name,
                "vector_dim": len(embed_result.vector),
                "elapsed_ms": embed_result.elapsed_ms,
            },
            "retrieve": {
                "candidate_k": retrieve_result.candidate_k,
                "retrieved_count": len(retrieve_result.candidates),
                "metadata_filter": retrieve_result.metadata_filter,
                "hybrid": retrieve_result.hybrid,
                "elapsed_ms": retrieve_result.elapsed_ms,
            },
            "rerank": {
                "method": rerank_result.method,
                "final_k": rerank_result.final_k,
                "returned_count": len(rerank_result.chunks),
                "elapsed_ms": rerank_result.elapsed_ms,
            },
            "assemble": {
                "chunk_count": assemble_result.chunk_count,
                "context_chars": len(assemble_result.context_block),
                "elapsed_ms": assemble_result.elapsed_ms,
            },
            "generate": {
                "model": generate_result.model,
                "answer_chars": len(generate_result.answer),
                "elapsed_ms": generate_result.elapsed_ms,
            },
        },
    )


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------


def save_pipeline_output(result: PipelineResult, *, label: str = "sample") -> None:
    """Persist the pipeline result as a JSON file in outputs/rag_pipeline/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pipeline_output_{label}.json"

    payload = {
        "query": result.query,
        "answer": result.answer,
        "sources": result.sources,
        "context_block": result.context_block,
        "stages": result.stages,
        "generated_at": result.generated_at,
        "pipeline_elapsed_ms": result.pipeline_elapsed_ms,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Pipeline output saved → %s", out_path)
    print(f"\nOutput saved → {out_path}")


def print_result(result: PipelineResult) -> None:
    """Pretty-print the pipeline result to stdout."""
    bar = "=" * 72
    print(f"\n{bar}")
    print("RAG PIPELINE — END-TO-END RESULT")
    print(bar)
    print(f"\nQUERY:\n  {result.query}\n")
    print(f"ANSWER:\n  {result.answer}\n")
    print("SOURCES:")
    for s in result.sources:
        page = s.get("page", "?")
        heading = s.get("heading", "")[:70]
        sim = s.get("cosine_similarity", 0.0) or 0.0
        rr = s.get("rerank_score")
        rr_str = f"  re-rank={rr:.4f}" if rr is not None else ""
        print(f"  [{s['rank']}] {s['id']}  Page={page}  sim={sim:.4f}{rr_str}")
        if heading:
            print(f"       Section: {heading}")
    print("\nSTAGE TIMINGS:")
    for stage, info in result.stages.items():
        ms = info.get("elapsed_ms", 0)
        print(f"  {stage:<10} {ms:>7.1f} ms")
    print(f"  {'TOTAL':<10} {result.pipeline_elapsed_ms:>7.1f} ms")
    print(bar)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BuildGuard RAG Pipeline — query-to-answer in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--query", "-q",
        default=DEFAULT_SAMPLE_QUERY,
        help="Natural-language question to answer.",
    )
    p.add_argument(
        "--filter-heading",
        default=None,
        help="Restrict retrieval to a specific NBC section heading.",
    )
    p.add_argument(
        "--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
        help=f"Candidate pool size before re-ranking (default: {DEFAULT_CANDIDATE_K}).",
    )
    p.add_argument(
        "--final-k", type=int, default=DEFAULT_FINAL_K,
        help=f"Final chunks kept after re-ranking (default: {DEFAULT_FINAL_K}).",
    )
    p.add_argument(
        "--no-hybrid", action="store_true",
        help="Disable lexical keyword boosting (pure vector search).",
    )
    p.add_argument(
        "--rerank-method", default=DEFAULT_RERANK_METHOD,
        choices=["cross_encoder", "custom_precision"],
        help="Re-ranking method (default: cross_encoder).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM call; use synthetic answer (no API key required).",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save output to disk.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    metadata_filter: dict[str, Any] | None = None
    if args.filter_heading:
        from src.embeddings.filtered_retrieval import build_metadata_filter
        metadata_filter = build_metadata_filter(heading=args.filter_heading)

    result = run_pipeline(
        args.query,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        metadata_filter=metadata_filter,
        hybrid=not args.no_hybrid,
        rerank_method=args.rerank_method,
        dry_run=args.dry_run,
    )

    print_result(result)

    if not args.no_save:
        label = "dry_run" if args.dry_run else "sample"
        save_pipeline_output(result, label=label)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
