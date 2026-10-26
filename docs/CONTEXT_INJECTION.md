# Context Injection & Prompt Augmentation

## 1. Overview & Problem Formulation

Standard vector retrieval returns relevant chunks from an embedding store, but raw chunks cannot simply be dumped into an LLM call. Real-world RAG systems require a structured **context injection & prompt augmentation** stage that:
1. **Controls Context Token Budgets:** LLM context windows are finite and costly. Uncontrolled context injection risks prompt truncation, dropped user questions, or model generation failures.
2. **Injects Unambiguous Source Markers:** In-context evidence must have distinct, machine-parseable, and human-readable identifiers (e.g. `[1]`, `[2]`, `[Volume 1, PDF p. 314-315]`) so the model can cite its sources precisely.
3. **Enforces Grounding Constraints:** Models tend to extrapolate or hallucinate when context is ambiguous. Explicit system prompt rules must direct the model to answer *only* from the supplied text and specify a standardized fallback phrase when context is insufficient.

---

## 2. Architecture & Prompt Flow

```
Retrieved Candidate Chunks (ranked by relevance)
                     │
                     ▼
       ┌───────────────────────────────┐
       │   Token Budget Calculator     │
       │                               │
       │  max_total_tokens (e.g. 4096) │
       │  - reserved_answer   (512)    │
       │  - system_prompt     (189)    │
       │  - query_overhead    (120)    │
       │  ──────────────────────────── │
       │  = available_context (3275)   │
       └───────────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────────┐
       │     Greedy Context Packer     │
       │  - Add chunk [1], [2]...      │
       │  - Truncate if partial fits   │
       │  - Skip remaining if full     │
       └───────────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────────┐
       │    Source Marker Formatter    │
       │  - [Index]                    │
       │  - Doc, Vol, Page, Clause     │
       │  - Section Heading            │
       │  - Similarity & Re-rank Score │
       └───────────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────────┐
       │     Augmented Prompt          │
       │                               │
       │  System: Grounding Rules      │
       │  User: Context + Query        │
       └───────────────────────────────┘
```

---

## 3. Token Budget Accounting

The token budget is strictly partitioned to guarantee headroom for model generation:

$$\text{Available Context Tokens} = \text{Max Total Window} - \text{Reserved Answer} - \text{System Instructions} - \text{Query Template Overhead}$$

| Metric | Typical Value | Purpose |
|:---|---:|:---|
| `max_total_tokens` | `4,096` | Upper limit for the full model context window |
| `reserved_answer_tokens` | `512` | Guaranteed output capacity for the model |
| `system_tokens` | `~189` | Grounding rules, persona, and fallback instructions |
| `query_template_overhead_tokens` | `~120` | User question, section framing, and prompt headers |
| `available_context_tokens` | `~3,275` | Maximum allowable tokens for injected NBC chunks |

### Packing & Truncation Strategy
1. **Ordered Insertion:** Chunks are considered in strict relevance order (top-ranked first).
2. **Exact Fit:** If the candidate chunk and its source marker fit within remaining context tokens, it is injected fully.
3. **Partial Truncation:** If a chunk exceeds remaining tokens, but remaining headroom is $\ge \text{min\_chunk\_tokens}$ (default: 25 tokens), the text is cleanly sliced at a token boundary and appended with `[... truncated to fit token budget ...]`.
4. **Safety Spillover:** Chunks that cannot fit even partially are skipped, and detailed budget telemetry is recorded.

---

## 4. Source Markers & Citation Syntax

Every injected chunk is annotated with a structured header and visual dividers:

```text
[1] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 314-315 | Clause: 6.1.1.2 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.8264
------------------------------------------------------------
120 min fire resistance rating wall. Such areas shall be provided with fire doors...
------------------------------------------------------------
```

### Supported Styles
- **`numeric` (default):** Short bracketed indices like `[1]`, `[2]`, `[3]`.
- **`verbose`:** Explicit identifiers like `[Source 1]`, `[Source 2]`.

---

## 5. Grounding Instructions & Negative Constraints

The system message provides strict operational constraints to prevent hallucination:

```text
You are an expert staff-support assistant answering technical compliance questions about the National Building Code of India (NBC).
Follow these strict operational rules:
1. Answer the question using ONLY the factual evidence directly stated in the RETRIEVED BUILDING CODE CONTEXT below.
2. Do NOT extrapolate, speculate, assume, or introduce outside knowledge not corroborated by the provided context.
3. Attribute every factual claim, requirement, rating, dimension, or specification using the bracketed source markers (e.g. [1], [2]) or explicit citations such as [Volume 1, PDF p. 314-315].
4. If the provided context does NOT contain sufficient factual evidence to answer the question reliably, reply with this exact fallback message:
   "I don't have enough information to answer that reliably based on the provided National Building Code context."
5. Keep the response professional, precise, and concise (typically 2 to 4 sentences).
```

---

## 6. Artifacts & Outputs

- **JSON Output:** [outputs/prompts/sample_augmented_prompt.json](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/prompts/sample_augmented_prompt.json) contains the complete machine-readable prompt, token budget metrics, and structured injected chunk list.
- **Markdown Report:** [outputs/prompts/sample_augmented_prompt.md](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/prompts/sample_augmented_prompt.md) displays the rendered prompt, budget table, and source metadata.

---

## 7. Verification

Run the test suite:
```bash
uv run python -m unittest tests.test_context_assembly -v
```

Execute the CLI against ChromaDB:
```bash
uv run python -m src.context_assembly
```
