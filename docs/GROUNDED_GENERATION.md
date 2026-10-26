# Grounded Answer Generation & Retrieval Comparison

## 1. Overview & Objectives

In production RAG systems for regulatory compliance like the National Building Code of India (NBC), generating answers based on parametric memory alone leads to dangerous hallucinations, incorrect numerical values, or citations to irrelevant foreign standards (e.g. IBC or NFPA).

The **Grounded Answer Generation** module guarantees:
1. **Context Grounding:** The LLM's responses are strictly confined to the factual evidence provided in the retrieved chunks.
2. **Source Accuracy Auditing:** Automated verification parses generated answers, extracts citation markers and technical metrics, cross-references them against source chunks, and flags unverified assertions.
3. **Missing-Context Fallback:** When a question lacks supporting context (out-of-domain questions or retrieval returns zero/low-similarity chunks), the system safely returns an explicit fallback message instead of guessing.
4. **Retrieval Impact Measurement:** Evaluates identical compliance queries with and without retrieval, proving the necessity of grounded RAG for statutory safety standards.

---

## 2. End-to-End Flow & Architecture

```
User Compliance Query
        │
        ▼
Retrieval & Context Assembly (ChromaDB Vector Store)
        │
        ▼
   [Are chunks present & similarity >= 0.55?]
        │
       ├─── NO ───► Return Missing-Context Fallback:
       │            "I don't have enough information to answer that
       │             reliably based on the provided National Building Code context."
       │
      YES
       │
       ▼
Grounded Prompt Injection (System Rules + Formatted Chunks [1], [2] + Question)
       │
       ▼
Model Generation (Temperature 0.0)
       │
       ▼
Source Accuracy Verification Engine
  ├─ Citation Syntax Validation (Verify [1], [2], [Volume X, PDF p. Y] map to context)
  ├─ Technical Metric Verification (Confirm ratings like '120 min', clauses in text)
  └─ Grounding Score Calculation (0.0 to 1.0)
       │
       ▼
Verified Grounded Answer Output + Audit Telemetry
```

---

## 3. Source Accuracy Verification Algorithm

The verification engine (`verify_source_accuracy`) performs multi-factor claim checking:

1. **Citation Extraction & Boundary Check:**
   - Detects all numeric citation markers (`[1]`, `[2]`) and volume/page references (`[Volume 1, PDF p. 314-315]`).
   - Ensures marker indices fall strictly within the valid range $[1, N]$ of retrieved chunks. Out-of-bounds citations are flagged as hallucinations.
2. **Technical Metric Cross-Referencing:**
   - Identifies numerical constraints and units (e.g. `120 min`, `2 hours`, `180 min`, `3 m`, `50`).
   - Normalizes units (`minutes` $\rightarrow$ `min`, `hours` $\rightarrow$ `h`) and verifies that each metric occurs in the source chunks.
3. **Clause Reference Verification:**
   - Detects clause notations (`Clause 6.1.1.2`, `3.4.8.1`) and ensures they match metadata and body text in the retrieved chunks.
4. **Scoring & Verdict:**
   - Grounding score $\ge 0.70$ with zero invalid citations $\rightarrow$ `VERIFIED_GROUNDED`.
   - Score between $0.40$ and $0.69$ $\rightarrow$ `PARTIALLY_GROUNDED`.
   - Score $< 0.40$ or unanchored $\rightarrow$ `UNGROUNDED`.
   - Fallback text detected $\rightarrow$ `FALLBACK_TRIGGERED`.

---

## 4. Missing-Context Fallback Policy

To maintain compliance integrity, the model is never permitted to guess. The fallback phrase:

> *"I don't have enough information to answer that reliably based on the provided National Building Code context."*

is triggered automatically under any of the following conditions:
- **Zero Retrieval:** Vector search returns zero candidates for the query.
- **Relevance Threshold Filtering:** The highest cosine similarity among retrieved chunks falls below the relevance threshold ($\tau < 0.55$).
- **Model Fallback:** The LLM's own internal compliance evaluation concludes that the provided chunks do not address the prompt.

---

## 5. Comparative Evaluation: With vs. Without Retrieval

| Evaluation Aspect | With Retrieval (Grounded RAG) | Without Retrieval (Unanchored Baseline) |
|:---|:---|:---|
| **Query** | *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?* | *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?* |
| **Generated Output** | *"Under Clause 6.1.1.2 of the National Building Code of India [1], walls separating a boiler room or central heating plant from other parts of a building must provide a minimum fire resistance rating of 120 minutes (2 hours). Furthermore, all connecting openings must be protected by self-closing fire doors with an equivalent 120-minute fire rating [Volume 1, PDF p. 314-315]."* | *"Generally, in commercial building standards such as the International Building Code (IBC) or NFPA standards, boiler room separating walls typically require a 1-hour or 2-hour fire resistance rating depending on whether an automatic fire sprinkler system is installed. Local municipal bylaws may specify different requirements."* |
| **Citations** | Specific (`[1]`, `[Volume 1, PDF p. 314-315]`) | None |
| **Factual Precision** | Exact 120-minute fire resistance rating for walls and self-closing doors | Vague range (1 to 2 hours) with no binding statutory weight |
| **Jurisdiction** | Strictly National Building Code of India 2016 | Cites foreign codes (IBC, NFPA) that do not apply in India |
| **Accuracy Audit** | `VERIFIED_GROUNDED` (Grounding Score: 100%) | `UNGROUNDED` (Grounding Score: 0%) |

---

## 6. Generated Artifacts

- **JSON Data:** [outputs/grounded_generation/grounded_answers_sample.json](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/grounded_generation/grounded_answers_sample.json) contains the complete machine-readable test queries, answers, supporting chunk metadata, and source accuracy audits.
- **Markdown Report:** [outputs/grounded_generation/grounded_answers_report.md](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/grounded_generation/grounded_answers_report.md) provides the human-readable comparative breakdown.

---

## 7. Verification & CLI Usage

Run unit tests:
```bash
uv run python -m unittest tests.test_grounded_generation -v
```

Run CLI against live ChromaDB:
```bash
uv run python -m src.grounded_generation
```

Run offline simulation:
```bash
uv run python -m src.grounded_generation --offline
```
