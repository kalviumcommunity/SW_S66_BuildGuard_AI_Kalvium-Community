# Hallucination Guardrails & Refusal Handling

## 1. Overview & Problem Statement

In safety-critical domain applications such as the National Building Code of India (NBC), generating answers when retrieval evidence is weak or absent leads to catastrophic errors—such as fabricating nonexistent fire ratings, inventing clause references, or applying irrelevant foreign standards (e.g. IBC or NFPA).

The **Hallucination Guardrails & Refusal Handling** system establishes a rigorous pre-generation firewall that:
1. **Audits Retrieval Signals:** Detects when retrieval returns zero results, weak cosine similarity, or low cross-encoder re-rank scores.
2. **Issues Safe Refusals:** Refuses to generate an answer when context is insufficient, returning domain-aware "I don't know" refusal messages rather than hallucinating.
3. **Enforces Quality Thresholds:** Uses multi-signal heuristics (similarity threshold, minimum corroborated chunks, topic overlap).
4. **Preserves Confident Answers:** Ensures that strong, relevant context seamlessly passes through to generate fully cited, verified answers.

---

## 2. Guardrail Architecture & Flow

```
                         User Query
                             │
                             ▼
                 Retrieved Candidate Chunks
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Retrieval Quality Auditor   │
             │                               │
             │ 1. Empty Candidate Check?     │
             │ 2. Max Similarity >= 0.65?   │
             │ 3. Chunks Above Threshold?    │
             │ 4. Mean Similarity >= 0.50?   │
             │ 5. Re-rank Score >= 0.0?      │
             │ 6. Subject Keyword Overlap?   │
             └───────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
       [Refusal Condition Met]        [All Thresholds Passed]
             │                               │
             ▼                               ▼
     Safe Refusal Message            Grounded Answer Generation
  "I cannot answer because..."       - Injected context block
  - Zero LLM Hallucination           - Full citations [1], [Vol 1, p. X]
  - Explicit reason telemetry        - High factual fidelity
```

---

## 3. Detection Signals & Threshold Heuristics

| Signal | Metric / Condition | Threshold | Failure Reason Code | Operational Action |
|:---|:---|:---:|:---:|:---|
| **Zero Retrieval** | `len(candidates) == 0` | $> 0$ | `EMPTY_RETRIEVAL` | Refuse: Topic unindexed in knowledge base |
| **Weak Vector Match** | `max(cosine_similarity)` | $\ge 0.65$ | `LOW_MAX_SIMILARITY` | Refuse: Top chunk too distant in embedding space |
| **Insufficient Evidence** | `count(sim >= threshold)` | $\ge 1$ (or 2) | `INSUFFICIENT_SUPPORTING_CHUNKS` | Refuse: Lack of corroborating sections |
| **Low Average Relevance** | `mean(candidate_similarities)` | $\ge 0.50$ | `LOW_AVERAGE_SIMILARITY` | Refuse: Overall candidate pool is noisy |
| **Cross-Encoder Rejection** | `max(rerank_logits)` | $\ge 0.0$ | `LOW_RERANK_SCORE` | Refuse: Cross-attention determined irrelevance |
| **Topic Keyword Mismatch** | Subject term overlap ratio | $> 0.0$ | `TOPIC_MISMATCH` | Refuse: Query terms absent from retrieved text |

---

## 4. Refusal Taxonomy

The guardrail returns targeted, professional refusal messages specific to the failure mode:

- **Empty Retrieval:**
  > *"I cannot answer this question because no relevant provisions were found in the National Building Code index for this topic."*
- **Low Similarity / Ambiguous Context:**
  > *"I cannot answer this question because the highest-scoring building code chunk falls below the required confidence threshold, indicating weak or ambiguous evidence."*
- **Cross-Encoder Rejection:**
  > *"I cannot answer this question because the cross-encoder re-ranker determined that candidate chunks do not directly address the query."*
- **Topic Mismatch / Out of Domain:**
  > *"I cannot answer this question because the query subject appears to fall outside the scope of the indexed National Building Code provisions."*

---

## 5. Confident Answer vs. Refusal Cases

| Test Case | Query | Guardrail Decision | Status | Output / Answer |
|:---:|:---|:---:|:---:|:---|
| **Confident Case** | *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?* | `PROCEED (OK)` (Max Sim: 0.8264) | `ANSWERED` | **"Under Clause 6.1.1.2 of the National Building Code of India [1], walls separating a boiler room... must provide a minimum fire resistance rating of 120 minutes (2 hours) [Volume 1, PDF p. 314-315]."** |
| **Refusal Case 1** | *What is the penalty for copyright infringement under the Indian Copyright Act?* | `REFUSE (EMPTY_RETRIEVAL)` | `REFUSED` | *"I cannot answer this question because no relevant provisions were found in the National Building Code index for this topic."* |
| **Refusal Case 2** | *What are the offshore structural drilling platform safety protocols in deep sea?* | `REFUSE (LOW_MAX_SIMILARITY)` (Sim: 0.3812 vs 0.65) | `REFUSED` | *"I cannot answer this question because the highest-scoring building code chunk falls below the required confidence threshold..."* |
| **Refusal Case 3** | *What is the legal capital gains tax rate under Section 54 of Income Tax?* | `REFUSE (LOW_RERANK_SCORE)` (Score: -0.80) | `REFUSED` | *"I cannot answer this question because the cross-encoder re-ranker determined that candidate chunks do not directly address the query."* |

---

## 6. Generated Artifacts

- **JSON Output:** [outputs/guardrails/guardrail_evaluations_sample.json](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/guardrails/guardrail_evaluations_sample.json) contains the machine-readable test cases, telemetry, and decisions.
- **Markdown Report:** [outputs/guardrails/guardrail_evaluations_report.md](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/guardrails/guardrail_evaluations_report.md) provides the human-readable audit tables and case analyses.

---

## 7. Verification

Run the test suite:
```bash
uv run python -m unittest tests.test_hallucination_guardrails -v
```

Execute the CLI against live ChromaDB:
```bash
uv run python -m src.hallucination_guardrails
```
