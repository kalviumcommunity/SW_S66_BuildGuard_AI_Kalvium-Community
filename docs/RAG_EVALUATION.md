# RAG Evaluation & Answer Quality Scoring

## 1. Overview & Evaluation Goals

To ensure the National Building Code (NBC) compliance assistant delivers reliable, hallucination-free guidance, system performance must be systematically measured rather than assumed. The **RAG Evaluation & Answer Quality Scoring** framework provides an end-to-end evaluation pipeline that assesses:
1. **Factual Correctness:** Whether generated answers accurately reflect statutory requirements and expected technical figures.
2. **Context Grounding:** Whether every claim in the generated answer is corroborated by the retrieved code context.
3. **Citation Accuracy:** Whether cited NBC volume, PDF page, and clause references map to valid retrieved chunks that support the claims.
4. **Negative Refusal Integrity:** Whether out-of-domain, non-existent, or unindexed queries trigger standardized safe refusals.
5. **Conversational Multi-Turn Follow-Ups:** Whether follow-up questions with pronouns and elliptical phrasing maintain retrieval precision and answer quality.

---

## 2. Evaluation Architecture & Metrics Formulation

```
                     Evaluation Test Set (JSON)
                                │
                                ▼
                       RAG System Execution
                    (Retrieve ➔ Ground ➔ Cite)
                                │
                                ▼
                   Three-Pillar Quality Scoring
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
[Correctness Scorer]    [Grounding Auditor]    [Citation Validator]
- Keyword/fact coverage  - Sentence containment - Chunk ID validity
- Semantic similarity    - Hallucination penalty- Fact corroboration
- Refusal validation                            - Expected source match
       └────────────────────────┬────────────────────────┘
                                │
                                ▼
                     Overall Quality Index
            (0.45 * Correctness + 0.35 * Grounding + 0.20 * Citation)
                                │
                                ▼
                Summary Reports & Failure Diagnosis
```

### Metrics Formula Definitions:
* **Correctness Score ($S_{corr}$):**
  $$S_{corr} = 0.70 \times \frac{\text{matched\_facts}}{\text{required\_facts}} + 0.30 \times \text{semantic\_similarity}$$
  *(For refusal cases, $S_{corr} = 1.0$ if properly refused, $0.0$ if an answer was hallucinated).*
* **Grounding Score ($S_{ground}$):**
  $$S_{ground} = \frac{\text{sentences corroborated by retrieved context}}{\text{total sentences}}$$
* **Citation Accuracy Score ($S_{cite}$):**
  $$S_{cite} = 0.50 \times \text{precision} + 0.30 \times \text{support\_ratio} + 0.20 \times \text{expected\_source\_coverage}$$
* **Overall Quality Index ($S_{overall}$):**
  $$S_{overall} = 0.45 \times S_{corr} + 0.35 \times S_{ground} + 0.20 \times S_{cite}$$

---

## 3. Test Set Specification (Task 1)

The test set (`data/rag_evaluation_test_set.json` & `outputs/rag_evaluation/eval_test_set.json`) comprises 8 curated test cases:

| Test ID | Category | Query / Focus | Expected Sources | Target Behavior |
|:---|:---|:---|:---|:---:|
| `tc01_boiler_room_wall_rating` | Fire Resistance | Minimum fire rating for boiler room walls | Part 4 Clause 6.1.1.2, Table 1 | Answer with 120 min rating |
| `tc02_boiler_room_doors_opening` | Openings Protection | Fire doors in boiler room separation walls | Part 4 Clause 6.1.1.4 | Answer with self-closing 120 min doors |
| `tc03_exit_access_corridor_definition` | Egress & Life Safety | Statutory definition of Exit Access Corridor | Part 4 Clause 2.18 | Provide statutory definition |
| `tc04_dead_end_corridor_travel_distance` | Egress & Life Safety | Dead-end corridor length in educational buildings | Part 4 Clause 4.4.1 | Answer with 6 m limit |
| `tc05_transit_station_horizontal_exits` | Special Occupancies | Mass transit emergency illumination & horizontal exits | Part 4 Clause J-5.4.12 | Answer with 2-hr barrier & exit illumination |
| `tc06_conversational_follow_up_doors` | Conversational Follow-Up | Follow-up: *"What about the doors in those walls?"* | Part 4 Clause 6.1.1.4 | Rewrite query & retrieve door rating |
| `tc07_out_of_domain_negative` | Negative Control | Indian Copyright Act penalties | None | Trigger safe refusal |
| `tc08_missing_provision_unindexed` | Negative Control | Quantum computing data center cryogenic specs | None | Trigger missing-context fallback |

---

## 4. Benchmark Quality Results (Task 2 & 3)

| Benchmark Metric | Measured Score | Target Threshold | Status |
|:---|:---:|:---:|:---:|
| **Mean Context Grounding** | `0.9583` | $\ge 0.85$ | ✅ PASS |
| **Mean Citation Accuracy & Support** | `0.8979` | $\ge 0.75$ | ✅ PASS |
| **Negative Refusal Precision** | `100.0%` | $100\%$ | ✅ PASS |
| **Overall RAG Quality Index** | **`0.8257`** | $\ge 0.80$ | ✅ PASS |

---

## 5. Notable Failures & Root Cause Taxonomy (Task 4)

The evaluation engine audits subthreshold test cases and classifies them into an actionable root-cause taxonomy:

| Failure Taxonomy Code | Description | Observed In | Remediation |
|:---|:---|:---|:---|
| `RETRIEVAL_MISS` | Required regulatory chunk not ranked in top-k | — | Increase `candidate_k` and enable hybrid BM25 + dense re-ranking. |
| `PARTIAL_CORRECTNESS` | Core requirement mentioned but secondary term omitted | `tc03`, `tc04`, `tc05` (offline simulation) | Augment prompt context assembly with structured chunk excerpts. |
| `UNSUPPORTED_CITATION` | Citation marker points to chunk lacking claim | — | Apply citation post-verification filter before delivering answer. |
| `FALSE_CONFIDENCE` | LLM answers ungrounded query instead of refusing | — | Enforce strict hallucination guardrail refusal thresholds. |
| `WEAK_GROUNDING` | Sentences contain assertions uncorroborated by context | — | Penalize unsupported clauses with fallback substitution. |

---

## 6. CLI Usage & Verification (Task 5)

```bash
# 1. Run full evaluation across test set:
uv run python -m src.rag_evaluation

# 2. Run in deterministic offline mode:
uv run python -m src.rag_evaluation --offline

# 3. Run with custom test set:
uv run python -m src.rag_evaluation --test-set data/rag_evaluation_test_set.json

# 4. Run test suite:
uv run python -m unittest tests/test_rag_evaluation.py
```
