# RAG Evaluation & Answer Quality Scoring Report

**Evaluated At:** `2026-10-26T19:28:35.926091+00:00`  
**Total Test Cases:** `8`  
**Pass Rate:** **`62.5%`** (`5/8` passed)  
**Mean Overall Score:** **`0.8257`**  

---

## 1. Executive Quality Benchmarks

| Evaluation Metric | Aggregate Score | Benchmark Target | Verdict |
|:---|:---:|:---:|:---:|
| **Factual Correctness** | `0.6904` | $\ge 0.80$ | ⚠️ SUBPAR |
| **Context Grounding** | `0.9583` | $\ge 0.85$ | ✅ PASS |
| **Citation Accuracy & Support** | `0.8979` | $\ge 0.75$ | ✅ PASS |
| **Negative Refusal Accuracy** | `100.0%` | $100\%$ | ✅ PASS |
| **Overall RAG Quality Index** | **`0.8257`** | $\ge 0.80$ | ✅ CERTIFIED |

---

## 2. Test Set Itemized Results

| Test ID | Category | Question | Correctness | Grounding | Citations | Overall | Verdict |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `tc01_boiler_room_wall_rating` | Fire Resistance & Separation | "What is the minimum fire resistance rati..." | `0.91` | `1.00` | `0.85` | **`0.93`** | ✅ PASS |
| `tc02_boiler_room_doors_opening` | Fire Doors & Openings Protection | "What are the fire resistance and closing..." | `0.89` | `1.00` | `0.80` | **`0.91`** | ✅ PASS |
| `tc03_exit_access_corridor_definition` | Egress & Life Safety | "What is the definition and regulatory re..." | `0.20` | `1.00` | `0.87` | **`0.62`** | ❌ FAIL |
| `tc04_dead_end_corridor_travel_distance` | Egress & Life Safety | "What is the maximum permissible length f..." | `0.25` | `1.00` | `0.87` | **`0.63`** | ❌ FAIL |
| `tc05_transit_station_horizontal_exits` | Special Occupancies | "What are the criteria for horizontal exi..." | `0.32` | `1.00` | `0.87` | **`0.67`** | ❌ FAIL |
| `tc06_conversational_follow_up_doors` | Conversational Follow-Up | "What about the doors in those walls?" | `0.95` | `0.67` | `0.93` | **`0.85`** | ✅ PASS |
| `tc07_out_of_domain_negative` | Negative / Out-of-Domain | "What is the maximum criminal penalty for..." | `1.00` | `1.00` | `1.00` | **`1.00`** | ✅ PASS |
| `tc08_missing_provision_unindexed` | Negative / Missing Provision | "What are the quantum computing data cent..." | `1.00` | `1.00` | `1.00` | **`1.00`** | ✅ PASS |

---

## 3. Category-Wise Performance Breakdown

| Category | Test Count | Mean Quality Score | Assessment |
|:---|:---:|:---:|:---|
| **Fire Resistance & Separation** | 1 | `0.9285` | Excellent |
| **Fire Doors & Openings Protection** | 1 | `0.9105` | Excellent |
| **Egress & Life Safety** | 2 | `0.6248` | Needs Improvement |
| **Special Occupancies** | 1 | `0.6694` | Needs Improvement |
| **Conversational Follow-Up** | 1 | `0.8475` | Adequate |
| **Negative / Out-of-Domain** | 1 | `1.0000` | Excellent |
| **Negative / Missing Provision** | 1 | `1.0000` | Excellent |

---

## 4. Notable Failures & Root Cause Diagnosis (Task 4)

### Failure Case: `tc03_exit_access_corridor_definition` (Egress & Life Safety)
- **Question:** "What is the definition and regulatory requirement for an Exit Access Corridor under Clause 2.18?"
- **Overall Score:** `0.6156` (Correctness: `0.20`, Grounding: `1.00`, Citation: `0.87`)
- **Diagnosed Root Cause:** `PARTIAL_CORRECTNESS: Missing required facts: Exit Access Corridor, Clause 2.18, passageway`
- **Generated Response:** > Based on the retrieved provisions in [1], 1). 2 Fire doors in exits shall have fire rating as required in this oo Part to meet the requirement of integrity and st... This requirement is formally specified in the code [Volume 1, PDF p. 264].

**Remediation Recommendation:**
Enhance candidate retrieval recall (increase candidate_k or tune hybrid keywords) and enforce stricter grounding thresholds.

### Failure Case: `tc04_dead_end_corridor_travel_distance` (Egress & Life Safety)
- **Question:** "What is the maximum permissible length for a dead-end corridor in educational and assembly buildings?"
- **Overall Score:** `0.6339` (Correctness: `0.25`, Grounding: `1.00`, Citation: `0.87`)
- **Diagnosed Root Cause:** `PARTIAL_CORRECTNESS: Missing required facts: 6 m, educational, Clause 4.4.1`
- **Generated Response:** > Based on the retrieved provisions in [1], 4.4.1 Exit Access c) The dead end corridor length in exit access a) A common pathoftravel is desirable in exit shall not... This requirement is formally specified in the code [Volume 1, PDF p. 285-286].

**Remediation Recommendation:**
Enhance candidate retrieval recall (increase candidate_k or tune hybrid keywords) and enforce stricter grounding thresholds.

### Failure Case: `tc05_transit_station_horizontal_exits` (Special Occupancies)
- **Question:** "What are the criteria for horizontal exits and emergency escape illumination in transit stations under Clause J-5.4.12?"
- **Overall Score:** `0.6694` (Correctness: `0.32`, Grounding: `1.00`, Citation: `0.87`)
- **Diagnosed Root Cause:** `PARTIAL_CORRECTNESS: Missing required facts: horizontal exit, transit`
- **Generated Response:** > Based on the retrieved provisions in [1], J-5.4.12 Emergency and escape lighting and c) Gate-type fare barriers in the means of egress illumination ofmeansofall e... This requirement is formally specified in the code [Volume 1, PDF p. 355].

**Remediation Recommendation:**
Enhance candidate retrieval recall (increase candidate_k or tune hybrid keywords) and enforce stricter grounding thresholds.

---

## 5. Verification Checklist

- [x] **Task 1 — Prepare a test set:** Curated 8 test cases spanning fire resistance, egress corridor lengths, mass transit, conversational follow-ups, and negative controls.
- [x] **Task 2 — Score correctness and grounding:** Evaluated keyword coverage, factual accuracy, and sentence-level context grounding.
- [x] **Task 3 — Check citation accuracy:** Audited citation existence, validity of chunk references, and support ratios.
- [x] **Task 4 — Summarize quality and failures:** Published mean metrics, category breakdowns, and diagnosed root cause failure taxonomy.
- [x] **Task 5 — Commit evaluation results:** Saved JSON and Markdown reports to `outputs/rag_evaluation/` and documented in `docs/RAG_EVALUATION.md`.