# Hallucination Guardrails & Refusal Handling Report

**Generated:** `2026-10-26T14:27:25Z`  
**System:** BuildGuard AI — National Building Code of India (NBC) Assistant  

## 1. Executive Summary

To eliminate hallucination risks in statutory building compliance:
1. **Pre-Generation Audit:** The system inspects retrieval quality *before* passing context to the LLM.
2. **Strict Refusal Trigger:** When retrieval is weak, empty, or mismatched, the system safely refuses instead of fabricating requirements.
3. **Preserved High-Confidence Answers:** When strong, authoritative context is retrieved, confident grounded answers with full citations are preserved.

## 2. Guardrail Configuration & Thresholds

| Guardrail Parameter | Configured Value | Operational Purpose |
|:---|---:|:---|
| `min_similarity_threshold` | **0.65** | Minimum cosine similarity of top candidate chunk |
| `min_chunks_above_threshold` | **1** | Minimum number of corroborated chunks required |
| `min_avg_similarity` | **0.50** | Minimum aggregate similarity of retrieved candidate set |
| `min_rerank_score` | **0.0** | Minimum cross-encoder logit score required |
| `enable_topic_overlap_check` | **True** | Enforces that query subject keywords appear in context |

## 3. Confident Answer Case (PROCEED)

**Query:** *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?*

- **Guardrail Decision:** `PROCEED` (`OK`)
- **Max Similarity:** `0.8264` (Threshold: `0.65`)
- **Chunks Above Threshold:** `3`
- **Status:** `ANSWERED`

### Generated Answer (Grounded with Citations)
> Under Clause 6.1.1.2 of the National Building Code of India [1], walls separating a boiler room or central heating equipment from other parts of a building must provide a minimum fire resistance rating of 120 minutes (2 hours). All connecting door openings must be protected by self-closing fire doors having an equivalent 120-minute rating [Volume 1, PDF p. 314-315].

**Citations Included:** `[1]`, `[Volume 1, PDF p. 314-315]`

## 4. Refusal Cases (REFUSE)

The system correctly intercepted weak retrieval conditions and issued safe refusals:

| Case | Query | Reason Code | Detection Signal | System Refusal Output |
|:---:|:---|:---:|:---|:---|
| **1** | *What is the penalty for copyright infringement und...* | `EMPTY_RETRIEVAL` | Vector search returned 0 candidates matching query.... | I cannot answer this question because no relevant provisions were foun... |
| **2** | *What are the offshore structural drilling platform...* | `LOW_MAX_SIMILARITY` | Top similarity (0.3812) is below minimum threshold (0.6500).... | I cannot answer this question because the highest-scoring building cod... |
| **3** | *What is the legal capital gains tax rate under Sec...* | `LOW_RERANK_SCORE` | Top re-rank score (-0.8000) is below threshold (0.0000).... | I cannot answer this question because the cross-encoder re-ranker dete... |

### Detailed Refusal Case Inspections

#### Case 1: `EMPTY_RETRIEVAL`
**Query:** *What is the penalty for copyright infringement under the Indian Copyright Act?*  
**Decision:** `REFUSE` — Vector search returned 0 candidates matching query.  
**Refusal Text Returned:**
> *"I cannot answer this question because no relevant provisions were found in the National Building Code index for this topic."*

#### Case 2: `LOW_MAX_SIMILARITY`
**Query:** *What are the offshore structural drilling platform safety protocols in deep sea?*  
**Decision:** `REFUSE` — Top similarity (0.3812) is below minimum threshold (0.6500). Context is too weak.  
**Refusal Text Returned:**
> *"I cannot answer this question because the highest-scoring building code chunk falls below the required confidence threshold, indicating weak or ambiguous evidence."*

#### Case 3: `LOW_RERANK_SCORE`
**Query:** *What is the legal capital gains tax rate under Section 54 of Income Tax?*  
**Decision:** `REFUSE` — Top re-rank score (-0.8000) is below threshold (0.0000).  
**Refusal Text Returned:**
> *"I cannot answer this question because the cross-encoder re-ranker determined that candidate chunks do not directly address the query."*