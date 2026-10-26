# Source Citation & Attribution Documentation

## Overview

In safety-critical regulatory domains like the **National Building Code of India (NBC)**, generated answers must not only be grounded—they must be **verifiable**. Users (architects, fire marshals, structural engineers) cannot accept answers on blind faith; they require direct attribution back to the specific volume, page range, and clause of the governing standard.

The `src/citation_attribution.py` module introduces an end-to-end citation and attribution framework that:
1. Emits numbered in-text citations (`[1]`, `[2]`, `[Volume 1, PDF p. 314-315]`) attached directly to each factual assertion.
2. Maps citation markers to structured metadata records (`chunk_id`, `source_document`, `volume`, `page_start`, `page_end`, `clause`, `section_heading`, similarity/re-rank scores).
3. Performs automated claim-by-claim verification auditing against retrieved chunk texts to certify factual alignment and extract evidence snippets.
4. Enforces strict anti-hallucination guardrails: if retrieval context is empty or falls below the minimum similarity threshold, the system returns a safe refusal with zero citations rather than fabricating false references.

---

## Architecture & Flow

```mermaid
flowchart TD
    UserQuery[User Query] --> Retrieve[Dense Vector Retrieval + Re-ranking]
    Retrieve --> ThresholdCheck{Context Strong enough?<br/>sim >= 0.65}
    
    %% Fallback path
    ThresholdCheck -- No / Empty Context --> SafeRefusal[No-Source Fallback<br/>Refusal Response]
    SafeRefusal --> ZeroCitations[Zero Citations Attached<br/>No Fabricated Markers]
    
    %% Grounded path
    ThresholdCheck -- Yes --> BuildMap[Build In-Memory Citations Map<br/>Markers: [1], [2], ...]
    BuildMap --> InjectPrompt[Inject Markers into Context Prompt]
    InjectPrompt --> LLMGen[Grounded Answer Generation]
    LLMGen --> ParseClaims[Parse Sentences & Citation Markers]
    ParseClaims --> AuditCheck{Automated Claim-Source Audit}
    
    AuditCheck --> VerifyMetrics[1. Metrics & Units Check<br/>120 min <-> 2 hours]
    AuditCheck --> VerifyClauses[2. Statutory Clause Match<br/>e.g. Clause 6.1.1.2]
    AuditCheck --> VerifyKeywords[3. Token Overlap & Snippet Extractor]
    
    VerifyMetrics --> AuditReport[CitationVerificationReport]
    VerifyClauses --> AuditReport
    VerifyKeywords --> AuditReport
    
    AuditReport --> UserDelivery[Deliver Answer + Citation Cards + Audit Report]
```

---

## Metadata Schema

Each retrieved chunk is assigned an immutable citation marker (e.g. `[1]`) and mapped to the following schema:

| Field | Type | Description | Example |
|:---|:---|:---|:---|
| `marker_id` | `int` | Numerical identifier (1-indexed) | `1` |
| `marker_tag` | `str` | In-text reference tag | `"[1]"` |
| `chunk_id` | `str` | Vector store unique chunk key | `"nbc-v1-p314-001012"` |
| `source_document` | `str` | Canonical filename | `"nationalbuilding01.pdf"` |
| `volume` | `int / str` | NBC volume designation | `1` |
| `page` | `str` | Physical PDF page range | `"314-315"` |
| `clause` | `str` | Section / clause number | `"6.1.1.2"` |
| `section_heading` | `str` | Chapter / subject heading | `"Part 4 — Fire and Life Safety"` |
| `similarity_score` | `float` | Cosine similarity from dense retriever | `0.8264` |
| `rerank_score` | `float` | Cross-encoder relevance score (optional) | `4.0295` |
| `snippet` | `str` | Representative excerpt for UI card preview | `"120 min fire resistance rating wall..."` |
| `full_text` | `str` | Raw retrieved text used for claim verification | `Full text content...` |

---

## Claim Verification Algorithm

To ensure a user can verify any cited source against the original retrieved text, `verify_claim_against_source()` applies a multi-signal factual verification audit:

1. **Numerical & Metric Normalization:**
   - Detects duration units (`120 min` $\leftrightarrow$ `2 hours`, `60 min` $\leftrightarrow$ `1 hour`).
   - Detects linear measurements (`1.5 m` $\leftrightarrow$ `1500 mm`).
   - Detects dimensionless building factors (slenderness ratio `50`, `30`).
2. **Statutory Clause Alignment:**
   - Extracts clause numbers (e.g. `6.1.1.2`, `3.4.8.1`) from claims and confirms presence in the cited chunk's metadata or text.
3. **Keyword & Concept Overlap:**
   - Filters out non-content stopwords (`under`, `must`, `with`, `that`, `from`).
   - Calculates content token overlap ratio against the source text.
4. **Evidence Snippet Extraction:**
   - Extracts the specific sentence from the raw source chunk exhibiting the highest semantic alignment to serve as immediate proof.
5. **Fabrication Detection:**
   - If a citation marker in the text does not exist in the retrieved chunk metadata (e.g. hallucinated `[99]`), it is flagged immediately as `FABRICATED_CITATION`.

---

## Sample Verified Outputs

### Task 1: Answer with Citations
**Query:**
> *"What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?"*

**Answer:**
> *"Under Clause 6.1.1.2 of the National Building Code of India, walls separating a boiler room from other parts of a building must provide a minimum fire resistance rating of 120 minutes (2 hours) [1]. All exterior and interior openings connecting to the boiler room must be protected by self-closing fire doors having an equivalent fire resistance rating of not less than 120 minutes [1] [Volume 1, PDF p. 314-315]."*

### Task 2: Citation-to-Metadata Mapping
| Marker | Chunk ID | Document | Vol | Page | Clause | Section Heading | Sim Score |
|:---:|:---|:---|:---:|:---:|:---|:---|---:|
| `[1]` | `nbc-v1-p314-001012` | `nationalbuilding01.pdf` | 1 | 314-315 | 6.1.1.2 | Part 4 — Fire and Life Safety | 0.8264 |
| `[2]` | `nbc-v1-p279-000915` | `nationalbuilding01.pdf` | 1 | 279-280 | 3.4.7.3 | Part 4 — Fire and Life Safety | 0.7682 |

### Task 3: Claim Verification Audit
- **Claim 1:** Separating walls require minimum 120-minute (2-hour) fire resistance rating [1].  
  - **Status:** `VERIFIED`  
  - **Evidence:** `"...shall be provided with fire doors, to be kept b) All exterior openings in a boiler room or..."`  
- **Claim 2:** Openings must be protected by self-closing fire doors with equal rating [1].  
  - **Status:** `VERIFIED`  
  - **Evidence:** `"...shall be protected by a fire assembly of not less than 2 hours rating."`  
- **Audit Accuracy:** `100% (2/2 claims verified, 0 fabricated citations)`

### Task 4: No-Source Fallback Case
**Query (Out of Domain):**
> *"What are the statutory guidelines for registering a trademark under the Indian Trade Marks Act?"*

**Response:**
> *"I don't have enough information to answer that reliably based on the provided National Building Code context."*

- **Attached Citations:** `[]` (0 markers)
- **Fabricated Citations:** `0`
- **Behavior:** Complete refusal without hallucinated source attribution.

---

## Generated Artifacts

- [`outputs/citations/sample_cited_answers.json`](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/citations/sample_cited_answers.json) — Full machine-readable payload containing cited answers, metadata mappings, claim verification reports, and fallback cases.
- [`outputs/citations/citation_verification_report.md`](file:///Users/abdulrahmanshihabuddin/Documents/college/SW_S66_BuildGuard_AI_Kalvium-Community/outputs/citations/citation_verification_report.md) — Comprehensive markdown verification audit report.
