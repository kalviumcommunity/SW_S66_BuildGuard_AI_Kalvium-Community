# Source Citation & Attribution Audit Report

**Generated:** `2026-10-26T14:49:33Z`  
**System:** BuildGuard AI — National Building Code of India (NBC) Assistant  

## 1. Executive Summary

To make regulatory AI answers verifiable and prevent citation fabrication:
1. **In-Text Citation Placement:** Every factual claim is referenced to a specific source chunk (e.g. `[1]`).
2. **Metadata Traceability:** Each citation marker maps to a real source document, volume number, physical page range, clause, and section heading.
3. **Claim-by-Claim Verification:** An automated audit verifies claim sentences against raw source chunk texts.
4. **No-Source Fallback:** When supporting context is absent or weak, the system returns a safe refusal without fabricating citations.

## 2. Task 1: Generated Answer with In-Text Citations

**Query:** *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?*

### Grounded Response
> Under Clause 6.1.1.2 of the National Building Code of India, walls separating a boiler room from other parts of a building must provide a minimum fire resistance rating of 120 minutes (2 hours) [1]. All exterior and interior openings connecting to the boiler room must be protected by self-closing fire doors having an equivalent fire resistance rating of not less than 120 minutes [1] [Volume 1, PDF p. 314-315].

## 3. Task 2: Citation-to-Metadata Mapping

| Marker | Chunk ID | Document | Vol | Page | Clause | Section Heading | Sim Score | Re-rank | Text Snippet |
|:---:|:---|:---|:---:|:---:|:---|:---|---:|---:|:---|
| `[1]` | `nbc-v1-p314-001012` | nationalbuilding01.pdf | 1 | 314-315 | 6.1.1.2 | Part 4 — Fire and Life Sa | 0.8264 | — | 120 min fire resistance rating wall. Such areas . shall be provid… |
| `[2]` | `nbc-v1-p279-000915` | nationalbuilding01.pdf | 1 | 279-280 | 3.4.7.3 | Part 4 — Fire and Life Sa | 0.7682 | — | 3.4.8.1 Air conditioning and mechanical ventilation —_3.4.8.2.4 T… |

## 4. Task 3: Claim-by-Claim Verification Audit

Every claim in the answer was inspected against the full text of its cited source:

| # | Claim Statement | Cited Marker | Status | Evidence from Source Chunk |
|:---:|:---|:---:|:---:|:---|
| 1 | Under Clause 6.1.1.2 of the National Building Code of India, walls separating a … | `[1]` | **VERIFIED** | *"shall be provided with fire doors, to be kept b) All exterior openings in a boiler room or…"* |
| 2 | All exterior and interior openings connecting to the boiler room must be protect… | `[1]` | **VERIFIED** | *"shall be provided with fire doors, to be kept b) All exterior openings in a boiler room or…"* |

### Verification Summary
- **Overall Verification Accuracy:** **100%**
- **Verified Claims:** `2/2`
- **Unsupported Claims:** `0`
- **Fabricated Citations Detected:** `None (0)`
- **Audit Status:** `PASSED`

## 5. Task 4: No-Source Fallback (Avoid Fabricated Citations)

When an out-of-domain or ungrounded question is posed, the system must refuse to answer and **must not fabricate citations**:

**Test Query (Out of Domain):** *What are the statutory guidelines for registering a trademark under the Indian Trade Marks Act?*  
**Fallback Mode:** `no_source_fallback`  
**System Response:**
> *"I don't have enough information to answer that reliably based on the provided National Building Code context."*

- **Citations Attached:** `[]` (Count: **0**)  
- **Fabricated Citations:** **0** (Safe Refusal Confirmed)  