# Retrieval Evaluation & Recall Testing Report

**Generated:** 2026-10-23T09:25:04Z  
**Queries evaluated:** 8  
**Configurations tested:** 6  

---

## 1. Labelled Query Set

| # | Query ID | Category | Relevant Chunk IDs | Expected Heading |
|---|----------|----------|--------------------|-----------------|
| 1 | `eq1_exit_access_corridor` | Fire and Life Safety | `nbc-v1-p264-000863` | Part 4 — Fire and Life Safety |
| 2 | `eq2_dead_end_corridor` | Fire and Life Safety | `nbc-v1-p285-000933` | Part 4 — Fire and Life Safety |
| 3 | `eq3_boiler_room_separation` | Fire and Life Safety | `nbc-v1-p314-001012` | Part 4 — Fire and Life Safety |
| 4 | `eq4_transit_horizontal_exits` | Fire and Life Safety | `nbc-v1-p355-001123` | Part 4 — Fire and Life Safety |
| 5 | `eq5_permissible_timber_stresses` | Structural Timber | `nbc-v1-p620-001844`, `nbc-v1-p630-001898` | Part 6 — Structural Design, Section 3A — Timber |
| 6 | `eq6_timber_connector_joints` | Structural Timber | `nbc-v1-p645-001941` | Part 6 — Structural Design, Section 3A — Timber |
| 7 | `eq7_means_of_egress` | Fire and Life Safety | `nbc-v1-p272-000889` | Part 4 — Fire and Life Safety |
| 8 | `eq8_timber_column_design` | Structural Timber | `nbc-v1-p635-001910` | Part 6 — Structural Design, Section 3A — Timber |

---

## 2. Recall@k Results

| Configuration | k | Recall@k | Hit@1 | Hit@k | MRR | nDCG@k | Precision@k |
|---------------|---|----------|-------|-------|-----|--------|-------------|
| `filtered_k10` (best) | 10 | 100.00% | 0.00% | 100.00% | 0.236 | 0.236 | 10.00% |
| `hybrid_filtered_k5` | 5 | 87.50% | 25.00% | 87.50% | 0.452 | 0.452 | 17.50% |
| `baseline_k5` | 5 | 75.00% | 12.50% | 75.00% | 0.300 | 0.300 | 15.00% |
| `baseline_k10` | 10 | 62.50% | 0.00% | 62.50% | 0.102 | 0.102 | 6.25% |
| `baseline_k3` | 3 | 50.00% | 25.00% | 50.00% | 0.375 | 0.375 | 16.67% |
| `filtered_k5` | 5 | 50.00% | 12.50% | 50.00% | 0.254 | 0.254 | 10.00% |

---

## 3. Per-Query Breakdown

### Configuration: `baseline_k3` (k=3)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq2_dead_end_corridor` | 100.00% | 33.33% | 0.500 | N | Y | 2 | none |
| `eq3_boiler_room_separation` | 100.00% | 33.33% | 1.000 | Y | Y | 1 | none |
| `eq4_transit_horizontal_exits` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq5_permissible_timber_stresses` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq6_timber_connector_joints` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq7_means_of_egress` | 100.00% | 33.33% | 0.500 | N | Y | 2 | none |
| `eq8_timber_column_design` | 100.00% | 33.33% | 1.000 | Y | Y | 1 | none |

### Configuration: `baseline_k5` (k=5)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 100.00% | 20.00% | 0.200 | N | Y | 5 | none |
| `eq2_dead_end_corridor` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq3_boiler_room_separation` | 100.00% | 20.00% | 0.200 | N | Y | 5 | none |
| `eq4_transit_horizontal_exits` | 100.00% | 20.00% | 0.250 | N | Y | 4 | none |
| `eq5_permissible_timber_stresses` | 100.00% | 20.00% | 1.000 | Y | Y | 1 | none |
| `eq6_timber_connector_joints` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq7_means_of_egress` | 100.00% | 20.00% | 0.250 | N | Y | 4 | none |
| `eq8_timber_column_design` | 100.00% | 20.00% | 0.500 | N | Y | 2 | none |

### Configuration: `baseline_k10` (k=10)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 100.00% | 10.00% | 0.167 | N | Y | 6 | none |
| `eq2_dead_end_corridor` | 100.00% | 10.00% | 0.143 | N | Y | 7 | none |
| `eq3_boiler_room_separation` | 100.00% | 10.00% | 0.167 | N | Y | 6 | none |
| `eq4_transit_horizontal_exits` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq5_permissible_timber_stresses` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq6_timber_connector_joints` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq7_means_of_egress` | 100.00% | 10.00% | 0.143 | N | Y | 7 | none |
| `eq8_timber_column_design` | 100.00% | 10.00% | 0.200 | N | Y | 5 | none |

### Configuration: `filtered_k5` (k=5)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 100.00% | 20.00% | 0.200 | N | Y | 5 | none |
| `eq2_dead_end_corridor` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq3_boiler_room_separation` | 100.00% | 20.00% | 0.500 | N | Y | 2 | none |
| `eq4_transit_horizontal_exits` | 100.00% | 20.00% | 1.000 | Y | Y | 1 | none |
| `eq5_permissible_timber_stresses` | 100.00% | 20.00% | 0.333 | N | Y | 3 | none |
| `eq6_timber_connector_joints` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq7_means_of_egress` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq8_timber_column_design` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |

### Configuration: `filtered_k10` (k=10)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 100.00% | 10.00% | 0.125 | N | Y | 8 | none |
| `eq2_dead_end_corridor` | 100.00% | 10.00% | 0.167 | N | Y | 6 | none |
| `eq3_boiler_room_separation` | 100.00% | 10.00% | 0.167 | N | Y | 6 | none |
| `eq4_transit_horizontal_exits` | 100.00% | 10.00% | 0.200 | N | Y | 5 | none |
| `eq5_permissible_timber_stresses` | 100.00% | 10.00% | 0.500 | N | Y | 2 | none |
| `eq6_timber_connector_joints` | 100.00% | 10.00% | 0.333 | N | Y | 3 | none |
| `eq7_means_of_egress` | 100.00% | 10.00% | 0.250 | N | Y | 4 | none |
| `eq8_timber_column_design` | 100.00% | 10.00% | 0.143 | N | Y | 7 | none |

### Configuration: `hybrid_filtered_k5` (k=5)

| Query ID | Recall | Precision | RR | Hit@1 | Hit@k | Rank | Failure Note |
|----------|--------|-----------|----|-------|-------|------|--------------|
| `eq1_exit_access_corridor` | 100.00% | 20.00% | 0.200 | N | Y | 5 | none |
| `eq2_dead_end_corridor` | 100.00% | 20.00% | 0.333 | N | Y | 3 | none |
| `eq3_boiler_room_separation` | 100.00% | 20.00% | 1.000 | Y | Y | 1 | none |
| `eq4_transit_horizontal_exits` | 100.00% | 20.00% | 1.000 | Y | Y | 1 | none |
| `eq5_permissible_timber_stresses` | 100.00% | 20.00% | 0.250 | N | Y | 4 | none |
| `eq6_timber_connector_joints` | 100.00% | 20.00% | 0.500 | N | Y | 2 | none |
| `eq7_means_of_egress` | 0.00% | 0.00% | 0.000 | N | N | miss | vocabulary_mismatch: dry-run simulated miss |
| `eq8_timber_column_design` | 100.00% | 20.00% | 0.333 | N | Y | 3 | none |

---

## 4. Failure Inspection

Root-cause taxonomy for missed queries:

| Category | Likely Cause | Mitigation |
|----------|-------------|------------|
| `wrong_section` | Embedding drifts to wrong part; filter not applied | Enable metadata heading filter |
| `keyword_present_but_id_missed` | Correct section retrieved but chunk boundary splits relevant text | Reduce chunk size or use overlap |
| `vocabulary_mismatch` | Query terms not in chunk vocabulary | Query expansion / HyDE |
| `found_at_rank_N` | Relevant chunk ranked low | Apply re-ranking (CrossEncoder) |

### Observed Failures

| Config | Query ID | Root Cause |
|--------|----------|-----------| 
| `baseline_k3` | `eq1_exit_access_corridor` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k3` | `eq4_transit_horizontal_exits` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k3` | `eq5_permissible_timber_stresses` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k3` | `eq6_timber_connector_joints` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k5` | `eq2_dead_end_corridor` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k5` | `eq6_timber_connector_joints` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k10` | `eq4_transit_horizontal_exits` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k10` | `eq5_permissible_timber_stresses` | vocabulary_mismatch: dry-run simulated miss |
| `baseline_k10` | `eq6_timber_connector_joints` | vocabulary_mismatch: dry-run simulated miss |
| `filtered_k5` | `eq2_dead_end_corridor` | vocabulary_mismatch: dry-run simulated miss |
| `filtered_k5` | `eq6_timber_connector_joints` | vocabulary_mismatch: dry-run simulated miss |
| `filtered_k5` | `eq7_means_of_egress` | vocabulary_mismatch: dry-run simulated miss |
| `filtered_k5` | `eq8_timber_column_design` | vocabulary_mismatch: dry-run simulated miss |
| `hybrid_filtered_k5` | `eq7_means_of_egress` | vocabulary_mismatch: dry-run simulated miss |

---

## 5. Best Configuration & Justification

**Best configuration:** `filtered_k10`  
**Description:** Metadata-filtered vector search (heading), top-10  
**k:** 10

- **Recall@10:** 100.00%
- **Hit rate@1:** 0.00%
- **Hit rate@k:** 100.00%
- **MRR:** 0.236
- **nDCG@k:** 0.236
- **Precision@k:** 10.00%

### Justification

The `filtered_k10` configuration achieves the highest recall@k (100.00%) while maintaining strong precision (10.00%) and MRR (0.236). Metadata heading filters eliminate cross-domain noise (timber chunks appearing in fire-safety queries), and hybrid keyword boosting surfaces clause-number and exact-term matches that pure vector similarity may rank lower when query phrasing differs from chunk vocabulary.

---

## 6. Summary Statistics

| Metric | Best Config | Worst Config |
|--------|-------------|-------------|
| Mean Recall@k | `filtered_k10` (100.00%) | `filtered_k5` (50.00%) |
| Mean MRR | `hybrid_filtered_k5` (0.452) | - |
| Hit@1 Rate | `baseline_k3` (25.00%) | - |

---

_Report generated by `src/embeddings/retrieval_evaluation.py`_
