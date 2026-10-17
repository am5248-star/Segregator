# ML Challenge 2026: Business Entity Resolution

**Team Name:** Not provided  
**Team Members:** Not provided  
**Submission Date:** 2026-10-17

---

## 1. Executive Summary

This solution resolves each Source 1 business record against Sources 2 and 3 using a scalable two-stage pipeline: multi-route blocking generates a compact candidate set, then a candidate-conditioned logistic regression ranks candidate links. On the fixed Source 1 validation holdout, the frozen matcher achieved macro F0.5 of **0.4689**. The Phase 5 test run wrote one candidate row and one prediction row for each of the **1,732,544** test Source 1 records, including empty lists, and passed streaming structural checks.

## 2. Methodology

### 2.1 Problem Analysis

The task has no shared cross-source key. Business names may differ in punctuation, legal suffixes, token order, abbreviations, or spelling; addresses may be abbreviated, reordered, incomplete, or expressed with landmarks. Records may have no counterpart or multiple counterparts. Test data also includes France, which is absent from training, so country handling must be open-set.

The provided files are tab-separated, and the ground truth stores comma-separated target IDs inside a TSV field. The implementation explicitly parses TSVs and preserves IDs as strings at file boundaries. Internally, numeric target suffixes are packed with the source number for compact indexes; the implementation checks that conversion is lossless.

### 2.2 Solution Strategy

**Approach Type:** Multi-route blocking plus supervised pairwise classifier.  
**Core contribution:** Union complementary name and address routes, retain per-route evidence as model features, and persist the exact candidate set supplied to the scorer.

The workflow is: assess the four training files; create a deterministic query-level validation split; benchmark candidate-generation routes; fit and calibrate a matcher using only training-partition queries; freeze the selected model and threshold; apply the frozen pipeline to test Source 1; then validate output coverage and constraints.

Normalization applies Unicode NFKC and case folding, extracts alphanumeric tokens, and canonicalizes common legal and street suffixes. Exact blocking routes skip normalized-empty keys, preventing suffix-only names such as `Private Inc` from matching every target with an empty canonical name. Target IDs are checked for source prefix and lossless numeric packing. No transliteration or external business lookup is used. The test country label is treated as a string, not restricted to the training countries.

## 3. Candidate Generation (Blocking)

The frozen union uses 12 routes: exact country-scoped name, exact address, canonical/core name, unordered core name, canonical address, rare name tokens, rare address tokens, expanded rare-token variants, bounded character n-grams, name-token pairs, and address-token pairs. Exact name/address-style routes are country-scoped; rare-token fallback routes can match across country labels. Posting lists are capped at 200 to control common-key explosions, and route-specific hit caps bound expansion.

On the fixed validation split, the union generated **75,649,390 pairs** for **441,363** Source 1 queries (about **171.4 candidates per query**). Candidate recall by true target was **0.9214**, and positive Source 1 entity coverage was **0.9888**. Relative to the full validation Cartesian product, candidate generation reduced comparisons by about **99.9983%**. These are validation measurements; the unlabeled test set cannot provide recall measurement.

The test run wrote **307,033,596 candidate pairs** across **1,732,544 Source 1 rows**. The candidate TSV records the actual sorted target IDs passed to the Phase 4 feature builder and scorer, so predicted matches are a subset by construction and by output validation.

## 4. Matching Model

**Name and address features:** 12 binary route-membership indicators (including exact, canonical, token, character n-gram, and token-pair evidence).  
**Other features:** target source indicator (S3 versus S2) and log-scaled per-query candidate count.  
**Model:** L2-regularized logistic regression, fitted with deterministic batch gradient descent.  
**Sampling:** Up to 30 deterministic hard negative candidates per fit query; all generated positive candidates are retained.  
**Threshold:** Selected to maximize macro F0.5 on a deterministic training-only calibration subset, then evaluated on the untouched fixed holdout.  
**Frozen inference threshold:** 0.1912855214.

The fixed validation split contains **441,363** Source 1 rows. The model training process sampled **20,000** non-validation queries, using 15,941 for fitting and 4,059 for calibration. The saved model contains 14 feature weights plus an intercept.

## 5. Results & Error Analysis

- **Validation macro F0.5:** **0.468871**.
- **Validation micro precision / recall / F0.5:** **0.2240 / 0.4703 / 0.2502**.
- **Candidate recall / positive entity coverage:** **0.9214 / 0.9888**.
- **Test Source 1 rows:** 1,732,544.
- **Test candidate pairs:** 307,033,596.
- **Test predicted matches:** 12,531,141.
- **Rows with no candidates / no predictions:** 465 / 137,474.

No labels exist for the test set, so test precision and recall are unknown. The validation results show a precision-heavy F0.5 objective still has a material false-positive risk: micro precision is 0.2240. Likely error modes include common business names or addresses that collide within a country, incomplete addresses, and entities with near-duplicate records. These are risk categories inferred from the matching setup, not manually confirmed test errors.

A held-out sample of missed validation links showed substantial name variation and inconsistent address overlap, especially for Source 3 examples. This supports retaining multiple independent name and address routes. Blocking still missed some true target links (candidate recall 0.9214), which no downstream classifier can recover. The separate Phase 5 similarity-feature experiment scored slightly below the frozen Phase 4 model and was not substituted into test inference.

All test output checks passed using a streaming validator: one ordered row per test Source 1 ID, valid existing target IDs, no duplicate IDs, and every predicted match contained in that Source 1 row's candidate set. The official submission validator also passed matching-file structure and coverage checks. Test labels were unavailable, so these checks validate format and consistency rather than match correctness.

## 6. Conclusion

The completed pipeline makes an otherwise infeasible all-pairs comparison tractable, preserves broad candidate recall with complementary routes, and applies a calibrated supervised scorer. The held-out evaluation and fully covered test outputs are reproducible from the included source and frozen artifacts. The principal improvement opportunity is reducing false positives while retaining the high candidate recall, followed by label-based test evaluation when ground truth becomes available.

## Appendix

### A. Code Artefacts

The submission ZIP contains runnable source under `code/business_entity_resolution/src/`, a `README.md`, a NumPy dependency file, the frozen Phase 4 metrics/weights, the fixed Phase 2 validation IDs and manifest, and both generated TSV outputs under `output/`. Run test inference with `python code/business_entity_resolution/src/phase5_test_inference.py` from the extracted project root after placing the challenge data under `dataset/test/`. Run the streaming consistency check with `python code/business_entity_resolution/src/validate_phase5_outputs.py`.

The validation and training entry points are `phase3_blocking_benchmark.py` and `phase4_pairwise_matcher.py`. See the code README for setup and commands. Challenge input datasets are not included in the package.

### B. Additional Results

- Phase 4 validation macro F0.5 by target source: S2 **0.4830**; S3 **0.4231**.
- Validation micro precision / recall by target source: S2 **0.1820 / 0.6345**; S3 **0.3949 / 0.3167**.
- An additional similarity-feature matcher experiment produced macro F0.5 **0.4679** and was retained only as an experiment; the Phase 4 model remained frozen for Phase 5 inference.\n- Eight synthetic edge-case checks passed for Unicode normalization, suffix-only names, malformed and overflow IDs, posting caps, empty candidates, and street aliases.
- Test output sizes: `candidate_pairs.tsv` approximately **3.98 GB**; `matching_results.tsv` approximately **184 MB**.



