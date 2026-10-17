# Business Entity Resolution — Reproducible Code

This directory contains the runnable Phase 3 blocking, Phase 4 validation matcher, and frozen Phase 5 inference pipeline. It ships with the challenge submission outputs under `output/`.

## Requirements

- Python 3.11 or newer
- NumPy (`python -m pip install -r code/business_entity_resolution/requirements.txt`)
- The challenge TSVs in `dataset/train/` and `dataset/test/` (not redistributed in this package)
- Approximately 5 GB of free space for the two TSV outputs; candidate output is about 4 GB

Run commands from the extracted submission root, alongside the challenge `dataset/` directory. Ensure `dataset/splits/` exists before inference (the scripts use it for temporary disk-backed indexes).

Run the synthetic edge-case checks with:

```bash
python code/business_entity_resolution/src/test_phase5_edge_cases.py
```

## Reproduce inference outputs

The frozen Phase 4 model coefficients and decision threshold are stored beside the scripts in `src/phase4_matcher_metrics.json`. Run:

```bash
python code/business_entity_resolution/src/phase5_test_inference.py
```

The script applies the frozen Phase 3 candidate routes and Phase 4 scorer to every test Source 1 row and overwrites `output/candidate_pairs.tsv` and `output/matching_results.tsv`. Each candidate list is the sorted target set passed into the feature builder and scorer. Every input Source 1 ID receives one output row, with an empty list where no candidate or match is produced.

Validate the two outputs without loading the multi-gigabyte candidate file into memory:

```bash
python code/business_entity_resolution/src/validate_phase5_outputs.py
```

## Reproduce validation and model training

The fixed Phase 2 holdout IDs are included as `src/validation_source1_ids.txt`. Create `dataset/splits/` if needed, then copy that file to `dataset/splits/validation_source1_ids.txt`; the scripts create/update metrics under `dataset/splits/`.

```bash
python code/business_entity_resolution/src/phase3_blocking_benchmark.py
python code/business_entity_resolution/src/phase4_pairwise_matcher.py
```

The Phase 4 training script deterministically samples training queries, creates route-based features, fits L2-regularized logistic regression, chooses a threshold on a training-only calibration subset, and evaluates the fixed holdout. Re-training updates `dataset/splits/phase4_matcher_metrics.json`; copy that file beside `phase5_test_inference.py` if you intend to use the newly trained model for inference.

The included `phase2_validation_manifest.json` records how the fixed validation IDs were derived. Training and test data remain external and must be supplied by the challenge organizer.

## Source map

- `phase3_blocking_benchmark.py` — normalization, target indexes, candidate routes, and validation benchmark.
- `phase4_pairwise_matcher.py` — feature construction, pairwise logistic scorer, calibration, and holdout evaluation.
- `phase5_test_inference.py` — full test inference and output writing.
- `validate_phase5_outputs.py` — streaming coverage, uniqueness, ID validity, and candidate-subset checks.
- `phase4_matcher_metrics.json` — frozen coefficients, threshold, and validation metrics.
- `validation_source1_ids.txt`, `phase2_validation_manifest.json` — reproducible Phase 2 split assets.

The scripts use only challenge-provided data; they make no network calls and use no pretrained model.

Exact blocking routes skip normalized-empty values. This prevents suffix-only names such as `Private Inc` from creating a shared empty-core block. Target IDs are checked for source prefix, canonical numeric suffix, and 32-bit packing bounds.


