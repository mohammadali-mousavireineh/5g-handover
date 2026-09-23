# Reproducibility guide

## 1. Environment

Install dependencies from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For the TensorFlow version family used by this project, Python 3.10/3.11 is recommended.

## 2. Dataset placement

The current scripts resolve dataset paths relative to the `src` directory. Place downloaded files here:

```text
src/datasets/
├── d1_anatel_ufjf_raw/
│   ├── drive_test_measurements01.csv
│   ├── drive_test_measurements02.csv
│   └── drive_test_measurements03.csv
├── d2_urban_multi_operator/
│   └── processed_dataset.csv
└── d3_oran_7_2_handover_events/
    ├── handover_events_1.txt ... handover_events_6.txt
    └── neigh_measurements_1.txt ... neigh_measurements_6.txt
```

Do not mix multiple alternative processed D2 files in the same folder unless you intend them all to be concatenated.

## 3. Exploratory benchmark

Run:

```bash
python src/multi_dataset_handover_feature_optimization_v4_memory_safe.py
```

Paper-grade seeds are `[1, 7, 21, 42, 100]`. Window size is 10 samples and the prediction horizon is 5 samples. D1 uses a stratified random split and is treated as a pilot/consistency analysis. D2 and exploratory D3 use group-based splitting when feasible.

For the large D3 exploratory benchmark, the memory-safe pipeline keeps positives and samples negatives to make the multi-stage comparison tractable. Therefore these exploratory D3 scores are not interpreted as natural-prevalence deployment estimates.

## 4. Confirmatory D3 protocol

Run:

```bash
python src/d3_confirmatory_natural_test_runner.py
```

The runner uses seeds `[1, 7, 21, 42, 100]` and implements the validity-focused protocol reported in the manuscript:

- 148 eligible session groups are split before any class sampling.
- Test-group fraction: 20%.
- Validation-group fraction: 15%.
- Negative sampling is restricted to training windows.
- Training negatives are capped at four per positive and total sampled training windows are capped at 30,000.
- Validation and test keep natural prevalence.
- Three fixed candidate pipelines are considered: raw windows + Random Forest; temporal-engineered windows + Random Forest; combined LSTM/GRU/TCN/Transformer latent features + Random Forest.
- Each candidate threshold is selected on validation F1.
- The final candidate is selected on validation F1.
- The test partition is not used for model or threshold selection.
- Natural validation/test inference is chunked to avoid constructing a multi-gigabyte engineered matrix.

## 5. Confirmatory outputs

The runner creates `src/d3_confirmatory_natural_test_outputs/` containing, among other files:

```text
confirmatory_natural_test_all_seeds.csv
confirmatory_natural_test_aggregate.csv
validation_selected_model_test_results.csv
confirmatory_run_config.json
dataset_and_protocol_report.json
feature_report.json
seed_*/validation_metrics.csv
seed_*/test_metrics.csv
seed_*/selected_model_from_validation.json
seed_*/*_validation_threshold_curve.csv
seed_*/*_natural_test_predictions.csv.gz
```

The repository commits the compact paper-level summary table under `results/confirmatory_d3/`. Large generated models, memory maps, and prediction files are intentionally not version-controlled; they are reproducible by running the provided script on the public D3 data.

## 6. Expected paper-level results

The validation-selected pipeline across the five untouched tests should reproduce approximately:

| Metric | Mean ± SD |
|---|---:|
| Positive prevalence | 0.0142 (mean) |
| Precision | 0.3988 ± 0.0354 |
| Recall | 0.6587 ± 0.0150 |
| F1 | 0.4959 ± 0.0262 |
| Average precision | 0.4231 ± 0.0357 |
| MCC | 0.5033 ± 0.0221 |
| Specificity | 0.9855 ± 0.0022 |
| False positives / 1000 negatives | 14.46 ± 2.23 |
| Event-level recall | 0.8811 ± 0.0160 |

Small platform-dependent numerical differences can occur in deep-learning training, but the protocol and qualitative comparison should remain the same.

## 7. Interpretation

Do not compare the exploratory D3 F1 near 0.92 with the confirmatory D3 F1 near 0.50 as if they came from the same test distribution. The former comes from an event-enriched exploratory evaluation; the latter comes from untouched natural-prevalence session groups with validation-only model and threshold selection. The difference is one of the manuscript's methodological findings.
