# Confirmatory D3 results provenance

The confirmatory D3 numerical values reported in the current manuscript are supported by the archived five-seed output files in:

```text
results/confirmatory_d3_natural_test/
```

## Primary source files

- `validation_selected_model_test_results.csv` — one validation-selected untouched-test result for each seed.
- `validation_selected_pipeline_aggregate.csv` — aggregate metrics across the five validation-selected test results.
- `confirmatory_natural_test_all_seeds.csv` — all three confirmatory candidates for every seed.
- `confirmatory_natural_test_aggregate.csv` — candidate-level aggregate metrics.
- `confirmatory_pairwise_wilcoxon.csv` — paired Wilcoxon comparisons across seeds.
- `confirmatory_run_config.json` — saved confirmatory experiment configuration.
- `dataset_and_protocol_report.json` — saved dataset/protocol counts and natural prevalence information.
- `feature_report.json` — selected/rejected input-feature report.

## Validation-selected five-seed aggregate

The saved aggregate file reports:

- precision: `0.398840 ± 0.035391`
- recall: `0.658709 ± 0.014973`
- F1: `0.495922 ± 0.026174`
- MCC: `0.503269 ± 0.022090`
- average precision: `0.423062 ± 0.035721`
- ROC-AUC: `0.985660 ± 0.001762`
- specificity: `0.985543 ± 0.002229`
- false positives per 1,000 negative windows: `14.456853 ± 2.229090`
- event-level recall: `0.881145 ± 0.015955`
- median first-alert lead: `5 samples`
- mean natural positive ratio: `0.014195 ± 0.000415`

Validation selected raw-window Random Forest in four seeds and temporal-engineering Random Forest in one seed.

## Candidate mean F1

From `confirmatory_natural_test_aggregate.csv`:

- Raw-window Random Forest: approximately `0.4956`
- Temporal-engineering Random Forest: approximately `0.4952`
- Combined deep-feature Random Forest: approximately `0.4816`

These archived values are the source for the rounded manuscript table and prose.
