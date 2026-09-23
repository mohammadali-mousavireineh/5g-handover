# Temporal Feature Representations for 5G Handover Prediction

Reproducibility repository for the manuscript:

**Temporal Feature Representations for 5G Handover Prediction: Cross-Dataset Evaluation with Proxy and Event-Log Labels**

Authors: Mohammadali Mousavireineh and Seyedmohammad Mousavi.

## What is included

- Exploratory eight-stage benchmark across D1, D2, and D3.
- Confirmatory D3 natural-prevalence evaluation with session-isolated train/validation/test splits.
- Train-only negative sampling for the confirmatory D3 experiment.
- Validation-only pipeline and threshold selection.
- Derived exploratory tables already used in the manuscript.
- A paper-level confirmatory D3 summary table matching Table 5 of the manuscript.

Raw public datasets are **not redistributed** here. Download them from their original sources and follow their licenses.

## Repository layout

```text
5g-handover/
├── README.md
├── requirements.txt
├── CITATION.cff
├── configs/
│   └── paper_experiment_config.json
├── docs/
│   └── REPRODUCIBILITY.md
├── results/
│   ├── ... exploratory result tables ...
│   └── confirmatory_d3/
│       └── paper_table_confirmatory_d3_rounded.csv
└── src/
    ├── multi_dataset_handover_feature_optimization_v4_memory_safe.py
    ├── d3_confirmatory_natural_test_runner.py
    └── datasets/
        ├── d1_anatel_ufjf_raw/
        ├── d2_urban_multi_operator/
        └── d3_oran_7_2_handover_events/
```

> The current pipeline resolves dataset paths relative to `src/`, so place downloaded datasets under `src/datasets/` exactly as shown above.

## Install

Python 3.10 or 3.11 is recommended for the TensorFlow stack used in the experiments.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Exploratory paper run

```bash
python src/multi_dataset_handover_feature_optimization_v4_memory_safe.py
```

Paper seeds: `1, 7, 21, 42, 100`.

The exploratory benchmark compares raw windows, temporal engineering, ANOVA selection, PCA, ANOVA+PCA, direct LSTM/GRU/TCN/Transformer models, individual deep features, and combined deep features. Exploratory D3 uses an event-enriched memory-safe modeling subset and is reported as descriptive rather than as a deployment-prevalence estimate.

## Confirmatory D3 paper run

```bash
python src/d3_confirmatory_natural_test_runner.py
```

Quick one-seed diagnostic:

```bash
python src/d3_confirmatory_natural_test_runner.py --quick
```

The confirmatory runner:

1. assigns complete D3 session groups to train, validation, and test before sampling;
2. samples negative windows only in training;
3. preserves natural prevalence in validation and test;
4. compares three pre-specified pipelines;
5. selects each decision threshold and the final pipeline on validation data; and
6. evaluates untouched test sessions and reports probability, classification, and event-level metrics.

The runner generates detailed per-seed CSV/JSON outputs, thresholds, natural-test predictions, PR curves, and aggregate tables in `src/d3_confirmatory_natural_test_outputs/`.

## Key manuscript-level confirmatory result

Across the five validation-selected untouched D3 tests, the manuscript reports:

- mean positive prevalence: **1.42%**
- precision: **0.3988 ± 0.0354**
- recall: **0.6587 ± 0.0150**
- F1: **0.4959 ± 0.0262**
- average precision: **0.4231 ± 0.0357**
- event-level recall: **0.8811 ± 0.0160**

The exploratory sampled D3 benchmark reached F1 = **0.9218 ± 0.0075** for temporal engineering + Random Forest. The contrast is intentional: it demonstrates how prevalence, sampling, and model-selection protocol affect reported performance.

## Data sources

- **D1:** ANATEL/UFJF drive-test files distributed with the `jpshlima/lstm-handover` materials. The study derives a future PCI/serving-cell-change proxy. Redistribution remains subject to the original source license.
- **D2:** Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments, Mendeley Data DOI `10.17632/dx5xyyfz2y.1`. The study derives a future CellID-change proxy.
- **D3:** Mobility Dataset from a 7.2 O-RAN deployment, Mendeley Data DOI `10.17632/khxgr6m8wz.1`. The study aligns `handover_events_*.txt` with `neigh_measurements_*.txt` and builds an event-log-based future handover target.

See `docs/REPRODUCIBILITY.md` for protocol details and expected outputs.

## AI-assisted development disclosure

OpenAI ChatGPT was used as an AI-assisted tool for code drafting/debugging and limited language/structural editing. The authors executed the code, inspected the outputs, verified reported numerical values, and remain fully responsible for the methodology, results, interpretation, and repository content.

## License

Source code in this repository is released under the MIT License. Dataset licenses remain with the original dataset providers.
