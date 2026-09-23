# Temporal Feature Representations for 5G Handover Prediction

Reproducibility repository for the manuscript:

**Temporal Feature Representations for 5G Handover Prediction: Cross-Dataset Evaluation with Proxy and Event-Log Labels**

Authors: Mohammadali Mousavireineh and Seyedmohammad Mousavi.

## Repository status

This package is based directly on the archived **v11 confirmatory natural-test repository** used to produce the saved confirmatory D3 results. The computational source files and result tables are preserved from that package. This submission refresh updates repository documentation, citation metadata, the license text, and the documented dataset paths; it does **not** rewrite the experimental pipeline or regenerate the reported results.

## What is included

- Exploratory multi-dataset pipeline for D1, D2, and sampled D3.
- Confirmatory D3 natural-prevalence runner.
- Saved five-seed confirmatory outputs used in the manuscript.
- Saved protocol/configuration JSON files.
- Derived result tables and manuscript figures.
- Documentation of the confirmatory protocol and data setup.

Raw datasets are not redistributed. Obtain them from their original sources and follow the source licenses.

## Repository layout

```text
5g-handover/
├── README.md
├── CITATION.cff
├── LICENSE
├── VERSION.md
├── requirements.txt
├── src/
│   ├── multi_dataset_handover_feature_optimization_v3_oran_jsonl_fixed.py
│   ├── multi_dataset_handover_feature_optimization_v4_memory_safe.py
│   ├── d3_confirmatory_natural_test_runner.py
│   └── datasets/
│       └── README.md
├── results/
│   └── confirmatory_d3_natural_test/
│       ├── confirmatory_natural_test_all_seeds.csv
│       ├── confirmatory_natural_test_aggregate.csv
│       ├── validation_selected_model_test_results.csv
│       ├── validation_selected_pipeline_aggregate.csv
│       ├── confirmatory_pairwise_wilcoxon.csv
│       ├── confirmatory_run_config.json
│       ├── dataset_and_protocol_report.json
│       └── feature_report.json
├── figures/
└── docs/
    ├── CONFIRMATORY_D3_PROTOCOL.md
    ├── REPRODUCIBILITY.md
    ├── RESULTS_PROVENANCE.md
    ├── RELATED_WORK.md
    └── references.bib
```

## Data setup

The current experimental code resolves dataset paths **relative to `src/`**. Therefore place the downloaded datasets here:

```text
src/datasets/
├── d1_anatel_ufjf_raw/
├── d2_urban_multi_operator/
└── d3_oran_7_2_handover_events/
```

Expected sources:

- **D1:** ANATEL/UFJF drive-test data distributed with the `jpshlima/lstm-handover` materials. The study constructs a future PCI/serving-cell-change proxy. Redistribution remains subject to the original source license.
- **D2:** Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments, Mendeley Data DOI `10.17632/dx5xyyfz2y.1`. The study constructs a future CellID-change proxy.
- **D3:** Mobility Dataset from a 7.2 O-RAN deployment, Mendeley Data DOI `10.17632/khxgr6m8wz.1`. The study aligns `neigh_measurements_*.txt` with `handover_events_*.txt` and constructs a future event-log-based handover target.

## Installation

Python 3.10 or 3.11 is recommended for broad compatibility with the TensorFlow stack used in the experiments.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Exploratory multi-dataset benchmark

Run:

```bash
python src/multi_dataset_handover_feature_optimization_v4_memory_safe.py
```

Paper seeds are `1, 7, 21, 42, 100`.

The exploratory benchmark compares raw windows, temporal engineering, ANOVA selection, PCA, ANOVA+PCA, direct LSTM/GRU/TCN/Transformer models, individual neural features, and combined neural features. The sampled D3 exploratory result is retained as an exploratory comparison and should not be interpreted as a natural-prevalence deployment estimate.

The script writes newly generated exploratory outputs under:

```text
src/multi_dataset_outputs/
```

## Confirmatory D3 natural-prevalence evaluation

Run:

```bash
python src/d3_confirmatory_natural_test_runner.py
```

For a single-seed diagnostic run:

```bash
python src/d3_confirmatory_natural_test_runner.py --quick
```

The confirmatory protocol:

1. splits complete session groups into train, validation, and test before sampling;
2. applies negative sampling only to the training partition;
3. preserves natural prevalence in validation and test;
4. evaluates three pre-specified candidate pipelines;
5. selects each threshold and the final pipeline using validation data; and
6. evaluates untouched test sessions once per seed.

Newly generated confirmatory outputs are written under:

```text
src/d3_confirmatory_natural_test_outputs/
```

The archived five-seed outputs used in the manuscript are preserved under:

```text
results/confirmatory_d3_natural_test/
```

## Main confirmatory D3 result

Across the five validation-selected untouched test partitions, the archived outputs give:

- mean positive prevalence: **0.014195 ± 0.000415** (~1.42%)
- precision: **0.398840 ± 0.035391**
- recall: **0.658709 ± 0.014973**
- F1: **0.495922 ± 0.026174**
- average precision: **0.423062 ± 0.035721**
- MCC: **0.503269 ± 0.022090**
- specificity: **0.985543 ± 0.002229**
- false positives per 1,000 negative windows: **14.456853 ± 2.229090**
- event-level recall: **0.881145 ± 0.015955**
- median first-alert lead: **5 samples**

Validation selected raw-window Random Forest for four seeds and temporal-engineering Random Forest for one seed. The saved candidate mean F1 values are approximately 0.4956 (Raw RF), 0.4952 (Temporal RF), and 0.4816 (Combined RF).

See `docs/RESULTS_PROVENANCE.md` for the exact source files behind these numbers.

## Interpretation

D1 and D2 are proxy-label tasks. D3 uses event-log-derived handover labels. The exploratory sampled D3 benchmark and the confirmatory natural-prevalence D3 experiment answer different evaluation questions and should not be directly interpreted as equivalent deployment estimates.

## AI-assisted development disclosure

OpenAI ChatGPT was used as a generative AI-assisted tool for code drafting/debugging and limited language/structural editing. The authors executed the code, inspected the outputs, checked reported numerical values against saved experiment outputs, and remain fully responsible for the methodology, results, interpretation, and repository content.

## License

Source code in this repository is released under the MIT License. Dataset licenses remain with their original providers.
