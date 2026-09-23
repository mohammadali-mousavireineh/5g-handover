# Reproducibility guide

## Environment

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The experiments use NumPy, pandas, scikit-learn, Matplotlib, TensorFlow/Keras, and joblib.

## Dataset paths

The existing source code resolves dataset paths from the directory containing the Python scripts. Therefore the correct local structure is:

```text
src/datasets/
├── d1_anatel_ufjf_raw/
├── d2_urban_multi_operator/
└── d3_oran_7_2_handover_events/
```

Do not place the datasets only at repository-root `datasets/`; the archived source files do not resolve that location.

## Exploratory run

```bash
python src/multi_dataset_handover_feature_optimization_v4_memory_safe.py
```

Key paper-level settings in the archived pipeline include:

- seeds: `1, 7, 21, 42, 100`
- window size: `10`
- prediction horizon: `5` samples
- ANOVA feature cap: `40`
- PCA retained variance: `0.95`
- deep epochs: `25`
- deep batch size: `32`

New exploratory outputs are written to `src/multi_dataset_outputs/`.

## Confirmatory D3 run

```bash
python src/d3_confirmatory_natural_test_runner.py
```

Quick diagnostic:

```bash
python src/d3_confirmatory_natural_test_runner.py --quick
```

The archived configuration is also saved in:

```text
results/confirmatory_d3_natural_test/confirmatory_run_config.json
```

The confirmatory protocol first assigns complete session groups to train/validation/test, applies negative sampling only to training, preserves natural validation/test prevalence, selects threshold and candidate pipeline on validation, and then evaluates untouched test sessions.

New confirmatory outputs are written to `src/d3_confirmatory_natural_test_outputs/`. The saved outputs used for the manuscript are under `results/confirmatory_d3_natural_test/`.

## Important distinction

The exploratory sampled D3 benchmark and the confirmatory D3 natural-prevalence experiment are intentionally different evaluations. The archived sampled D3 F1 near 0.92 should not be substituted for the confirmatory deployment-oriented F1 near 0.50.
