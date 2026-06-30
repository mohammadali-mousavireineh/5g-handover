# Reproducibility Guide

This repository is designed so the main comparison can be reproduced without regenerating the neural-network RSRP predictions.

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## 2. Default experiment

```bash
python anatel_grid_compare.py
```

Expected output files:

- `results/anatel_grid_results_all.csv`
- `results/anatel_grid_results_best.csv`
- `results/anatel_comparison_report.html`
- `results/figures/*.png`

## 3. Main protocol

- Stratified train/test split
- Test size: 25%
- Random seed: 42
- GridSearchCV on the training set only
- Held-out test set is not resampled
- Default scoring: F1-score

## 4. Optional SMOTE protocol

```bash
python anatel_grid_compare.py --use-smote --cv 5 --full-grid
```

SMOTE is inserted inside an imbalanced-learn pipeline, so resampling occurs only inside training/CV folds. The held-out test set remains untouched.

## 5. Feature-generation scripts

The feature-generation scripts are preserved in `experiments/legacy_feature_generation/`. They require the original raw drive-test files and TensorFlow/Keras:

```bash
pip install -r requirements-training.txt
```

Raw drive-test CSV files are not included in this GitHub-ready package.
