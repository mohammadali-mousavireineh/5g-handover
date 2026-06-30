# Experiments

The main reproducible experiment is launched from the repository root:

```bash
python anatel_grid_compare.py
```

The `legacy_feature_generation/` folder preserves the adapted scripts used to train/load LSTM, GRU, and TCN RSRP predictors and generate the processed feature bases. These scripts are retained for transparency and traceability, while the cleaned comparison pipeline is implemented in `src/handover_compare/`.
