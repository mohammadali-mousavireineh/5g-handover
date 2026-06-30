# Model Artifacts

This folder contains saved Keras model structures and weights used to generate the processed RSRP feature bases:

- `lstm/`: baseline LSTM RSRP predictor artifacts.
- `gru/`: GRU RSRP predictor artifacts.
- `tcn/`: TCN RSRP predictor artifacts.

The main repository experiment does not retrain these neural models. It uses the processed feature bases in `data/processed/` and evaluates downstream handover classifiers.

To regenerate feature bases, use the scripts in `experiments/legacy_feature_generation/` after placing the original raw drive-test CSV files in the expected folder.
