# Methods

## Feature bases

The experiment compares three processed ANATEL feature bases:

1. **LSTM-generated predicted-RSRP windows**: the baseline recurrent representation.
2. **GRU-generated predicted-RSRP windows**: a lighter recurrent representation using fewer gates than LSTM.
3. **TCN-generated predicted-RSRP windows**: a temporal-convolutional representation using causal/dilated convolutions.

Each row contains 50 predicted RSRP values followed by a binary handover label.

## Why GRU and TCN?

LSTM is a strong baseline for time-series modeling, but it can be computationally heavier because of its multiple gating mechanisms. GRU uses a simpler recurrent structure, which can provide a lower-complexity alternative while preserving temporal memory. TCN replaces recurrence with temporal convolutions, enabling parallelizable sequence modeling and receptive-field expansion through dilation.

These properties make GRU and TCN relevant candidates for lightweight predictive handover pipelines, especially when deployment at edge/MEC/O-RAN components is considered.

## Evaluation protocol

The default protocol uses:

- Stratified train/test split.
- Untouched held-out test data.
- GridSearchCV on the training split only.
- Optional SMOTE inside the pipeline, which means resampling happens only inside training/CV folds.

## Metrics

The repository reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Because the positive handover class is the minority class, F1-score and recall should be interpreted alongside accuracy.
