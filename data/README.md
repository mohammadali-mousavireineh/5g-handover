# Data

This folder contains the processed feature bases used in the associated paper:

**Enhanced 5G Handover Prediction over Real-Network RSRP Traces Using Lightweight Temporal Neural Representations**

## Files

- `processed/anatel_concatbases.csv`: LSTM-generated predicted-RSRP windows and labels.
- `processed/anatel_concatbases_gru.csv`: GRU-generated predicted-RSRP windows and labels.
- `processed/anatel_concatbases_tcn.csv`: TCN-generated predicted-RSRP windows and labels.

Each CSV has 50 temporal feature columns and one binary label column.

## Important note

The raw drive-test files are not included in this GitHub-ready package. The processed feature bases are included so the main classifier comparison can be reproduced directly.
