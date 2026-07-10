# Deep Temporal Feature Optimization for 5G Handover Prediction

This repository contains the reproducibility package for the manuscript:

**Deep Temporal Feature Optimization for 5G Handover Prediction Using Proxy and Event-Log Labels Across Three Mobility Datasets**

## Dataset structure

Place datasets under:

```text
datasets/
├── d1_anatel_ufjf_raw/
├── d2_urban_multi_operator/
└── d3_oran_7_2_handover_events/
```

Expected inputs:

- **D1:** ANATEL/UFJF raw drive-test CSV files from the `jpshlima/lstm-handover` repository. Target: future PCI-change proxy.
- **D2:** Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments. Target: future CellID-change proxy.
- **D3:** Mobility Dataset from a 7.2 O-RAN deployment. Use paired `neigh_measurements_*.txt` and `handover_events_*.txt`. Target: event-log-based future handover label matched by UE and timestamp.

## Run

```bash
pip install -r requirements.txt
python src/multi_dataset_handover_feature_optimization_v4_memory_safe.py
```

For a quick smoke test, edit `RUN_SEEDS = [42]`. For paper-level results, use five seeds: `[1, 7, 21, 42, 100]`.

## Output

The script writes repeated-seed result tables and stage comparisons. The `results/` and `figures/` directories include the analyzed outputs used in the revised manuscript.

## Important interpretation

D1 and D2 are proxy-label tasks. D3 uses event-log-based labels. Do not describe D1/D2 results as official handover-command prediction.