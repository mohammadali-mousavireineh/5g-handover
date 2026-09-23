# Dataset placement

The experimental scripts resolve dataset paths relative to the `src/` directory. Place the downloaded raw files under this directory structure:

```text
src/datasets/
├── d1_anatel_ufjf_raw/
├── d2_urban_multi_operator/
└── d3_oran_7_2_handover_events/
```

## D1

ANATEL/UFJF drive-test data distributed with the `jpshlima/lstm-handover` materials. Follow the original license and citation requirements. The study derives a future PCI/serving-cell-change proxy.

## D2

Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments.

Mendeley Data DOI: `10.17632/dx5xyyfz2y.1`

The study derives a future CellID-change proxy.

## D3

Mobility Dataset from a 7.2 O-RAN deployment.

Mendeley Data DOI: `10.17632/khxgr6m8wz.1`

Expected paired files include:

```text
handover_events_1.txt ... handover_events_6.txt
neigh_measurements_1.txt ... neigh_measurements_6.txt
```

The study aligns the event logs and measurement reports by session/UE and timestamp to construct the future handover-event target.

Raw datasets are intentionally not committed to this repository.
