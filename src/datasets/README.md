# Dataset placement

The scripts in this repository resolve dataset paths relative to the `src` directory. Download the public data from the original providers and place files under `src/datasets/`:

- `src/datasets/d1_anatel_ufjf_raw/` — D1 ANATEL/UFJF drive-test CSVs.
- `src/datasets/d2_urban_multi_operator/` — D2 urban multi-operator dataset (prefer the intended processed CSV only).
- `src/datasets/d3_oran_7_2_handover_events/` — D3 paired `neigh_measurements_*.txt` and `handover_events_*.txt` files.

D2 source: Mendeley Data DOI `10.17632/dx5xyyfz2y.1`.

D3 source: Mendeley Data DOI `10.17632/khxgr6m8wz.1`.

D1 should be obtained from the original `jpshlima/lstm-handover` materials and redistributed only if its source license permits it.
