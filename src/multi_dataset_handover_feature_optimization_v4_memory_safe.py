"""
Multi-Dataset 5G Handover / Serving-Cell-Change Feature Optimization Pipeline
================================================================================

One-click runner for three dataset folders:

D1: ANATEL/UFJF raw drive-test files from jpshlima/lstm-handover
    Folder: datasets/d1_anatel_ufjf_raw/
    Put: drive_test_measurements01.csv, drive_test_measurements02.csv, drive_test_measurements03.csv
    Target: derived future PCI / serving-cell-change proxy, because these raw files normally do not expose an official HO label.

D2: Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments
    Folder: datasets/d2_urban_multi_operator/
    Put: processed_dataset.csv only, or the dataset CSV you want to evaluate.
    Target: derived future CellID / serving-cell-change proxy unless an official handover column is found.

D3: Mobility Dataset from a 7.2 O-RAN deployment
    Folder: datasets/d3_oran_7_2_handover_events/
    Put ALL matching pairs downloaded from Mendeley:
        handover_events_1.txt ... handover_events_6.txt
        neigh_measurements_1.txt ... neigh_measurements_6.txt
    Target: event-log-derived future handover label created from the dataset-provided handover_events files.
            If event/measurement timestamps cannot be aligned, the script reports a clear error.

Run:
    python multi_dataset_handover_feature_optimization.py

Outputs:
    multi_dataset_outputs/
        D1_.../
        D2_.../
        D3_.../
        cross_dataset_best_by_stage.csv
        cross_dataset_deep_architecture_comparison.csv
        cross_dataset_dataset_report.csv

Notes:
    - For paper-grade runs, keep RUN_SEEDS = [1, 7, 21, 42, 100].
    - For quick debugging, set RUN_SEEDS = [42].
    - If a dataset has an official HO label column and auto-detection fails, set official_target_col manually in DATASET_CONFIGS.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import re
import warnings
import gc
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    GRU,
    Conv1D,
    GlobalAveragePooling1D,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


# =========================================================
# Global execution settings
# =========================================================
RUN_SEEDS = [1, 7, 21, 42, 100]      # paper run
# RUN_SEEDS = [42]                   # uncomment for fast debugging

OUTPUT_ROOT = Path(__file__).resolve().parent / "multi_dataset_outputs"

WINDOW_SIZE = 10
HANDOVER_HORIZON = 5
TEST_SIZE = 0.20

FEATURE_SELECTION_METHOD = "anova"  # "anova" or "mutual_info"
FEATURE_SELECTION_K = 40
PCA_VARIANCE = 0.95

DEEP_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 5

INCLUDE_LOCATION_FEATURES = True
INCLUDE_CELL_ID_AS_FEATURE = False
INCLUDE_LOW_CARDINALITY_CATEGORICALS = True
MAX_CATEGORICAL_UNIQUE = 20

# Memory-safe settings for very large datasets such as D3 O-RAN.
# D3 has hundreds of thousands of measurement rows and only ~1-2% positives.
# Keeping all positives and a controlled number of negatives prevents laptop RAM crashes
# while preserving the handover-event class for model comparison.
LARGE_DATASET_NEGATIVE_TO_POSITIVE_RATIO = 4
LARGE_DATASET_MAX_MODELING_ROWS = 40000
RBF_SVM_MAX_TRAIN_SAMPLES = 12000  # above this, use LinearSVC as the SVM baseline


# =========================================================
# Dataset configuration
# =========================================================
@dataclass
class DatasetConfig:
    name: str
    data_path: str
    csv_glob: str = "*.csv"
    official_target_col: str | None = None
    official_target_mode: str = "future_horizon"  # "future_horizon" or "as_is"
    cell_id_col: str | None = None
    session_col: str | None = None
    time_col: str | None = None
    split_mode: str = "group_by_session"          # "random" or "group_by_session"
    include_location_features: bool = INCLUDE_LOCATION_FEATURES
    include_cell_id_as_feature: bool = INCLUDE_CELL_ID_AS_FEATURE
    window_size: int = WINDOW_SIZE
    handover_horizon: int = HANDOVER_HORIZON
    notes: str = ""
    # Optional memory-safe modeling subset. Use only for huge, heavily imbalanced datasets.
    # If enabled, all positives are kept and negatives are sampled up to neg_pos_ratio * positives,
    # bounded by max_modeling_rows.
    neg_pos_ratio: int | None = None
    max_modeling_rows: int | None = None


DATASET_CONFIGS = [
    DatasetConfig(
        name="D1_ANATEL_UFJF_raw_drive_test_proxy",
        data_path="datasets/d1_anatel_ufjf_raw",
        csv_glob="drive_test_measurements*.csv",
        official_target_col=None,
        cell_id_col="PCI",       # if your files use a different column, set it here
        session_col=None,         # source file will be used if no session column exists
        time_col=None,            # row order will be used if no time column exists
        split_mode="random",      # keep D1 as pilot consistency check
        notes=(
            "ANATEL/UFJF-related raw drive-test files from the lstm-handover repository. "
            "The raw files used here do not usually contain an official Layer-3 handover label, "
            "so the target is derived as a future PCI/serving-cell-change proxy."
        ),
    ),
    DatasetConfig(
        name="D2_urban_multi_operator_public_proxy",
        data_path="datasets/d2_urban_multi_operator",
        csv_glob="*.csv",
        official_target_col=None,
        cell_id_col="CellID",
        session_col="SessionID",
        time_col="Timestamp",
        split_mode="group_by_session",
        notes=(
            "Urban Multi-Operator QoE-Aware public dataset. Unless a true handover label is supplied, "
            "the target is derived from future CellID changes. Use processed_dataset.csv only to avoid mixing raw and processed data."
        ),
    ),
    DatasetConfig(
        name="D3_ORAN_7_2_event_log_handover_labels",
        data_path="datasets/d3_oran_7_2_handover_events",
        csv_glob="*.txt",
        official_target_col="handover_event_marker",
        official_target_mode="future_horizon",
        cell_id_col=None,
        session_col="oran_session_id",
        time_col="oran_sample_order",
        split_mode="group_by_session",
        neg_pos_ratio=LARGE_DATASET_NEGATIVE_TO_POSITIVE_RATIO,
        max_modeling_rows=LARGE_DATASET_MAX_MODELING_ROWS,
        notes=(
            "Mobility Dataset from a 7.2 O-RAN deployment. Put all handover_events_*.txt and neigh_measurements_*.txt files here. "
            "The script aligns dataset-provided handover event logs with neighbor measurements and creates a future-event label. "
            "This is stronger than a serving-cell-change proxy because it is derived from provided handover-event files."
        ),
    ),
]


# =========================================================
# Utilities
# =========================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_name(col: str) -> str:
    name = str(col).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "col"


def canonical(col: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(col).lower())


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def find_column(data: pd.DataFrame, exact_candidates=None, contains_candidates=None, exclude_contains=None) -> str | None:
    exact_candidates = [canonical(x) for x in (exact_candidates or [])]
    contains_candidates = [canonical(x) for x in (contains_candidates or [])]
    exclude_contains = [canonical(x) for x in (exclude_contains or [])]
    canon_map = {canonical(c): c for c in data.columns}

    for cand in exact_candidates:
        if cand in canon_map:
            return canon_map[cand]

    for c in data.columns:
        cc = canonical(c)
        if any(ex in cc for ex in exclude_contains):
            continue
        if any(cand in cc for cand in contains_candidates):
            return c
    return None



def _read_json_lines(path: Path) -> pd.DataFrame:
    """Read newline-delimited JSON logs and flatten nested 5G/O-RAN fields."""
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} at line {line_no}: {exc}")
    if not records:
        raise ValueError(f"No JSON records found in {path}")
    return pd.json_normalize(records, sep="_")


def read_table_any(path: Path) -> pd.DataFrame:
    """Read CSV/TXT/JSONL. O-RAN files are newline-delimited JSON, not normal CSV."""
    with open(path, "rb") as f:
        raw = f.read(4096)
    sample = raw.decode("utf-8", errors="ignore")
    first_line = sample.splitlines()[0].strip() if sample.splitlines() else ""

    # The O-RAN Mendeley files are JSON Lines: one JSON object per line.
    # Do this before delimiter detection, because JSON has many commas and can be misread as CSV.
    if first_line.startswith("{") and first_line.endswith("}"):
        return _read_json_lines(path)

    candidates = [",", "\t", ";", "|"]
    counts = {sep: first_line.count(sep) for sep in candidates}
    best_sep, best_count = max(counts.items(), key=lambda kv: kv[1])

    read_kwargs = {"low_memory": False}
    if best_count > 0:
        read_kwargs["sep"] = best_sep
        read_kwargs["engine"] = "c"
    else:
        read_kwargs["sep"] = r"\s+"
        read_kwargs["engine"] = "python"

    try:
        return pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **read_kwargs)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def _series_to_order(s: pd.Series) -> pd.Series:
    """Convert a timestamp-like column into comparable numeric order."""
    num = coerce_numeric_series(s)
    if num.notna().sum() > len(s) * 0.5:
        return num
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() > len(s) * 0.5:
        return dt.astype("int64")
    return pd.Series(np.arange(len(s)), index=s.index)


def _detect_time_for_oran(df: pd.DataFrame) -> str | None:
    return find_column(
        df,
        exact_candidates=["Timestamp", "Time", "Datetime", "DateTime", "ts", "t", "slot", "frame", "sfn"],
        contains_candidates=["timestamp", "datetime", "time", "date", "slot", "frame", "sfn"],
        exclude_contains=["timezone"],
    )


def load_oran_7_2_event_dataset(folder: Path) -> pd.DataFrame:
    """
    Load the Mendeley 7.2 O-RAN dataset from paired files:
      - neigh_measurements_i.txt: measurement features (RSRP/RSRQ/SINR, etc.)
      - handover_events_i.txt: dataset-provided handover event log

    The returned table contains only measurement rows as model samples, plus a binary
    `handover_event_marker` aligned to the nearest measurement sample at each event time.
    Later, create_target_and_order() converts this event marker into a future-horizon
    prediction target. This avoids using serving-cell-change proxy for D3.
    """
    neigh_files = sorted(folder.glob("neigh_measurements_*.txt"))
    event_files = sorted(folder.glob("handover_events_*.txt"))
    if not neigh_files:
        raise FileNotFoundError(f"No neigh_measurements_*.txt files found in {folder}")
    if not event_files:
        raise FileNotFoundError(f"No handover_events_*.txt files found in {folder}")

    event_by_id: dict[str, Path] = {}
    for p in event_files:
        m = re.search(r"handover_events_(\d+)", p.name)
        if m:
            event_by_id[m.group(1)] = p

    frames = []
    print("\nLoading D3 O-RAN paired handover-event files")
    print(f"Folder: {folder.resolve()}")
    print(f"Neighbor files: {len(neigh_files)} | Event files: {len(event_files)}")

    for file_id, neigh_path in enumerate(neigh_files, start=1):
        m = re.search(r"neigh_measurements_(\d+)", neigh_path.name)
        pair_id = m.group(1) if m else str(file_id)
        event_path = event_by_id.get(pair_id)
        if event_path is None:
            print(f"  WARNING: no matching handover_events_{pair_id}.txt for {neigh_path.name}; skipping this pair.")
            continue

        neigh = read_table_any(neigh_path)
        events = read_table_any(event_path)
        neigh.columns = [safe_name(c) for c in neigh.columns]
        events.columns = [safe_name(c) for c in events.columns]

        neigh_time_col = _detect_time_for_oran(neigh)
        event_time_col = _detect_time_for_oran(events)
        if neigh_time_col is None or event_time_col is None:
            raise ValueError(
                f"Could not detect time columns for D3 pair {pair_id}. "
                f"Neighbor columns={list(neigh.columns)[:30]}, event columns={list(events.columns)[:30]}. "
                "Set/rename a timestamp column or inspect the TXT headers."
            )

        neigh = neigh.copy()
        events = events.copy()

        neigh_ue_col = find_column(neigh, exact_candidates=["UE", "Ue", "ue"], contains_candidates=["ue", "gnbcucpueid"])
        event_ue_col = find_column(events, exact_candidates=["UE", "Ue", "ue"], contains_candidates=["ue", "gnbcuef1apid", "gnbcucpueid"])
        if neigh_ue_col is None or event_ue_col is None:
            raise ValueError(
                f"Could not detect UE columns for D3 pair {pair_id}. "
                f"Neighbor columns={list(neigh.columns)[:40]}, event columns={list(events.columns)[:40]}."
            )

        neigh["oran_pair_id"] = f"oran_pair_{pair_id}"
        neigh["oran_ue_id"] = neigh[neigh_ue_col].astype(str)
        neigh["oran_session_id"] = neigh["oran_pair_id"].astype(str) + "__UE_" + neigh["oran_ue_id"].astype(str)
        neigh["source_file"] = neigh_path.name
        neigh["source_file_id"] = int(pair_id) if str(pair_id).isdigit() else file_id
        neigh["source_row_order"] = np.arange(len(neigh))
        neigh["oran_sample_order"] = _series_to_order(neigh[neigh_time_col])
        neigh["handover_event_marker"] = 0

        events["_event_order"] = _series_to_order(events[event_time_col])
        events["_event_ue"] = events[event_ue_col].astype(str)

        marked = 0
        # Align each event only to measurement rows for the SAME UE.
        for ue_value, ev_group in events.dropna(subset=["_event_order"]).groupby("_event_ue", sort=False):
            n_mask = neigh["oran_ue_id"].astype(str) == str(ue_value)
            if not n_mask.any():
                continue
            n_idx = neigh.index[n_mask].to_numpy()
            n_order = neigh.loc[n_idx, "oran_sample_order"].to_numpy()
            valid = pd.notna(n_order)
            if not valid.any():
                continue
            n_idx = n_idx[valid]
            n_order = n_order[valid]
            sort_pos = np.argsort(n_order)
            sorted_order = n_order[sort_pos]
            sorted_idx = n_idx[sort_pos]

            for ev in ev_group["_event_order"].to_numpy():
                pos = int(np.searchsorted(sorted_order, ev, side="left"))
                candidates = []
                if 0 <= pos < len(sorted_order):
                    candidates.append(pos)
                if 0 <= pos - 1 < len(sorted_order):
                    candidates.append(pos - 1)
                if not candidates:
                    continue
                nearest_pos = min(candidates, key=lambda k: abs(sorted_order[k] - ev))
                nearest_idx = sorted_idx[nearest_pos]
                neigh.loc[nearest_idx, "handover_event_marker"] = 1
                marked += 1

        print(
            f"  Pair {pair_id}: {neigh_path.name} {neigh.shape}, {event_path.name} {events.shape}, "
            f"UE_col={neigh_ue_col}/{event_ue_col}, marked_events={marked}"
        )
        frames.append(neigh)

    if not frames:
        raise ValueError("No valid D3 O-RAN pairs were loaded. Download matching handover_events_i and neigh_measurements_i files.")

    data = pd.concat(frames, ignore_index=True)
    print(f"D3 combined measurement table: {data.shape}")
    print(f"D3 event markers on measurement rows: {int(data['handover_event_marker'].sum())}")
    if int(data["handover_event_marker"].sum()) == 0:
        raise ValueError(
            "D3 handover_events files were loaded but no events could be aligned to neighbor measurements. "
            "Open one handover_events_i.txt and one neigh_measurements_i.txt and check their timestamp columns."
        )
    return data

def load_dataset_csvs(cfg: DatasetConfig) -> pd.DataFrame:
    folder = Path(__file__).resolve().parent / cfg.data_path
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}\nCreate it and put the dataset files there.")

    # Special paired loader for D3 O-RAN: handover_events_i.txt + neigh_measurements_i.txt.
    if "oran" in canonical(cfg.name) and any(folder.glob("handover_events_*.txt")) and any(folder.glob("neigh_measurements_*.txt")):
        return load_oran_7_2_event_dataset(folder)

    files = sorted(folder.glob(cfg.csv_glob))
    if not files:
        # Fallback for users who put TXT files while the pattern still says CSV.
        files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.txt")))
    if not files:
        raise FileNotFoundError(f"No dataset files found in {folder} using pattern {cfg.csv_glob}")

    frames = []
    print(f"\nLoading dataset: {cfg.name}")
    print(f"Folder: {folder.resolve()}")
    for file_id, path in enumerate(files, start=1):
        df = read_table_any(path)
        df.columns = [safe_name(c) for c in df.columns]
        df["source_file"] = path.name
        df["source_file_id"] = file_id
        df["source_row_order"] = np.arange(len(df))
        frames.append(df)
        print(f"  {path.name}: {df.shape}")

    data = pd.concat(frames, ignore_index=True)
    unnamed = [c for c in data.columns if canonical(c).startswith("unnamed")]
    if unnamed:
        data = data.drop(columns=unnamed)
    print(f"Combined shape: {data.shape}")
    return data


def detect_session_column(data: pd.DataFrame, cfg: DatasetConfig) -> str:
    if cfg.session_col and cfg.session_col in data.columns:
        return cfg.session_col
    col = find_column(
        data,
        exact_candidates=["SessionID", "Session", "RouteID", "TripID", "LogID", "DriveID", "TrajectoryID"],
        contains_candidates=["session", "route", "trip", "log", "trajectory", "drive"],
    )
    return col or "source_file"


def detect_time_column(data: pd.DataFrame, cfg: DatasetConfig) -> str | None:
    if cfg.time_col and cfg.time_col in data.columns:
        return cfg.time_col
    return find_column(
        data,
        exact_candidates=["Timestamp", "Time", "ElapsedTime", "Datetime", "DateTime", "sample_order"],
        contains_candidates=["timestamp", "datetime", "elapsed", "time", "date"],
        exclude_contains=["timezone"],
    )


def detect_cell_id_column(data: pd.DataFrame, cfg: DatasetConfig) -> str | None:
    if cfg.cell_id_col and cfg.cell_id_col in data.columns:
        return cfg.cell_id_col
    return find_column(
        data,
        exact_candidates=[
            "PCI", "CellID", "Cell_Id", "Cell", "CID", "ECI", "eNodeB", "eNBID", "gNBID", "NodeID",
            "ServingCell", "ServingCellID", "CellName", "NR_PCI", "LTE_PCI", "serving_cell", "serving_cell_id",
        ],
        contains_candidates=["servingcellid", "servingcell", "cellid", "cellid", "pci", "eci", "cid", "enbid", "enodeb", "gnbid", "nodeid", "cellname"],
        exclude_contains=["latitude", "longitude", "lat", "lon", "second", "neighbor", "neighbour"],
    )


def detect_official_target_column(data: pd.DataFrame, cfg: DatasetConfig) -> str | None:
    if cfg.official_target_col and cfg.official_target_col in data.columns:
        return cfg.official_target_col

    target_keywords = [
        "handover", "handoff", "hand_off", "hoevent", "holabel", "hotarget", "ho", "event", "handoverevent",
        "servingcellchange", "cellchange", "label", "target",
    ]
    candidates = []
    for col in data.columns:
        cc = canonical(col)
        if any(k in cc for k in target_keywords):
            if not any(bad in cc for bad in ["latitude", "longitude", "time", "timestamp", "session", "cellid", "pci", "rsrp", "rsrq", "sinr", "snr"]):
                candidates.append(col)

    # Pick a binary / nearly binary column first.
    for col in candidates:
        non_null = data[col].dropna()
        if len(non_null) == 0:
            continue
        if non_null.nunique() <= 3:
            return col
    return None


def binary_from_official_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if len(vals) == 0:
            return pd.Series(np.nan, index=s.index)
        if len(vals) == 1:
            return pd.Series(0, index=s.index)
        # assume largest value is positive
        return (s == vals[-1]).astype(int)

    y = s.astype(str).str.lower().str.strip()
    positive_exact = {"1", "true", "yes", "y", "handover", "ho", "event", "positive", "pos"}
    negative_exact = {"0", "false", "no", "n", "none", "normal", "negative", "neg", "nan", ""}

    def map_one(x: str) -> int:
        if x in positive_exact:
            return 1
        if x in negative_exact:
            return 0
        if "handover" in x or "handoff" in x or x.startswith("ho"):
            return 1
        return 0

    return y.apply(map_one).astype(int)


def future_event_target(event_values: np.ndarray, horizon: int) -> np.ndarray:
    n = len(event_values)
    out = np.full(n, np.nan)
    for i in range(0, n - horizon):
        out[i] = int(np.any(event_values[i + 1 : i + horizon + 1] == 1))
    return out


def create_target_and_order(data: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, dict]:
    data = data.copy()
    session_col = detect_session_column(data, cfg)
    time_col = detect_time_column(data, cfg)
    cell_col = detect_cell_id_column(data, cfg)
    official_col = detect_official_target_column(data, cfg)

    # Stable drive/session id. Combining with source_file prevents accidental collisions.
    data["drive_id"] = pd.factorize(data[session_col].astype(str) + "__" + data["source_file"].astype(str))[0] + 1

    if time_col is not None:
        t_num = coerce_numeric_series(data[time_col])
        if t_num.notna().sum() > len(data) * 0.5:
            data["sample_order"] = t_num
        else:
            t_dt = pd.to_datetime(data[time_col], errors="coerce")
            if t_dt.notna().sum() > len(data) * 0.5:
                data["sample_order"] = t_dt.astype("int64")
            else:
                data["sample_order"] = data["source_row_order"]
    else:
        data["sample_order"] = data["source_row_order"]

    target_source = ""
    result_parts = []

    if official_col is not None:
        data["official_event_binary"] = binary_from_official_series(data[official_col])
        for _, g in data.groupby("drive_id", sort=False):
            g = g.sort_values("sample_order").copy()
            ev = g["official_event_binary"].to_numpy(dtype="int32")
            if cfg.official_target_mode == "as_is":
                g["handover_target"] = ev
            elif cfg.official_target_mode == "future_horizon":
                g["handover_target"] = future_event_target(ev, cfg.handover_horizon)
            else:
                raise ValueError("official_target_mode must be 'as_is' or 'future_horizon'")
            result_parts.append(g.dropna(subset=["handover_target"]))
        data = pd.concat(result_parts, ignore_index=True)
        data["handover_target"] = data["handover_target"].astype(int)
        target_source = f"official_column:{official_col};mode:{cfg.official_target_mode}"
    else:
        if cell_col is None:
            raise ValueError(
                f"{cfg.name}: No official target column and no serving cell identifier were detected. "
                "Set official_target_col or cell_id_col in DATASET_CONFIGS."
            )
        for _, g in data.groupby("drive_id", sort=False):
            g = g.sort_values("sample_order").copy()
            cell = g[cell_col].astype(str).to_numpy()
            n = len(g)
            target = np.full(n, np.nan)
            for i in range(0, n - cfg.handover_horizon):
                future_cells = cell[i + 1 : i + cfg.handover_horizon + 1]
                target[i] = int(np.any(future_cells != cell[i]))
            g["handover_target"] = target
            result_parts.append(g.dropna(subset=["handover_target"]))
        data = pd.concat(result_parts, ignore_index=True)
        data["handover_target"] = data["handover_target"].astype(int)
        target_source = f"derived_future_cell_change:{cell_col}"

    if data["handover_target"].nunique() < 2:
        raise ValueError(f"{cfg.name}: target has only one class after processing. Check target/cell column and horizon.")

    info = {
        "session_col": session_col,
        "time_col": time_col if time_col else "source_row_order",
        "cell_col": cell_col,
        "official_target_col_detected": official_col,
        "target_source": target_source,
        "rows_after_target_creation": len(data),
        "positive_ratio_after_target_creation": float(data["handover_target"].mean()),
    }

    print("\nTarget ready")
    print(json.dumps(info, indent=2, default=str))
    print(data["handover_target"].value_counts().sort_index())
    return data, info



def maybe_make_modeling_subset(data: pd.DataFrame, cfg: DatasetConfig, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Create a memory-safe event-balanced modeling subset for very large datasets.

    This is mainly for D3 O-RAN, where the full table can create multi-gigabyte
    flattened engineered windows. The method keeps all positive handover-event rows
    whenever possible and samples negative rows. It is recorded in dataset_report.txt
    and run_config.json so the paper can describe it transparently.
    """
    info = {
        "modeling_subset_enabled": False,
        "rows_before_subset": int(len(data)),
        "rows_after_subset": int(len(data)),
        "positive_rows_before_subset": int((data["handover_target"] == 1).sum()) if "handover_target" in data.columns else None,
        "negative_rows_before_subset": int((data["handover_target"] == 0).sum()) if "handover_target" in data.columns else None,
        "neg_pos_ratio": cfg.neg_pos_ratio,
        "max_modeling_rows": cfg.max_modeling_rows,
    }

    if cfg.neg_pos_ratio is None and cfg.max_modeling_rows is None:
        return data, info
    if "handover_target" not in data.columns or data["handover_target"].nunique() < 2:
        return data, info

    pos = data[data["handover_target"] == 1]
    neg = data[data["handover_target"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return data, info

    max_rows = cfg.max_modeling_rows if cfg.max_modeling_rows is not None else len(data)
    ratio = cfg.neg_pos_ratio if cfg.neg_pos_ratio is not None else max(1, (max_rows - len(pos)) // max(len(pos), 1))

    # If positives alone exceed max_rows, sample positives too, but normally D3 positives are small enough.
    if len(pos) >= max_rows:
        pos_keep = pos.sample(n=max_rows // 2, random_state=seed)
        neg_keep_n = max_rows - len(pos_keep)
    else:
        pos_keep = pos
        neg_keep_n = min(len(neg), len(pos_keep) * ratio, max_rows - len(pos_keep))

    neg_keep = neg.sample(n=max(1, int(neg_keep_n)), random_state=seed)
    subset = pd.concat([pos_keep, neg_keep], ignore_index=True)
    subset = subset.sort_values(["drive_id", "sample_order"]).reset_index(drop=True)

    info.update({
        "modeling_subset_enabled": True,
        "rows_after_subset": int(len(subset)),
        "positive_rows_after_subset": int((subset["handover_target"] == 1).sum()),
        "negative_rows_after_subset": int((subset["handover_target"] == 0).sum()),
        "positive_ratio_after_subset": float(subset["handover_target"].mean()),
        "subset_note": "All positive handover-event rows are kept when possible; negative rows are sampled for memory-safe event-focused modeling.",
    })

    print("\nMemory-safe modeling subset enabled")
    print(json.dumps(info, indent=2))
    return subset, info


def prepare_feature_table(data: pd.DataFrame, cfg: DatasetConfig, cell_col: str | None, target_source: str) -> tuple[pd.DataFrame, list[str], dict]:
    data = data.copy()
    exclude = {"handover_target", "drive_id", "sample_order", "source_row_order", "source_file_id", "official_event_binary"}
    # Exclude identifiers/log metadata so the model learns radio context, not UE/file identity.
    for c in ["source_file", "UE", "oran_ue_id", "oran_pair_id", "oran_session_id", "topic", "CELL", "DU", "ENB"]:
        if c in data.columns:
            exclude.add(c)
    if not cfg.include_cell_id_as_feature and cell_col:
        exclude.add(cell_col)
    if target_source.startswith("official_column:"):
        official_col = target_source.split(";", 1)[0].split(":", 1)[1]
        exclude.add(official_col)

    location_keys = ["lat", "latitude", "lon", "long", "longitude", "gps"]
    id_like_keys = ["id", "ue", "session", "route", "trip", "log", "name", "imsi", "imei", "mcc", "mnc", "tac", "topic"]

    numeric_features, categorical_features = [], []
    for col in data.columns:
        if col in exclude:
            continue
        cc = canonical(col)
        if not cfg.include_location_features and any(k in cc for k in location_keys):
            continue
        if any(k in cc for k in id_like_keys) and not any(k in cc for k in ["pci", "rsrp", "rsrq", "sinr", "snr", "rssi", "cqi"]):
            continue
        num = coerce_numeric_series(data[col])
        if num.notna().mean() >= 0.70:
            data[col] = num.replace([np.inf, -np.inf], np.nan)
            med = data[col].median()
            data[col] = data[col].fillna(0 if pd.isna(med) else med)
            numeric_features.append(col)
        elif INCLUDE_LOW_CARDINALITY_CATEGORICALS:
            n_unique = data[col].nunique(dropna=True)
            if 1 < n_unique <= MAX_CATEGORICAL_UNIQUE:
                categorical_features.append(col)

    feature_cols = list(numeric_features)
    one_hot_cols = []
    for col in categorical_features:
        dummies = pd.get_dummies(data[col].astype(str).fillna("unknown"), prefix=col, dtype=int)
        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
        one_hot_cols.extend(dummies.columns.tolist())
    feature_cols.extend(one_hot_cols)

    non_constant = [c for c in feature_cols if data[c].nunique(dropna=False) > 1]
    dropped_constant = sorted(set(feature_cols) - set(non_constant))
    feature_cols = non_constant
    if not feature_cols:
        raise ValueError("No usable feature columns were detected.")

    info = {
        "numeric_features": numeric_features,
        "categorical_features_one_hot_encoded": categorical_features,
        "one_hot_feature_count": len(one_hot_cols),
        "dropped_constant_features": dropped_constant,
        "final_feature_count": len(feature_cols),
    }
    print(f"Final feature count: {len(feature_cols)}")
    return data, feature_cols, info


def add_temporal_engineered_features(data: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    data = data.sort_values(["drive_id", "sample_order"]).copy()
    engineered = []
    numeric_base = [c for c in base_cols if pd.api.types.is_numeric_dtype(data[c])]
    for col in numeric_base:
        grouped = data.groupby("drive_id", sort=False)[col]
        for lag in [1, 3, 5]:
            new_col = f"{col}_diff{lag}"
            data[new_col] = grouped.diff(lag)
            engineered.append(new_col)
        for win in [3, 5]:
            for stat, func in {
                "mean": lambda s: s.rolling(window=win, min_periods=1).mean(),
                "std": lambda s: s.rolling(window=win, min_periods=2).std(),
                "min": lambda s: s.rolling(window=win, min_periods=1).min(),
                "max": lambda s: s.rolling(window=win, min_periods=1).max(),
            }.items():
                new_col = f"{col}_roll{win}_{stat}"
                data[new_col] = grouped.transform(func)
                engineered.append(new_col)
    if engineered:
        data[engineered] = data.groupby("drive_id", sort=False)[engineered].transform(lambda g: g.fillna(0))
    return data, base_cols + engineered


def create_windows(data: pd.DataFrame, feature_cols: list[str], window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_parts, y_parts, groups, meta_rows = [], [], [], []
    for drive_id, group in data.groupby("drive_id", sort=False):
        group = group.sort_values("sample_order").copy()
        if len(group) < window_size:
            continue
        feats = group[feature_cols].to_numpy(dtype="float32")
        target = group["handover_target"].to_numpy(dtype="int32")
        order = group["sample_order"].to_numpy()
        for end_idx in range(window_size - 1, len(group)):
            start_idx = end_idx - window_size + 1
            X_parts.append(feats[start_idx:end_idx+1])
            y_parts.append(target[end_idx])
            groups.append(drive_id)
            meta_rows.append({"drive_id": drive_id, "start": str(order[start_idx]), "end": str(order[end_idx])})
    X = np.asarray(X_parts, dtype="float32")
    y = np.asarray(y_parts, dtype="int32")
    groups = np.asarray(groups)
    if len(y) == 0:
        raise ValueError("No windows created. Reduce window_size.")
    if len(np.unique(y)) < 2:
        raise ValueError("Windowed target has only one class.")
    return X, y, groups, pd.DataFrame(meta_rows)


def flatten_windows(X_seq: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    n, w, f = X_seq.shape
    cols = []
    for t in range(w):
        lag = w - 1 - t
        lag_name = "t0" if lag == 0 else f"t-{lag}"
        for feature in feature_cols:
            cols.append(f"{str(feature).replace(' ', '_').replace('(', '').replace(')', '')}_{lag_name}")
    return pd.DataFrame(X_seq.reshape(n, w * f), columns=cols)


def split_indices(y: np.ndarray, groups: np.ndarray, cfg: DatasetConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    if cfg.split_mode == "group_by_session" and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(idx, y, groups=groups))
        # If test accidentally has one class, fallback to stratified random for robustness.
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=y)
        return train_idx, test_idx
    return train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=y)


def scale_flat(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Memory-safe StandardScaler replacement.

    sklearn StandardScaler may allocate a huge temporary NaN mask on very wide matrices
    such as D3 engineered flattened windows. This implementation works in float32 and
    avoids that extra multi-GB allocation.
    """
    cols = X_train.columns
    Xtr = X_train.to_numpy(dtype=np.float32, copy=True)
    Xte = X_test.to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(Xtr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(Xte, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return pd.DataFrame(Xtr, columns=cols), pd.DataFrame(Xte, columns=cols)


def scale_seq(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Memory-safe sequence scaling over training timesteps only."""
    ntr, w, f = X_train.shape
    nte = X_test.shape[0]
    tr2 = X_train.reshape(-1, f).astype(np.float32, copy=True)
    te2 = X_test.reshape(-1, f).astype(np.float32, copy=True)
    np.nan_to_num(tr2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(te2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean = tr2.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = tr2.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    tr = ((tr2 - mean) / std).reshape(ntr, w, f)
    te = ((te2 - mean) / std).reshape(nte, w, f)
    return tr.astype("float32"), te.astype("float32")


def select_features(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k = min(FEATURE_SELECTION_K, X_train.shape[1])
    score_func = f_classif if FEATURE_SELECTION_METHOD == "anova" else mutual_info_classif
    selector = SelectKBest(score_func=score_func, k=k)
    Xtr = selector.fit_transform(X_train, y_train)
    Xte = selector.transform(X_test)
    cols = X_train.columns[selector.get_support()].tolist()
    scores = selector.scores_[selector.get_support()]
    info = pd.DataFrame({"feature": cols, "score": scores}).sort_values("score", ascending=False)
    return pd.DataFrame(Xtr, columns=cols), pd.DataFrame(Xte, columns=cols), info


def pca_extract(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=PCA_VARIANCE, random_state=0)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    cols = [f"PCA_{i+1}" for i in range(Xtr.shape[1])]
    return pd.DataFrame(Xtr, columns=cols), pd.DataFrame(Xte, columns=cols)


def get_classifiers(seed: int, n_samples: int | None = None) -> dict[str, Any]:
    # RBF SVM is very expensive on large datasets. For large training sets, use a linear SVM baseline.
    # It is still an SVM baseline, but it avoids impractical O(n^2) kernel memory/time.
    if n_samples is not None and n_samples > RBF_SVM_MAX_TRAIN_SAMPLES:
        svm_model = LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000)
    else:
        svm_model = SVC(kernel="rbf", random_state=seed, class_weight="balanced", cache_size=512)
    return {
        "Decision_Tree": DecisionTreeClassifier(random_state=seed, max_depth=10, class_weight="balanced"),
        "Random_Forest": RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1, class_weight="balanced"),
        "SVM": svm_model,
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic_Regression": LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_handover": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_handover": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_handover": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }


def add_result(rows: list[dict], dataset: str, seed: int, stage: str, feature_set: str, opt: str, family: str, model: str, classifier: str, y_true: np.ndarray, y_pred: np.ndarray, n_features: int, notes: str = ""):
    row = {
        "dataset": dataset,
        "seed": seed,
        "experiment_stage": stage,
        "feature_set": feature_set,
        "optimization_method": opt,
        "model_family": family,
        "model_name": model,
        "classifier": classifier,
        "n_features": n_features,
        "notes": notes,
    }
    row.update(metrics(y_true, y_pred))
    rows.append(row)
    print(f"{stage:42s} | {model:15s} | {classifier:20s} | F1_HO={row['f1_handover']:.4f}")


def run_classical(rows, dataset, seed, X_train, X_test, y_train, y_test, stage, feature_set, opt, notes=""):
    print(f"\n--- {stage} --- {X_train.shape}")
    for name, clf in get_classifiers(seed).items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        add_result(rows, dataset, seed, stage, feature_set, opt, "classical_ml", name, name, y_test, pred, X_train.shape[1], notes)


# =========================================================
# Deep models
# =========================================================
def build_lstm(input_shape):
    return Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(32, activation="relu", name="deep_features"),
        Dense(1, activation="sigmoid", name="output"),
    ], name="LSTM_model")


def build_gru(input_shape):
    return Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True), Dropout(0.2),
        GRU(32), Dropout(0.2),
        Dense(32, activation="relu", name="deep_features"),
        Dense(1, activation="sigmoid", name="output"),
    ], name="GRU_model")


def build_tcn(input_shape):
    return Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, padding="causal", dilation_rate=1, activation="relu"), Dropout(0.2),
        Conv1D(32, kernel_size=3, padding="causal", dilation_rate=2, activation="relu"), Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu", name="deep_features"),
        Dense(1, activation="sigmoid", name="output"),
    ], name="TCN_model")


def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    # Project input features to a small model dimension.
    x = Dense(64)(inputs)
    attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    ff = Dense(128, activation="relu")(x)
    ff = Dropout(0.2)(ff)
    ff = Dense(64)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu", name="deep_features")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs, outputs, name="Transformer_model")


def train_deep_models(rows, dataset, seed, X_train_seq, X_test_seq, y_train, y_test):
    builders = {
        "LSTM": build_lstm,
        "GRU": build_gru,
        "TCN": build_tcn,
        "Transformer": build_transformer,
    }
    train_features, test_features = {}, {}
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    for arch, builder in builders.items():
        print(f"\nTraining {arch}...")
        set_all_seeds(seed)
        model = builder(input_shape)
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        class_counts = pd.Series(y_train).value_counts().to_dict()
        total = len(y_train)
        class_weight = {0: total / (2 * class_counts.get(0, 1)), 1: total / (2 * class_counts.get(1, 1))}
        model.fit(
            X_train_seq, y_train,
            epochs=DEEP_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            callbacks=[es],
            verbose=0,
            shuffle=True,
            class_weight=class_weight,
        )
        prob = model.predict(X_test_seq, verbose=0).ravel()
        pred = (prob >= 0.5).astype("int32")
        add_result(rows, dataset, seed, "06_direct_deep_learning", "engineered_sequence_features", "deep_temporal_learning", "deep_learning_direct", arch, "direct_sigmoid", y_test, pred, X_train_seq.shape[2], "Direct sequence model.")

        extractor = Model(inputs=model.inputs, outputs=model.get_layer("deep_features").output)
        tr_feat = extractor.predict(X_train_seq, verbose=0)
        te_feat = extractor.predict(X_test_seq, verbose=0)
        train_features[arch] = pd.DataFrame(tr_feat, columns=[f"{arch}_deep_feature_{i+1}" for i in range(tr_feat.shape[1])])
        test_features[arch] = pd.DataFrame(te_feat, columns=[f"{arch}_deep_feature_{i+1}" for i in range(te_feat.shape[1])])

    return train_features, test_features


def run_deep_feature_classifiers(rows, dataset, seed, train_features, test_features, y_train, y_test):
    for arch in ["LSTM", "GRU", "TCN", "Transformer"]:
        run_classical(
            rows, dataset, seed,
            train_features[arch], test_features[arch], y_train, y_test,
            "07_individual_deep_features_to_classifiers",
            f"{arch}_deep_features",
            f"{arch}_as_feature_extractor",
            f"Classifiers trained only on {arch} latent temporal features.",
        )
    Xtr_all = pd.concat([train_features[a] for a in ["LSTM", "GRU", "TCN", "Transformer"]], axis=1)
    Xte_all = pd.concat([test_features[a] for a in ["LSTM", "GRU", "TCN", "Transformer"]], axis=1)
    run_classical(
        rows, dataset, seed,
        Xtr_all, Xte_all, y_train, y_test,
        "08_combined_deep_features_to_classifiers",
        "LSTM_GRU_TCN_Transformer_deep_features",
        "combined_deep_feature_extraction",
        "Classifiers trained on concatenated LSTM, GRU, TCN, and Transformer features.",
    )


# =========================================================
# Reporting
# =========================================================
def aggregate_repeated_seeds(summary: pd.DataFrame, out_dir: Path):
    group_cols = ["dataset", "experiment_stage", "feature_set", "optimization_method", "model_family", "model_name", "classifier", "n_features"]
    metric_cols = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "precision_handover", "recall_handover", "f1_handover"]
    agg = summary.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in agg.columns]
    agg.to_csv(out_dir / "repeated_seed_aggregate.csv", index=False)

    # best per dataset/stage based on mean f1_handover
    best = agg.sort_values("f1_handover_mean", ascending=False).groupby(["dataset", "experiment_stage"], as_index=False).head(1)
    best = best.sort_values(["dataset", "experiment_stage"])
    best.to_csv(out_dir / "repeated_seed_best_by_stage_mean.csv", index=False)

    deep = agg[agg["experiment_stage"].isin(["06_direct_deep_learning", "07_individual_deep_features_to_classifiers"])]
    deep.to_csv(out_dir / "repeated_seed_deep_architecture_comparison.csv", index=False)
    return agg, best


def save_dataset_report(cfg: DatasetConfig, raw_rows: int, labeled: pd.DataFrame, y: np.ndarray, groups: np.ndarray, target_info: dict, feature_info: dict, out_dir: Path):
    lines = []
    lines.append(f"Dataset report: {cfg.name}")
    lines.append("=" * 80)
    lines.append(f"Notes: {cfg.notes}")
    lines.append(f"Rows before target creation: {raw_rows}")
    lines.append(f"Rows after target creation: {len(labeled)}")
    lines.append(f"Number of groups/sessions: {labeled['drive_id'].nunique()}")
    for k, v in target_info.items():
        lines.append(f"{k}: {v}")
    lines.append(f"Window size: {cfg.window_size}")
    lines.append(f"Handover horizon: {cfg.handover_horizon}")
    lines.append(f"Split mode: {cfg.split_mode}")
    lines.append("Feature info:")
    for k, v in feature_info.items():
        lines.append(f"  {k}: {v}")
    lines.append("Target distribution after windowing:")
    counts = pd.Series(y).value_counts().sort_index()
    for cls, cnt in counts.items():
        lines.append(f"  {cls}: {cnt}")
    lines.append(f"  positive_ratio: {pd.Series(y).mean():.4f}")
    lines.append("Windows per group/session:")
    for g, cnt in pd.Series(groups).value_counts().sort_index().head(80).items():
        lines.append(f"  group {g}: {cnt}")
    if str(target_info.get("target_source", "")).startswith("official_column"):
        lines.append("Interpretation: this dataset run used a dataset-provided handover/event target column.")
    else:
        lines.append("Interpretation: this dataset run used a future serving-cell-change proxy, not an official handover command label.")
    (out_dir / "dataset_report.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_best_by_stage(best_df: pd.DataFrame, out_dir: Path):
    for dataset, dfg in best_df.groupby("dataset"):
        plt.figure(figsize=(14, 6))
        plt.bar(dfg["experiment_stage"], dfg["f1_handover_mean"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean F1-score for handover/proxy class")
        plt.title(f"{dataset}: Best Mean F1 by Stage across Seeds")
        plt.tight_layout()
        safe_dataset = safe_name(dataset)
        plt.savefig(out_dir / f"{safe_dataset}_best_mean_f1_by_stage.png", dpi=300, bbox_inches="tight")
        plt.close()


# =========================================================
# Main execution per dataset
# =========================================================
def run_one_dataset(cfg: DatasetConfig) -> tuple[pd.DataFrame, dict] | None:
    dataset_out = OUTPUT_ROOT / safe_name(cfg.name)
    dataset_out.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_dataset_csvs(cfg)
    except FileNotFoundError as e:
        print(f"\nSKIPPING {cfg.name}: {e}")
        return None

    labeled, target_info = create_target_and_order(raw, cfg)
    labeled, subset_info = maybe_make_modeling_subset(labeled, cfg, seed=RUN_SEEDS[0])
    target_info["modeling_subset"] = subset_info
    labeled, base_cols, feature_info = prepare_feature_table(labeled, cfg, target_info.get("cell_col"), target_info.get("target_source", ""))
    labeled, engineered_cols = add_temporal_engineered_features(labeled, base_cols)

    X_raw_seq, y, groups, _ = create_windows(labeled, base_cols, cfg.window_size)
    X_eng_seq, y_eng, groups_eng, _ = create_windows(labeled, engineered_cols, cfg.window_size)
    if not np.array_equal(y, y_eng):
        raise RuntimeError(f"{cfg.name}: raw and engineered targets differ.")

    save_dataset_report(cfg, len(raw), labeled, y, groups, target_info, feature_info, dataset_out)

    X_raw_flat = flatten_windows(X_raw_seq, base_cols)
    X_eng_flat = flatten_windows(X_eng_seq, engineered_cols)

    dataset_rows = []

    for seed in RUN_SEEDS:
        print("\n" + "=" * 90)
        print(f"DATASET: {cfg.name} | SEED: {seed}")
        print("=" * 90)
        set_all_seeds(seed)
        seed_out = dataset_out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)

        train_idx, test_idx = split_indices(y, groups, cfg, seed)
        y_train, y_test = y[train_idx], y[test_idx]
        print("Train classes:", pd.Series(y_train).value_counts().to_dict())
        print("Test classes :", pd.Series(y_test).value_counts().to_dict())

        X_raw_train, X_raw_test = X_raw_flat.iloc[train_idx].reset_index(drop=True), X_raw_flat.iloc[test_idx].reset_index(drop=True)
        X_eng_train, X_eng_test = X_eng_flat.iloc[train_idx].reset_index(drop=True), X_eng_flat.iloc[test_idx].reset_index(drop=True)
        X_raw_train_s, X_raw_test_s = scale_flat(X_raw_train, X_raw_test)
        X_eng_train_s, X_eng_test_s = scale_flat(X_eng_train, X_eng_test)

        rows = []
        run_classical(rows, cfg.name, seed, X_raw_train_s, X_raw_test_s, y_train, y_test, "01_raw_flattened_windows", "raw_time_windows", "none_raw_baseline")
        run_classical(rows, cfg.name, seed, X_eng_train_s, X_eng_test_s, y_train, y_test, "02_engineered_flattened_windows", "engineered_time_windows", "temporal_feature_engineering")

        X_fs_train, X_fs_test, selected_info = select_features(X_eng_train_s, y_train, X_eng_test_s)
        selected_info.to_csv(seed_out / "selected_features.csv", index=False)
        run_classical(rows, cfg.name, seed, X_fs_train, X_fs_test, y_train, y_test, "03_feature_selection_on_engineered", "selected_engineered_features", f"feature_selection_{FEATURE_SELECTION_METHOD}_top_{X_fs_train.shape[1]}")

        X_pca_train, X_pca_test = pca_extract(X_eng_train_s, X_eng_test_s)
        run_classical(rows, cfg.name, seed, X_pca_train, X_pca_test, y_train, y_test, "04_pca_on_engineered", "pca_engineered_features", f"pca_{PCA_VARIANCE}_feature_variance")

        X_fs_pca_train, X_fs_pca_test = pca_extract(X_fs_train, X_fs_test)
        run_classical(rows, cfg.name, seed, X_fs_pca_train, X_fs_pca_test, y_train, y_test, "05_feature_selection_plus_pca", "selected_pca_engineered_features", f"feature_selection_{FEATURE_SELECTION_METHOD}_plus_pca")

        X_train_seq, X_test_seq = X_eng_seq[train_idx], X_eng_seq[test_idx]
        X_train_seq_s, X_test_seq_s = scale_seq(X_train_seq, X_test_seq)
        tr_feats, te_feats = train_deep_models(rows, cfg.name, seed, X_train_seq_s, X_test_seq_s, y_train, y_test)
        run_deep_feature_classifiers(rows, cfg.name, seed, tr_feats, te_feats, y_train, y_test)

        seed_df = pd.DataFrame(rows)
        seed_df.to_csv(seed_out / "feature_optimization_summary.csv", index=False)
        dataset_rows.append(seed_df)

        # Free large per-seed matrices before the next seed.
        for _name in [
            "X_raw_train", "X_raw_test", "X_eng_train", "X_eng_test",
            "X_raw_train_s", "X_raw_test_s", "X_eng_train_s", "X_eng_test_s",
            "X_fs_train", "X_fs_test", "X_pca_train", "X_pca_test",
            "X_fs_pca_train", "X_fs_pca_test", "X_train_seq", "X_test_seq",
            "X_train_seq_s", "X_test_seq_s", "tr_feats", "te_feats"
        ]:
            if _name in locals():
                del locals()[_name]
        gc.collect()

    dataset_summary = pd.concat(dataset_rows, ignore_index=True)
    dataset_summary.to_csv(dataset_out / "all_seeds_feature_optimization_summary.csv", index=False)
    agg, best = aggregate_repeated_seeds(dataset_summary, dataset_out)
    plot_best_by_stage(best, dataset_out)

    cfg_json = asdict(cfg)
    cfg_json.update({
        "RUN_SEEDS": RUN_SEEDS,
        "WINDOW_SIZE_GLOBAL_DEFAULT": WINDOW_SIZE,
        "HANDOVER_HORIZON_GLOBAL_DEFAULT": HANDOVER_HORIZON,
        "FEATURE_SELECTION_METHOD": FEATURE_SELECTION_METHOD,
        "FEATURE_SELECTION_K": FEATURE_SELECTION_K,
        "PCA_VARIANCE": PCA_VARIANCE,
        "DEEP_EPOCHS": DEEP_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "PATIENCE": PATIENCE,
        "target_info": target_info,
        "feature_info": feature_info,
    })
    (dataset_out / "run_config.json").write_text(json.dumps(cfg_json, indent=2, default=str), encoding="utf-8")
    return dataset_summary, target_info


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    dataset_reports = []

    for cfg in DATASET_CONFIGS:
        result = run_one_dataset(cfg)
        if result is None:
            dataset_reports.append({"dataset": cfg.name, "status": "skipped_missing_files", "path": cfg.data_path})
            continue
        summary, target_info = result
        all_summaries.append(summary)
        dataset_reports.append({"dataset": cfg.name, "status": "completed", **target_info})

    pd.DataFrame(dataset_reports).to_csv(OUTPUT_ROOT / "cross_dataset_dataset_report.csv", index=False)

    if not all_summaries:
        print("\nNo datasets were executed. Create the dataset folders and put CSV files inside them.")
        return

    full = pd.concat(all_summaries, ignore_index=True)
    full.to_csv(OUTPUT_ROOT / "cross_dataset_all_seeds_summary.csv", index=False)
    agg, best = aggregate_repeated_seeds(full, OUTPUT_ROOT)
    plot_best_by_stage(best, OUTPUT_ROOT)

    # Cross-dataset deep architecture comparison.
    deep = agg[agg["experiment_stage"].isin(["06_direct_deep_learning", "07_individual_deep_features_to_classifiers", "08_combined_deep_features_to_classifiers"])]
    deep.to_csv(OUTPUT_ROOT / "cross_dataset_deep_architecture_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("MULTI-DATASET PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Outputs saved in: {OUTPUT_ROOT.resolve()}")
    print("Main files:")
    print("  - cross_dataset_all_seeds_summary.csv")
    print("  - repeated_seed_aggregate.csv")
    print("  - repeated_seed_best_by_stage_mean.csv")
    print("  - cross_dataset_deep_architecture_comparison.csv")
    print("  - cross_dataset_dataset_report.csv")


if __name__ == "__main__":
    main()
