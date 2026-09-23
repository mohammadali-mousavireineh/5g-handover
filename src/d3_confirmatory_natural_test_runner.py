"""
D3 confirmatory evaluation with an untouched natural-prevalence test set
=======================================================================

Why this script exists
----------------------
The exploratory multi-dataset pipeline was designed to compare many feature
representations on a laptop. For the large O-RAN D3 dataset, it sampled
negative rows before splitting, which changed the class prevalence in the test
set. It also reported the highest test F1 among several classifiers. Both are
reasonable for exploratory benchmarking, but a journal reviewer can correctly
ask for a stricter confirmatory experiment.

This runner fixes those two issues for D3 only:

1. Complete O-RAN sessions are assigned to train, validation, and test BEFORE
   any negative sampling.
2. Negative sampling is applied only to TRAIN windows. Validation and test keep
   their natural event prevalence.
3. The validation set is used for threshold selection and for ranking three
   pre-specified configurations. The test set is evaluated only after the
   validation step.
4. Natural validation/test windows are processed in chunks, so a full
   N x 3000 engineered matrix is never created in RAM.
5. Three confirmatory configurations are evaluated:
      - raw windows + Random Forest
      - temporal engineered windows + Random Forest
      - combined LSTM/GRU/TCN/Transformer features + Random Forest
6. The script saves PR-AUC, ROC-AUC, MCC, Brier score, confusion matrices,
   natural-prevalence predictions, and event-level warning statistics.

The script imports the D3 JSONL loader and target construction from the latest
memory-safe multi-dataset script. Keep both files in the same folder:

    d3_confirmatory_natural_test_runner.py
    multi_dataset_handover_feature_optimization_v4_memory_safe.py
    datasets/
        d3_oran_7_2_handover_events/
            handover_events_1.txt ... handover_events_6.txt
            neigh_measurements_1.txt ... neigh_measurements_6.txt

Full paper run:
    python d3_confirmatory_natural_test_runner.py

One-seed diagnostic run (same model sizes, only one seed):
    python d3_confirmatory_natural_test_runner.py --quick

Resume after interruption:
    Run the same command again. Completed seed folders are skipped.

Force a fresh run:
    python d3_confirmatory_natural_test_runner.py --force

IMPORTANT
---------
Do not replace the old exploratory results until this script completes. The
new outputs are a confirmatory add-on. They are intended to update the D3
results, limitations, and repository documentation in the manuscript.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import multi_dataset_handover_feature_optimization_v4_memory_safe as base
except ModuleNotFoundError as exc:
    if exc.name == "multi_dataset_handover_feature_optimization_v4_memory_safe":
        raise SystemExit(
            "Could not import multi_dataset_handover_feature_optimization_v4_memory_safe.py.\n"
            "Put that latest V4 file in the same folder as this runner."
        ) from exc
    raise SystemExit(
        f"A required Python package is missing while loading the V4 pipeline: {exc.name}.\n"
        "Install the packages from requirements_confirmatory.txt inside the project virtual environment."
    ) from exc


# =============================================================================
# Paper-grade defaults
# =============================================================================
FULL_SEEDS = [1, 7, 21, 42, 100]
QUICK_SEEDS = [42]

WINDOW_SIZE = 10
HANDOVER_HORIZON = 5
TEST_GROUP_FRACTION = 0.20
VALIDATION_GROUP_FRACTION = 0.15

# Sample only TRAIN windows. All positive train windows are kept where possible.
TRAIN_NEGATIVE_TO_POSITIVE_RATIO = 4
TRAIN_MAX_WINDOWS = 30000

# Chunk sizes keep natural validation/test inference memory-safe.
RAW_CHUNK_SIZE = 5000
ENGINEERED_CHUNK_SIZE = 750
DEEP_CHUNK_SIZE = 512

RF_TREES = 250
RF_N_JOBS = 2
RF_RANDOM_MAX_DEPTH = None

DEEP_EPOCHS = 25
DEEP_BATCH_SIZE = 32
DEEP_PATIENCE = 5
DEEP_LEARNING_RATE = 0.001

OUTPUT_ROOT_NAME = "d3_confirmatory_natural_test_outputs"

# Three fixed confirmatory configurations. No classifier sweep is performed.
MODEL_RAW_RF = "raw_windows_random_forest"
MODEL_TEMPORAL_RF = "temporal_engineering_random_forest"
MODEL_COMBINED_RF = "combined_deep_features_random_forest"
MODEL_ORDER = [MODEL_RAW_RF, MODEL_TEMPORAL_RF, MODEL_COMBINED_RF]


@dataclass
class GroupArrays:
    group_id: int
    x: np.ndarray                 # [rows, base_features], float32
    target: np.ndarray            # row-level future-event target
    event_marker: np.ndarray      # aligned event indication at the row
    sample_order: np.ndarray
    source_file: str


# =============================================================================
# Reproducibility and logging helpers
# =============================================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    base.tf.random.set_seed(seed)


def configure_tensorflow_memory() -> None:
    """Avoid TensorFlow reserving all GPU memory when a GPU is available."""
    try:
        for gpu in base.tf.config.list_physical_devices("GPU"):
            base.tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


# =============================================================================
# D3 loading and train-only preprocessing
# =============================================================================
def d3_config() -> base.DatasetConfig:
    return base.DatasetConfig(
        name="D3_ORAN_7_2_event_log_handover_labels_confirmatory",
        data_path="datasets/d3_oran_7_2_handover_events",
        csv_glob="*.txt",
        official_target_col="handover_event_marker",
        official_target_mode="future_horizon",
        cell_id_col=None,
        session_col="oran_session_id",
        time_col="oran_sample_order",
        split_mode="group_by_session",
        window_size=WINDOW_SIZE,
        handover_horizon=HANDOVER_HORIZON,
        neg_pos_ratio=None,
        max_modeling_rows=None,
        notes=(
            "Confirmatory D3 run. Group split is performed before any sampling; "
            "negative sampling is restricted to training windows; validation and "
            "test retain natural prevalence."
        ),
    )


def detect_numeric_feature_columns(data: pd.DataFrame, target_info: dict) -> tuple[list[str], dict]:
    """Detect D3 numeric radio/context features without imputing from test data."""
    exclude = {
        "handover_target",
        "handover_event_marker",
        "official_event_binary",
        "drive_id",
        "sample_order",
        "source_row_order",
        "source_file_id",
        "source_file",
        "UE",
        "oran_ue_id",
        "oran_pair_id",
        "oran_session_id",
        "topic",
        "CELL",
        "DU",
        "ENB",
    }
    cell_col = target_info.get("cell_col")
    if cell_col:
        exclude.add(cell_col)

    id_like = ["id", "ue", "session", "route", "trip", "log", "name", "imsi", "imei", "mcc", "mnc", "tac", "topic"]
    radio_exceptions = ["pci", "rsrp", "rsrq", "sinr", "snr", "rssi", "cqi"]

    numeric_cols: list[str] = []
    rejected: dict[str, str] = {}
    for col in data.columns:
        if col in exclude:
            continue
        cc = base.canonical(col)
        if any(k in cc for k in id_like) and not any(k in cc for k in radio_exceptions):
            rejected[col] = "identifier-like"
            continue
        numeric = base.coerce_numeric_series(data[col])
        ratio = float(numeric.notna().mean())
        if ratio >= 0.70 and numeric.nunique(dropna=True) > 1:
            numeric_cols.append(col)
        else:
            rejected[col] = f"numeric_ratio={ratio:.3f} or constant"

    if not numeric_cols:
        raise RuntimeError("No numeric D3 predictor columns were detected.")

    return numeric_cols, {
        "numeric_feature_count": len(numeric_cols),
        "numeric_features": numeric_cols,
        "rejected_columns": rejected,
        "imputation": "median fitted on training-session rows only",
        "categoricals": "not used in this confirmatory D3 run",
        "current_cell_identifier_included": False,
    }


def build_numeric_matrix_with_nan(data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    arrays = []
    for col in feature_cols:
        arr = base.coerce_numeric_series(data[col]).to_numpy(dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan
        arrays.append(arr)
    return np.column_stack(arrays).astype(np.float32, copy=False)


def fit_train_medians(x_nan: np.ndarray, train_row_mask: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x_nan[train_row_mask], axis=0).astype(np.float32)
    med[~np.isfinite(med)] = 0.0
    return med


def impute_matrix(x_nan: np.ndarray, medians: np.ndarray) -> np.ndarray:
    x = np.array(x_nan, dtype=np.float32, copy=True)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(medians, np.where(bad)[1])
    return x


# =============================================================================
# Session split and endpoint tables
# =============================================================================
def make_endpoint_table(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, tuple[int, int]]]:
    """Create one row per valid sequence-window endpoint without materializing windows."""
    refs = []
    group_slices: dict[int, tuple[int, int]] = {}

    start = 0
    for gid, g in data.groupby("drive_id", sort=False):
        n = len(g)
        end = start + n
        group_slices[int(gid)] = (start, end)
        if n >= WINDOW_SIZE:
            local_pos = np.arange(WINDOW_SIZE - 1, n, dtype=np.int32)
            global_pos = start + local_pos
            refs.append(pd.DataFrame({
                "group_id": int(gid),
                "local_pos": local_pos,
                "global_endpoint_row": global_pos,
                "y": g["handover_target"].to_numpy(dtype=np.int8)[local_pos],
                "sample_order": g["sample_order"].to_numpy()[local_pos],
                "source_file": g["source_file"].astype(str).iloc[0],
            }))
        start = end

    if not refs:
        raise RuntimeError("No valid D3 windows were found.")
    table = pd.concat(refs, ignore_index=True)
    return table, group_slices


def group_stats_from_endpoints(endpoint_table: pd.DataFrame) -> pd.DataFrame:
    out = endpoint_table.groupby("group_id", as_index=False).agg(
        windows=("y", "size"),
        positives=("y", "sum"),
    )
    out["negatives"] = out["windows"] - out["positives"]
    return out


def split_groups_balanced(group_stats: pd.DataFrame, seed: int, attempts: int = 2000) -> dict[str, list[int]]:
    """Choose disjoint session groups while keeping all splits two-class and near target sizes."""
    gids = group_stats["group_id"].to_numpy(dtype=int)
    n_groups = len(gids)
    n_test = max(1, int(round(n_groups * TEST_GROUP_FRACTION)))
    n_val = max(1, int(round(n_groups * VALIDATION_GROUP_FRACTION)))
    if n_test + n_val >= n_groups:
        raise RuntimeError("Not enough D3 groups for train/validation/test splitting.")

    stat = group_stats.set_index("group_id")
    total_windows = float(stat["windows"].sum())
    overall_prev = float(stat["positives"].sum() / max(total_windows, 1.0))

    best = None
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        perm = rng.permutation(gids)
        test = perm[:n_test]
        val = perm[n_test:n_test + n_val]
        train = perm[n_test + n_val:]

        candidate = {"train": train, "validation": val, "test": test}
        valid = True
        score = 0.0
        target_fracs = {
            "train": 1.0 - TEST_GROUP_FRACTION - VALIDATION_GROUP_FRACTION,
            "validation": VALIDATION_GROUP_FRACTION,
            "test": TEST_GROUP_FRACTION,
        }
        for name, arr in candidate.items():
            s = stat.loc[arr]
            windows = float(s["windows"].sum())
            positives = float(s["positives"].sum())
            negatives = float(s["negatives"].sum())
            if positives <= 0 or negatives <= 0:
                valid = False
                break
            frac = windows / total_windows
            prev = positives / windows
            score += abs(frac - target_fracs[name])
            score += 0.25 * abs(prev - overall_prev) / max(overall_prev, 1e-6)
        if valid and (best is None or score < best[0]):
            best = (score, candidate)

    if best is None:
        raise RuntimeError("Could not find a group-isolated split containing both classes in all partitions.")

    return {k: sorted(map(int, v.tolist())) for k, v in best[1].items()}


def summarize_split(endpoint_table: pd.DataFrame, split_groups: dict[str, list[int]]) -> dict:
    summary = {}
    for name, gids in split_groups.items():
        d = endpoint_table[endpoint_table["group_id"].isin(gids)]
        summary[name] = {
            "groups": len(gids),
            "windows": int(len(d)),
            "positives": int(d["y"].sum()),
            "negatives": int((d["y"] == 0).sum()),
            "positive_ratio": float(d["y"].mean()),
        }
    return summary


def sample_training_windows(train_refs: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    pos = train_refs[train_refs["y"] == 1]
    neg = train_refs[train_refs["y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Training partition must contain both classes.")

    if len(pos) >= TRAIN_MAX_WINDOWS:
        pos_keep = pos.sample(n=TRAIN_MAX_WINDOWS // 2, random_state=seed)
    else:
        pos_keep = pos

    neg_n = min(
        len(neg),
        len(pos_keep) * TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
        TRAIN_MAX_WINDOWS - len(pos_keep),
    )
    if neg_n <= 0:
        raise RuntimeError("TRAIN_MAX_WINDOWS is too small for the positive training windows.")
    neg_keep = neg.sample(n=int(neg_n), random_state=seed)
    sampled = pd.concat([pos_keep, neg_keep], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled, {
        "sampling_applied_to": "training windows only",
        "all_positive_train_windows_before_sampling": int(len(pos)),
        "all_negative_train_windows_before_sampling": int(len(neg)),
        "positive_windows_kept": int((sampled["y"] == 1).sum()),
        "negative_windows_kept": int((sampled["y"] == 0).sum()),
        "sampled_train_windows": int(len(sampled)),
        "sampled_train_positive_ratio": float(sampled["y"].mean()),
        "negative_to_positive_cap": TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
        "max_train_windows": TRAIN_MAX_WINDOWS,
    }


# =============================================================================
# Window extraction without a full engineered table
# =============================================================================
def build_group_arrays(
    data: pd.DataFrame,
    x_imputed: np.ndarray,
    group_slices: dict[int, tuple[int, int]],
) -> dict[int, GroupArrays]:
    groups: dict[int, GroupArrays] = {}
    for gid, (start, end) in group_slices.items():
        g = data.iloc[start:end]
        groups[gid] = GroupArrays(
            group_id=gid,
            x=x_imputed[start:end],
            target=g["handover_target"].to_numpy(dtype=np.int8),
            event_marker=g["official_event_binary"].to_numpy(dtype=np.int8),
            sample_order=g["sample_order"].to_numpy(),
            source_file=str(g["source_file"].iloc[0]),
        )
    return groups


def raw_windows_for_positions(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    offsets = np.arange(-(WINDOW_SIZE - 1), 1, dtype=np.int32)
    idx = positions[:, None] + offsets[None, :]
    return x[idx].astype(np.float32, copy=False)


def _rolling_stats_for_rows(x: np.ndarray, q: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling stats for every [batch, window] row using only prior rows in the same group."""
    offsets = np.arange(win - 1, -1, -1, dtype=np.int32)
    idx = q[:, :, None] - offsets[None, None, :]
    valid = idx >= 0
    clipped = np.clip(idx, 0, len(x) - 1)
    vals = x[clipped]  # [b, W, win, F]
    mask = valid[..., None]
    count = mask.sum(axis=2).astype(np.float32)
    count_safe = np.maximum(count, 1.0)
    summed = np.where(mask, vals, 0.0).sum(axis=2)
    mean = summed / count_safe

    centered = np.where(mask, vals - mean[:, :, None, :], 0.0)
    sq = (centered * centered).sum(axis=2)
    denom = np.maximum(count - 1.0, 1.0)
    std = np.sqrt(sq / denom)
    std[count[..., 0] < 2] = 0.0

    min_vals = np.where(mask, vals, np.inf).min(axis=2)
    max_vals = np.where(mask, vals, -np.inf).max(axis=2)
    min_vals[~np.isfinite(min_vals)] = 0.0
    max_vals[~np.isfinite(max_vals)] = 0.0
    return mean.astype(np.float32), std.astype(np.float32), min_vals.astype(np.float32), max_vals.astype(np.float32)


def engineered_windows_for_positions(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Match the V4 temporal feature family without creating a full N x 300 table."""
    offsets = np.arange(-(WINDOW_SIZE - 1), 1, dtype=np.int32)
    q = positions[:, None] + offsets[None, :]
    base_win = x[q].astype(np.float32, copy=False)
    b, w, f = base_win.shape

    diffs = []
    for lag in (1, 3, 5):
        prev_idx = q - lag
        valid = prev_idx >= 0
        prev = x[np.clip(prev_idx, 0, len(x) - 1)]
        d = base_win - prev
        d[~valid[..., None].repeat(f, axis=2)] = 0.0
        diffs.append(d.astype(np.float32))

    r3 = _rolling_stats_for_rows(x, q, 3)
    r5 = _rolling_stats_for_rows(x, q, 5)

    out = np.empty((b, w, f + f * 11), dtype=np.float32)
    out[:, :, :f] = base_win
    cursor = f
    for j in range(f):
        derived_j = np.stack([
            diffs[0][:, :, j],
            diffs[1][:, :, j],
            diffs[2][:, :, j],
            r3[0][:, :, j],
            r3[1][:, :, j],
            r3[2][:, :, j],
            r3[3][:, :, j],
            r5[0][:, :, j],
            r5[1][:, :, j],
            r5[2][:, :, j],
            r5[3][:, :, j],
        ], axis=2)
        out[:, :, cursor:cursor + 11] = derived_j
        cursor += 11
    return out


def engineered_feature_names(base_cols: list[str]) -> list[str]:
    names = list(base_cols)
    for col in base_cols:
        names.extend([
            f"{col}_diff1", f"{col}_diff3", f"{col}_diff5",
            f"{col}_roll3_mean", f"{col}_roll3_std", f"{col}_roll3_min", f"{col}_roll3_max",
            f"{col}_roll5_mean", f"{col}_roll5_std", f"{col}_roll5_min", f"{col}_roll5_max",
        ])
    return names


def iter_window_chunks(
    refs: pd.DataFrame,
    groups: dict[int, GroupArrays],
    representation: str,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (row_indices_in_refs, sequence_array) while preserving refs order."""
    for gid, sub in refs.groupby("group_id", sort=False):
        all_indices = sub.index.to_numpy(dtype=np.int64)
        all_positions = sub["local_pos"].to_numpy(dtype=np.int32)
        g = groups[int(gid)]
        for start in range(0, len(sub), chunk_size):
            idx = all_indices[start:start + chunk_size]
            pos = all_positions[start:start + chunk_size]
            if representation == "raw":
                seq = raw_windows_for_positions(g.x, pos)
            elif representation == "engineered":
                seq = engineered_windows_for_positions(g.x, pos)
            else:
                raise ValueError(f"Unknown representation: {representation}")
            yield idx, seq


def materialize_windows(
    refs: pd.DataFrame,
    groups: dict[int, GroupArrays],
    representation: str,
    chunk_size: int,
) -> np.ndarray:
    if representation == "raw":
        f = next(iter(groups.values())).x.shape[1]
        shape = (len(refs), WINDOW_SIZE, f)
    else:
        f = next(iter(groups.values())).x.shape[1] * 12
        shape = (len(refs), WINDOW_SIZE, f)
    out = np.empty(shape, dtype=np.float32)
    refs = refs.reset_index(drop=True)
    for idx, seq in iter_window_chunks(refs, groups, representation, chunk_size):
        out[idx] = seq
    return out


# =============================================================================
# Model fitting, chunked prediction, and threshold selection
# =============================================================================
def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=seed,
        n_jobs=RF_N_JOBS,
        class_weight="balanced",
        max_depth=RF_RANDOM_MAX_DEPTH,
    )


def predict_rf_chunks(
    model: RandomForestClassifier,
    refs: pd.DataFrame,
    groups: dict[int, GroupArrays],
    representation: str,
    chunk_size: int,
) -> np.ndarray:
    refs = refs.reset_index(drop=True)
    prob = np.empty(len(refs), dtype=np.float32)
    for idx, seq in iter_window_chunks(refs, groups, representation, chunk_size):
        flat = seq.reshape(len(seq), -1)
        prob[idx] = model.predict_proba(flat)[:, 1].astype(np.float32)
        del seq, flat
    return prob


def fit_sequence_scaler(x_train_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train_seq.reshape(-1, x_train_seq.shape[2])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def scale_sequence_inplace(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x -= mean[None, None, :]
    x /= std[None, None, :]
    return x


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    counts = pd.Series(y).value_counts().to_dict()
    total = len(y)
    return {
        0: total / (2.0 * max(counts.get(0, 1), 1)),
        1: total / (2.0 * max(counts.get(1, 1), 1)),
    }


def build_deep_model(arch: str, input_shape: tuple[int, int]):
    builders = {
        "LSTM": base.build_lstm,
        "GRU": base.build_gru,
        "TCN": base.build_tcn,
        "Transformer": base.build_transformer,
    }
    model = builders[arch](input_shape)
    model.compile(
        optimizer=base.Adam(learning_rate=DEEP_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_extract_combined_features(
    seed: int,
    seed_dir: Path,
    x_train_eng: np.ndarray,
    y_train: np.ndarray,
    val_refs: pd.DataFrame,
    test_refs: pd.DataFrame,
    groups: dict[int, GroupArrays],
) -> tuple[np.ndarray, np.memmap, np.memmap, dict]:
    """Train four encoders and write natural val/test latent features to disk-backed arrays."""
    mean, std = fit_sequence_scaler(x_train_eng)
    scale_sequence_inplace(x_train_eng, mean, std)
    np.save(seed_dir / "deep_sequence_mean.npy", mean)
    np.save(seed_dir / "deep_sequence_std.npy", std)

    n_train = len(x_train_eng)
    train_combined = np.empty((n_train, 128), dtype=np.float32)
    val_path = seed_dir / "validation_combined_deep_features.float32.mmap"
    test_path = seed_dir / "test_combined_deep_features.float32.mmap"
    val_combined = np.memmap(val_path, mode="w+", dtype="float32", shape=(len(val_refs), 128))
    test_combined = np.memmap(test_path, mode="w+", dtype="float32", shape=(len(test_refs), 128))

    fit_idx, es_idx = train_test_split(
        np.arange(n_train),
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )

    histories = {}
    for arch_i, arch in enumerate(["LSTM", "GRU", "TCN", "Transformer"]):
        print(f"\nTraining confirmatory {arch} encoder...")
        set_all_seeds(seed)
        model = build_deep_model(arch, (x_train_eng.shape[1], x_train_eng.shape[2]))
        es = base.EarlyStopping(
            monitor="val_loss",
            patience=DEEP_PATIENCE,
            restore_best_weights=True,
        )
        history = model.fit(
            x_train_eng[fit_idx],
            y_train[fit_idx],
            validation_data=(x_train_eng[es_idx], y_train[es_idx]),
            epochs=DEEP_EPOCHS,
            batch_size=DEEP_BATCH_SIZE,
            callbacks=[es],
            verbose=0,
            shuffle=True,
            class_weight=class_weight_dict(y_train[fit_idx]),
        )
        histories[arch] = {
            "epochs_run": len(history.history.get("loss", [])),
            "best_val_loss": safe_float(np.min(history.history.get("val_loss", [np.nan]))),
        }

        extractor = base.Model(inputs=model.inputs, outputs=model.get_layer("deep_features").output)
        col_slice = slice(arch_i * 32, (arch_i + 1) * 32)
        train_combined[:, col_slice] = extractor.predict(
            x_train_eng,
            batch_size=max(64, DEEP_BATCH_SIZE),
            verbose=0,
        ).astype(np.float32)

        for refs, target_mm, label in [
            (val_refs.reset_index(drop=True), val_combined, "validation"),
            (test_refs.reset_index(drop=True), test_combined, "test"),
        ]:
            for idx, seq in iter_window_chunks(refs, groups, "engineered", DEEP_CHUNK_SIZE):
                scale_sequence_inplace(seq, mean, std)
                feat = extractor.predict(seq, batch_size=max(64, DEEP_BATCH_SIZE), verbose=0)
                target_mm[idx, col_slice] = feat.astype(np.float32)
                del seq, feat
            target_mm.flush()
            print(f"  {arch}: wrote {label} latent features")

        model.save(seed_dir / f"{arch.lower()}_confirmatory_encoder.keras")
        del extractor, model, history
        base.tf.keras.backend.clear_session()
        gc.collect()

    json_dump(seed_dir / "deep_training_history_summary.json", histories)
    return train_combined, val_combined, test_combined, histories


def threshold_from_validation(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, pd.DataFrame({"threshold": [0.5], "precision": [0.0], "recall": [0.0], "f1": [0.0]})
    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    best_idx = int(np.nanargmax(f1))
    curve = pd.DataFrame({
        "threshold": thresholds,
        "precision": p,
        "recall": r,
        "f1": f1,
    })
    return float(thresholds[best_idx]), curve


def probability_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "positive_ratio": float(np.mean(y_true)),
        "predicted_positive_ratio": float(np.mean(pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positives_per_1000_negative_windows": float(1000.0 * fp / max(tn + fp, 1)),
    }


def event_level_metrics(
    refs: pd.DataFrame,
    groups: dict[int, GroupArrays],
    pred: np.ndarray,
) -> dict[str, Any]:
    refs = refs.reset_index(drop=True).copy()
    refs["pred"] = pred.astype(np.int8)
    total_events = 0
    evaluable_events = 0
    detected = 0
    lead_samples = []
    alert_counts = []

    for gid, sub in refs.groupby("group_id", sort=False):
        g = groups[int(gid)]
        event_positions = np.where(g.event_marker == 1)[0]
        total_events += int(len(event_positions))
        pred_by_pos = dict(zip(sub["local_pos"].astype(int), sub["pred"].astype(int)))
        for event_pos in event_positions:
            candidates = list(range(max(WINDOW_SIZE - 1, event_pos - HANDOVER_HORIZON), event_pos))
            available = [p for p in candidates if p in pred_by_pos]
            if not available:
                continue
            evaluable_events += 1
            alerts = [p for p in available if pred_by_pos[p] == 1]
            if alerts:
                detected += 1
                first_alert = min(alerts)
                lead_samples.append(int(event_pos - first_alert))
                alert_counts.append(len(alerts))

    return {
        "raw_event_markers_in_test_groups": int(total_events),
        "evaluable_events_with_prior_window": int(evaluable_events),
        "detected_events": int(detected),
        "missed_events": int(evaluable_events - detected),
        "event_level_recall": float(detected / max(evaluable_events, 1)),
        "median_first_alert_lead_samples": safe_float(np.median(lead_samples)) if lead_samples else None,
        "mean_first_alert_lead_samples": safe_float(np.mean(lead_samples)) if lead_samples else None,
        "mean_alert_windows_per_detected_event": safe_float(np.mean(alert_counts)) if alert_counts else None,
        "lead_time_unit": "samples; datasets do not provide one common sampling interval",
    }


def save_predictions(path: Path, refs: pd.DataFrame, prob: np.ndarray, threshold: float) -> None:
    out = refs.reset_index(drop=True).copy()
    out["probability"] = prob
    out["threshold"] = threshold
    out["prediction"] = (prob >= threshold).astype(np.int8)
    out.to_csv(path, index=False, compression="gzip")


# =============================================================================
# One seed
# =============================================================================
def run_seed(
    seed: int,
    root: Path,
    data: pd.DataFrame,
    x_nan: np.ndarray,
    feature_cols: list[str],
    endpoint_table: pd.DataFrame,
    group_slices: dict[int, tuple[int, int]],
    group_stats: pd.DataFrame,
    force: bool,
) -> pd.DataFrame:
    seed_dir = root / f"seed_{seed}"
    done_file = seed_dir / "SEED_COMPLETE.json"
    if done_file.exists() and not force:
        print(f"\nSeed {seed} already completed; loading saved metrics.")
        return pd.read_csv(seed_dir / "test_metrics.csv")

    seed_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    start_time = time.time()

    split_groups = split_groups_balanced(group_stats, seed)
    split_summary = summarize_split(endpoint_table, split_groups)
    json_dump(seed_dir / "group_split.json", split_groups)
    json_dump(seed_dir / "natural_split_summary.json", split_summary)

    train_refs_full = endpoint_table[endpoint_table["group_id"].isin(split_groups["train"])].reset_index(drop=True)
    val_refs = endpoint_table[endpoint_table["group_id"].isin(split_groups["validation"])].reset_index(drop=True)
    test_refs = endpoint_table[endpoint_table["group_id"].isin(split_groups["test"])].reset_index(drop=True)
    train_refs, sampling_info = sample_training_windows(train_refs_full, seed)
    json_dump(seed_dir / "training_sampling.json", sampling_info)

    train_row_mask = data["drive_id"].isin(split_groups["train"]).to_numpy()
    medians = fit_train_medians(x_nan, train_row_mask)
    np.save(seed_dir / "train_fitted_medians.npy", medians)
    x = impute_matrix(x_nan, medians)
    groups = build_group_arrays(data, x, group_slices)

    y_train = train_refs["y"].to_numpy(dtype=np.int8)
    y_val = val_refs["y"].to_numpy(dtype=np.int8)
    y_test = test_refs["y"].to_numpy(dtype=np.int8)

    print("\n" + "=" * 96)
    print(f"CONFIRMATORY D3 | SEED {seed}")
    print("=" * 96)
    print("Natural split:", json.dumps(split_summary, indent=2))
    print("Sampled train:", sampling_info)

    model_prob_val: dict[str, np.ndarray] = {}
    model_prob_test: dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # 1) Raw windows + RF
    # -------------------------------------------------------------------------
    print("\n[1/3] Raw windows + Random Forest")
    x_raw_train = materialize_windows(train_refs, groups, "raw", RAW_CHUNK_SIZE)
    rf_raw = make_rf(seed)
    rf_raw.fit(x_raw_train.reshape(len(x_raw_train), -1), y_train)
    model_prob_val[MODEL_RAW_RF] = predict_rf_chunks(rf_raw, val_refs, groups, "raw", RAW_CHUNK_SIZE)
    model_prob_test[MODEL_RAW_RF] = predict_rf_chunks(rf_raw, test_refs, groups, "raw", RAW_CHUNK_SIZE)
    import joblib
    joblib.dump(rf_raw, seed_dir / "raw_windows_random_forest.joblib", compress=3)
    del x_raw_train, rf_raw
    gc.collect()

    # -------------------------------------------------------------------------
    # Materialize sampled engineered training windows once.
    # Natural validation/test remain chunked.
    # -------------------------------------------------------------------------
    print("\nBuilding sampled engineered training windows (test remains untouched/chunked)...")
    x_eng_train = materialize_windows(train_refs, groups, "engineered", ENGINEERED_CHUNK_SIZE)

    # -------------------------------------------------------------------------
    # 2) Temporal engineering + RF
    # -------------------------------------------------------------------------
    print("\n[2/3] Temporal engineering + Random Forest")
    rf_temporal = make_rf(seed)
    rf_temporal.fit(x_eng_train.reshape(len(x_eng_train), -1), y_train)
    model_prob_val[MODEL_TEMPORAL_RF] = predict_rf_chunks(
        rf_temporal, val_refs, groups, "engineered", ENGINEERED_CHUNK_SIZE
    )
    model_prob_test[MODEL_TEMPORAL_RF] = predict_rf_chunks(
        rf_temporal, test_refs, groups, "engineered", ENGINEERED_CHUNK_SIZE
    )
    joblib.dump(rf_temporal, seed_dir / "temporal_engineering_random_forest.joblib", compress=3)
    del rf_temporal
    gc.collect()

    # -------------------------------------------------------------------------
    # 3) Combined deep features + RF
    # -------------------------------------------------------------------------
    print("\n[3/3] Combined LSTM/GRU/TCN/Transformer features + Random Forest")
    train_combined, val_combined, test_combined, _ = train_and_extract_combined_features(
        seed, seed_dir, x_eng_train, y_train, val_refs, test_refs, groups
    )
    del x_eng_train
    gc.collect()

    rf_combined = make_rf(seed)
    rf_combined.fit(train_combined, y_train)
    model_prob_val[MODEL_COMBINED_RF] = rf_combined.predict_proba(val_combined)[:, 1].astype(np.float32)
    model_prob_test[MODEL_COMBINED_RF] = np.empty(len(test_refs), dtype=np.float32)
    for start in range(0, len(test_refs), 10000):
        stop = min(start + 10000, len(test_refs))
        model_prob_test[MODEL_COMBINED_RF][start:stop] = rf_combined.predict_proba(test_combined[start:stop])[:, 1]
    joblib.dump(rf_combined, seed_dir / "combined_deep_features_random_forest.joblib", compress=3)
    del rf_combined, train_combined, val_combined, test_combined
    gc.collect()

    # -------------------------------------------------------------------------
    # Validation thresholds, model ranking, and untouched test evaluation
    # -------------------------------------------------------------------------
    validation_rows = []
    test_rows = []
    thresholds = {}

    for model_name in MODEL_ORDER:
        threshold, curve = threshold_from_validation(y_val, model_prob_val[model_name])
        thresholds[model_name] = threshold
        curve.to_csv(seed_dir / f"{model_name}_validation_threshold_curve.csv", index=False)

        val_m = probability_metrics(y_val, model_prob_val[model_name], threshold)
        val_m.update({"seed": seed, "model": model_name, "partition": "validation_natural"})
        validation_rows.append(val_m)

    validation_df = pd.DataFrame(validation_rows).sort_values("f1", ascending=False)
    selected_model = str(validation_df.iloc[0]["model"])
    json_dump(seed_dir / "selected_model_from_validation.json", {
        "selected_model": selected_model,
        "selection_metric": "validation natural-prevalence F1",
        "selected_threshold": thresholds[selected_model],
        "candidate_models": MODEL_ORDER,
        "test_was_not_used_for_selection": True,
    })

    for model_name in MODEL_ORDER:
        threshold = thresholds[model_name]
        prob = model_prob_test[model_name]
        test_m = probability_metrics(y_test, prob, threshold)
        pred = (prob >= threshold).astype(np.int8)
        event_m = event_level_metrics(test_refs, groups, pred)
        test_m.update(event_m)
        test_m.update({
            "seed": seed,
            "model": model_name,
            "partition": "test_natural_untouched",
            "selected_by_validation": bool(model_name == selected_model),
        })
        test_rows.append(test_m)
        save_predictions(
            seed_dir / f"{model_name}_natural_test_predictions.csv.gz",
            test_refs,
            prob,
            threshold,
        )

    validation_df.to_csv(seed_dir / "validation_metrics.csv", index=False)
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(seed_dir / "test_metrics.csv", index=False)
    json_dump(seed_dir / "thresholds_selected_on_validation.json", thresholds)

    # PR curve on natural test distribution.
    plt.figure(figsize=(7, 5))
    for model_name in MODEL_ORDER:
        p, r, _ = precision_recall_curve(y_test, model_prob_test[model_name])