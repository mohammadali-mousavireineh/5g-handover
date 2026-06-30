from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .config import DatasetSpec
from .data import dataset_summary, load_feature_base
from .models import classifier_search_spaces


def estimator_scores(estimator, X_test: np.ndarray) -> np.ndarray | None:
    """Return probability/decision scores for ROC-AUC when available."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_test))
    return None


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return metrics


def run_grid_experiment(
    data_dir: Path,
    datasets: Iterable[DatasetSpec],
    scoring: str = "f1",
    cv_folds: int = 3,
    test_size: float = 0.25,
    random_state: int = 42,
    n_jobs: int = 1,
    use_smote: bool = False,
    full_grid: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full LSTM/GRU/TCN comparison."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    spaces = classifier_search_spaces(use_smote=use_smote, full_grid=full_grid, random_state=random_state)
    rows: list[dict] = []

    for spec in datasets:
        path = data_dir / spec.filename
        X, y, _ = load_feature_base(path)
        summary = dataset_summary(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        for clf_name, (pipeline, param_grid) in spaces.items():
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            y_pred = best.predict(X_test)
            y_score = estimator_scores(best, X_test)
            metrics = classification_metrics(y_test, y_pred, y_score)
            cm = confusion_matrix(y_test, y_pred).tolist()

            rows.append({
                "feature_model": spec.feature_model,
                "source_csv": spec.filename,
                "classifier": clf_name,
                "cv_scoring": scoring,
                "cv_best_score": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_roc_auc": metrics["roc_auc"],
                "confusion_matrix": json.dumps(cm),
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "total_samples": summary["samples"],
                "positive_ratio": summary["positive_ratio"],
                "use_smote": use_smote,
                "full_grid": full_grid,
                "random_state": random_state,
            })

    results = pd.DataFrame(rows).sort_values(["test_f1", "test_accuracy"], ascending=False).reset_index(drop=True)
    best = (
        results.sort_values(["feature_model", "test_f1", "test_accuracy"], ascending=[True, False, False])
        .groupby("feature_model", as_index=False)
        .head(1)
        .sort_values(["test_f1", "test_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    return results, best
