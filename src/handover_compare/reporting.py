from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_metric_chart(best_df: pd.DataFrame, output_path: Path | None = None) -> str | None:
    plot_df = best_df.sort_values("test_f1", ascending=False).copy()
    labels = plot_df["feature_model"] + " + " + plot_df["classifier"]
    x = np.arange(len(plot_df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width, plot_df["test_accuracy"], width, label="Accuracy")
    ax.bar(x, plot_df["test_balanced_accuracy"], width, label="Balanced accuracy")
    ax.bar(x + width, plot_df["test_f1"], width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Best classifier for each RSRP predictor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    if output_path:
        _save_fig(fig, output_path)
        return None
    return _fig_to_base64(fig)


def save_confusion_matrices(best_df: pd.DataFrame, output_path: Path | None = None) -> str | None:
    rows = best_df.to_dict("records")
    fig, axes = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 3.8))
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        cm_raw = row["confusion_matrix"]
        cm = np.asarray(json.loads(cm_raw) if isinstance(cm_raw, str) else cm_raw)
        im = ax.imshow(cm)
        ax.set_title(f"{row['feature_model']} + {row['classifier']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrices for best models", y=1.04)
    if output_path:
        _save_fig(fig, output_path)
        return None
    return _fig_to_base64(fig)


def save_dataset_distribution(y_counts: dict[str, dict], output_path: Path) -> None:
    names = list(y_counts)
    negatives = [y_counts[n]["negatives"] for n in names]
    positives = [y_counts[n]["positives"] for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x, negatives, label="No handover")
    ax.bar(x, positives, bottom=negatives, label="Handover")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Samples")
    ax.set_title("Class distribution of processed ANATEL feature bases")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, output_path)


def _html_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return view.to_html(index=False, escape=False, classes="result-table")


def build_html_report(results_df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path) -> None:
    metric_chart = save_metric_chart(best_df)
    confusion_chart = save_confusion_matrices(best_df)
    winner = results_df.iloc[0]

    columns = [
        "feature_model", "classifier", "test_accuracy", "test_balanced_accuracy",
        "test_precision", "test_recall", "test_f1", "test_roc_auc", "cv_best_score", "best_params",
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ANATEL Handover Prediction Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 34px; background: #f7f7f7; color: #222; }}
.card {{ background: white; border-radius: 14px; padding: 22px; margin-bottom: 22px; box-shadow: 0 2px 14px rgba(0,0,0,.08); }}
h1, h2 {{ margin-top: 0; }}
.badge {{ display: inline-block; padding: 5px 11px; border-radius: 999px; background: #eee; margin: 4px; }}
table.result-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
table.result-table th {{ background: #222; color: white; text-align: left; padding: 9px; }}
table.result-table td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
table.result-table tr:nth-child(even) {{ background: #f4f4f4; }}
img {{ max-width: 100%; }}
.note {{ color: #555; line-height: 1.55; }}
</style>
</head>
<body>
<div class="card">
<h1>ANATEL Handover Prediction - Unified Grid Search</h1>
<p class="note">This report compares handover classifiers trained on predicted RSRP windows generated by LSTM, GRU, and TCN feature models. The held-out test set is kept untouched; optional SMOTE is applied only inside the training/CV pipeline.</p>
</div>
<div class="card">
<h2>Overall best result</h2>
<span class="badge">Feature model: <b>{winner['feature_model']}</b></span>
<span class="badge">Classifier: <b>{winner['classifier']}</b></span>
<span class="badge">F1: <b>{winner['test_f1']:.4f}</b></span>
<span class="badge">Accuracy: <b>{winner['test_accuracy']:.4f}</b></span>
<span class="badge">Recall: <b>{winner['test_recall']:.4f}</b></span>
</div>
<div class="card"><h2>Best classifier per feature model</h2>{_html_table(best_df, columns)}</div>
<div class="card"><h2>Metric comparison</h2><img src="data:image/png;base64,{metric_chart}"></div>
<div class="card"><h2>Confusion matrices</h2><img src="data:image/png;base64,{confusion_chart}"></div>
<div class="card"><h2>All evaluated configurations</h2>{_html_table(results_df.sort_values('test_f1', ascending=False), columns)}</div>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def save_outputs(results_df: pd.DataFrame, best_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "anatel_grid_results_all.csv", index=False)
    best_df.to_csv(output_dir / "anatel_grid_results_best.csv", index=False)
    save_metric_chart(best_df, output_dir / "figures" / "best_model_metrics.png")
    save_confusion_matrices(best_df, output_dir / "figures" / "best_confusion_matrices.png")
    build_html_report(results_df, best_df, output_dir / "anatel_comparison_report.html")
