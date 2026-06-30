from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATASETS
from .data import dataset_summary, load_feature_base
from .evaluation import run_grid_experiment
from .reporting import save_dataset_distribution, save_outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare LSTM, GRU, and TCN handover prediction feature bases.")
    parser.add_argument("--data-dir", default="data/processed", help="Folder containing processed concatbase CSV files.")
    parser.add_argument("--output-dir", default="results", help="Folder for CSV, figures, and HTML report.")
    parser.add_argument("--scoring", default="f1", help="GridSearchCV scoring metric. Recommended: f1")
    parser.add_argument("--cv", type=int, default=3, help="Stratified cross-validation folds.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test-set fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE inside CV/training pipeline.")
    parser.add_argument("--full-grid", action="store_true", help="Use a larger classifier hyperparameter grid.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    y_counts = {}
    for spec in DEFAULT_DATASETS:
        _, y, _ = load_feature_base(data_dir / spec.filename)
        y_counts[spec.feature_model] = dataset_summary(y)

    results_df, best_df = run_grid_experiment(
        data_dir=data_dir,
        datasets=DEFAULT_DATASETS,
        scoring=args.scoring,
        cv_folds=args.cv,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        use_smote=args.use_smote,
        full_grid=args.full_grid,
    )

    save_outputs(results_df, best_df, output_dir)
    save_dataset_distribution(y_counts, output_dir / "figures" / "class_distribution.png")

    print("\nExperiment complete.")
    print(f"All results:  {output_dir / 'anatel_grid_results_all.csv'}")
    print(f"Best results: {output_dir / 'anatel_grid_results_best.csv'}")
    print(f"HTML report:  {output_dir / 'anatel_comparison_report.html'}")
    print("\nBest model for each feature generator:")
    cols = ["feature_model", "classifier", "test_accuracy", "test_balanced_accuracy", "test_recall", "test_f1", "test_roc_auc"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(best_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
