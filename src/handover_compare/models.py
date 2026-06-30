from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


def make_pipeline(estimator, use_smote: bool, random_state: int = 42):
    """Build a leakage-safe classification pipeline.

    When SMOTE is enabled, it is inserted after scaling and before the
    classifier. Because the pipeline is passed to GridSearchCV, resampling is
    performed inside training folds only, not on the held-out test split.
    """
    steps = [("scaler", StandardScaler())]
    if use_smote:
        if not IMBLEARN_AVAILABLE:
            raise ImportError("Install imbalanced-learn or run without --use-smote.")
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=3)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)

    steps.append(("clf", estimator))
    return SklearnPipeline(steps)


def classifier_search_spaces(use_smote: bool, full_grid: bool, random_state: int = 42):
    """Return candidate classifiers and hyperparameter grids."""
    if full_grid:
        rf_grid = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 3],
            "clf__class_weight": [None, "balanced"],
        }
        svm_grid = {
            "clf__C": [1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__class_weight": [None, "balanced"],
        }
        knn_grid = {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
        log_grid = {
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": [None, "balanced"],
        }
        gb_grid = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [2, 3],
        }
    else:
        rf_grid = {"clf__n_estimators": [100], "clf__max_depth": [None, 10], "clf__class_weight": [None, "balanced"]}
        svm_grid = {"clf__C": [1, 10], "clf__gamma": ["scale"], "clf__class_weight": [None, "balanced"]}
        knn_grid = {"clf__n_neighbors": [3, 5], "clf__weights": ["distance"], "clf__p": [1]}
        log_grid = {"clf__C": [1, 10], "clf__class_weight": [None, "balanced"]}
        gb_grid = {"clf__n_estimators": [100], "clf__learning_rate": [0.1], "clf__max_depth": [3]}

    spaces = {
        "RandomForest": (
            make_pipeline(RandomForestClassifier(random_state=random_state, n_jobs=1), use_smote, random_state),
            rf_grid,
        ),
        "SVM_RBF": (
            make_pipeline(SVC(kernel="rbf", probability=True, random_state=random_state), use_smote, random_state),
            svm_grid,
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier(), use_smote, random_state),
            knn_grid,
        ),
        "LogisticRegression": (
            make_pipeline(LogisticRegression(max_iter=5000, random_state=random_state), use_smote, random_state),
            log_grid,
        ),
    }
    if full_grid:
        spaces["GradientBoosting"] = (
            make_pipeline(GradientBoostingClassifier(random_state=random_state), use_smote, random_state),
            gb_grid,
        )
    return spaces
