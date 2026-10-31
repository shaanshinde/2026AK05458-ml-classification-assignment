"""Train and evaluate the classification models for ML Assignment 2.

Each model is tuned with GridSearchCV on data/train_data.csv only, then
evaluated once on test_data.csv. test_data.csv is never used for training
or tuning.

Usage:
    python train_models.py --model logistic_regression   # train one model
    python train_models.py                               # train every model in MODELS

For each trained model this script:
- saves the best pipeline to model/<name>.joblib
- adds or updates its row in model/metrics.csv
- adds or updates its column of per-class test F1 in model/per_class_f1.csv
- adds or updates its best parameters in model/best_params.json
- prints the test metrics, best parameters and classification report
It also saves model/feature_columns.json and model/class_labels.json.

To add a new model, add an entry to MODELS.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_utils import FEATURE_COLUMNS, METRIC_NAMES, TARGET_COLUMN, compute_metrics

RANDOM_STATE = 42
CV_FOLDS = 5
CV_SCORING = "f1_macro"

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train_data.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.csv"
PER_CLASS_F1_PATH = MODEL_DIR / "per_class_f1.csv"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.json"

# Model definitions. The key is used for the file name (model/<key>.joblib),
# "display_name" is what appears in metrics.csv and the README table.
MODELS = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]),
        "param_grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy"],
        },
    },
    "knn": {
        "display_name": "KNN",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    },
    "naive_bayes": {
        "display_name": "Naive Bayes",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "param_grid": {"model__var_smoothing": np.logspace(-12, -2, 10)},
    },
    "svm": {
        "display_name": "Support Vector Machine",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "param_grid": {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1, 1],
        },
    },
}
TOP_FEATURES_TO_SHOW = 10


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {sorted(missing)}")
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def update_metrics_csv(display_name: str, metrics: dict) -> None:
    row = {"Model": display_name, **metrics}
    if METRICS_PATH.exists():
        table = pd.read_csv(METRICS_PATH)
        if display_name in table["Model"].values:
            table.loc[table["Model"] == display_name, METRIC_NAMES] = [
                metrics[m] for m in METRIC_NAMES
            ]
        else:
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table[["Model"] + METRIC_NAMES].to_csv(METRICS_PATH, index=False, float_format="%.4f")


def update_per_class_f1(display_name: str, model, y_true, y_pred) -> None:
    # Rows are the fault types, one column per model.
    scores = f1_score(y_true, y_pred, labels=model.classes_, average=None, zero_division=0)
    column = pd.Series(scores, index=pd.Index(model.classes_, name="Fault_Type")).round(4)
    if PER_CLASS_F1_PATH.exists():
        table = pd.read_csv(PER_CLASS_F1_PATH, index_col="Fault_Type")
    else:
        table = pd.DataFrame(index=column.index)
    table[display_name] = column
    table.to_csv(PER_CLASS_F1_PATH, float_format="%.4f")


def update_best_params(name: str, best_params: dict) -> None:
    all_params = {}
    if BEST_PARAMS_PATH.exists():
        all_params = json.loads(BEST_PARAMS_PATH.read_text())
    all_params[name] = best_params
    BEST_PARAMS_PATH.write_text(json.dumps(all_params, indent=2) + "\n")


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def train_one(name: str, X_train, y_train, X_test, y_test) -> None:
    spec = MODELS[name]
    display_name = spec["display_name"]
    print(f"\n=== {display_name} ===")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        spec["pipeline"],
        spec["param_grid"],
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(best_model, model_path)

    metrics = compute_metrics(best_model, X_test, y_test)
    y_pred = best_model.predict(X_test)
    update_metrics_csv(display_name, metrics)
    update_per_class_f1(display_name, best_model, y_test, y_pred)
    update_best_params(name, search.best_params_)

    print(f"Saved model: {model_path.relative_to(ROOT)}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV {CV_SCORING} (train data): {search.best_score_:.4f}")
    print("\nTest metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<10}{value:.4f}")
    print("\nClassification report (test data):")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    final_step = best_model.named_steps["model"]
    if hasattr(final_step, "feature_importances_"):
        importances = pd.Series(final_step.feature_importances_, index=FEATURE_COLUMNS)
        top = importances.sort_values(ascending=False).head(TOP_FEATURES_TO_SHOW)
        print(f"Top {TOP_FEATURES_TO_SHOW} most important features:")
        print(top.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        help="Train only this model. If left out, every model in MODELS is trained.",
    )
    args = parser.parse_args()

    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}, "
          f"features: {len(FEATURE_COLUMNS)}")

    MODEL_DIR.mkdir(exist_ok=True)
    save_json(FEATURE_COLUMNS_PATH, FEATURE_COLUMNS)
    save_json(CLASS_LABELS_PATH, sorted(y_train.unique()))

    names = [args.model] if args.model else list(MODELS)
    for name in names:
        train_one(name, X_train, y_train, X_test, y_test)

    print(f"\nMetrics table: {METRICS_PATH.relative_to(ROOT)}")
    print(pd.read_csv(METRICS_PATH).to_string(index=False))


if __name__ == "__main__":
    main()
