"""Shared code used by both train_models.py and the Streamlit app (app.py).

Keeping the feature list and the metric calculation in one place makes sure
the models are trained and evaluated in exactly the same way everywhere.
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "Fault_Type"

# The 27 input features of the Steel Plates Faults dataset (UCI id 198),
# in the same order as in data/steel_plates_clean.csv.
FEATURE_COLUMNS = [
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
]

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(model, X, y) -> dict:
    """Return the 6 evaluation metrics for a fitted classifier, rounded to 4 decimals.

    Precision, Recall and F1 are macro-averaged over the classes.
    AUC is one-vs-rest, macro-averaged, computed from predict_proba.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(
            y, y_proba, multi_class="ovr", average="macro", labels=model.classes_
        ),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}
