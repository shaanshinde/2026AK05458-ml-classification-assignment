"""Streamlit app for ML Assignment 2: Steel Plates Fault Classification.

Loads the 5 models trained offline by train_models.py from model/ and
evaluates them on an uploaded CSV file. The app never trains or tunes a model.

Run with: streamlit run app.py
"""
import json
import math
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import FEATURE_COLUMNS, METRIC_NAMES, TARGET_COLUMN, compute_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.json"
SAMPLE_DATA_PATH = ROOT / "test_data.csv"

# Name shown in the app -> saved model file in model/.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_HELP = {
    "Accuracy": "Share of plates whose fault type was predicted correctly.",
    "AUC": "One-vs-rest ROC AUC, macro-averaged over the fault types. "
           "Shows how well the model ranks the true fault above the others.",
    "Precision": "Macro-averaged: when the model predicts a fault type, how often it is right.",
    "Recall": "Macro-averaged: the share of each fault type's plates that the model finds.",
    "F1": "Macro-averaged balance of precision and recall.",
    "MCC": "Matthews correlation coefficient: 1 is perfect, 0 is no better than chance.",
}

# Fault type counts in the full dataset (1,941 plates).
FAULT_COUNTS = {
    "Other_Faults": 673,
    "Bumps": 402,
    "K_Scratch": 391,
    "Z_Scratch": 190,
    "Pastry": 158,
    "Stains": 72,
    "Dirtiness": 55,
}

HIGHLIGHT_STYLE = "font-weight: bold; background-color: rgba(46, 160, 67, 0.25)"
TOP_BAR_COLOUR = "#2EA043"
OTHER_BAR_COLOUR = "#9AA4AE"


@st.cache_resource
def load_models() -> dict:
    """Load the saved pipelines once and share them across reruns and users."""
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_resource
def load_class_labels() -> list[str]:
    return json.loads(CLASS_LABELS_PATH.read_text())


def format_metric(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.4f}"


def read_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read **{uploaded_file.name}** as a CSV file: {exc}")
        return None
    if df.empty:
        st.error(f"**{uploaded_file.name}** has no data rows.")
        return None
    return df


def validate(df: pd.DataFrame, class_labels: list[str]) -> tuple[pd.DataFrame, pd.Series] | None:
    """Check the uploaded data. Show a message for every problem found.

    Returns the features and true labels if the data can be evaluated, else None.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(
            f"The file is missing {len(missing)} of the {len(FEATURE_COLUMNS)} feature "
            f"columns the models need: {', '.join(missing)}"
        )
    if TARGET_COLUMN not in df.columns:
        st.error(
            f"The file has no **{TARGET_COLUMN}** column. The true fault type of each "
            "plate is needed to calculate the evaluation metrics, so please add it."
        )
    if missing or TARGET_COLUMN not in df.columns:
        return None

    extra = [c for c in df.columns if c not in FEATURE_COLUMNS and c != TARGET_COLUMN]
    if extra:
        st.info(f"These extra columns are not used by the models and are ignored: {', '.join(extra)}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_columns = [c for c in FEATURE_COLUMNS if X[c].isna().any()]
    if bad_columns:
        st.error(f"These feature columns have empty or non-numeric values: {', '.join(bad_columns)}")
        return None

    y = df[TARGET_COLUMN]
    if y.isna().any():
        st.error(f"{int(y.isna().sum())} rows have an empty **{TARGET_COLUMN}** value.")
        return None
    y = y.astype(str).str.strip()
    unknown = sorted(set(y) - set(class_labels))
    if unknown:
        st.error(
            f"Unknown fault labels in **{TARGET_COLUMN}**: {', '.join(unknown)}. "
            f"Allowed labels are: {', '.join(class_labels)}"
        )
        return None
    return X, y


def show_selected_model(model_name: str, model, X, y, class_labels: list[str]) -> None:
    st.subheader(model_name)
    metrics = compute_metrics(model, X, y)
    columns = st.columns(3)
    for i, name in enumerate(METRIC_NAMES):
        columns[i % 3].metric(name, format_metric(metrics[name]), help=METRIC_HELP[name], border=True)
    if math.isnan(metrics["AUC"]):
        st.info(
            "AUC could not be calculated because not every fault type appears in the "
            "uploaded file. AUC is measured for each fault type against the rest, so each "
            "of the 7 fault types needs at least one plate."
        )

    y_pred = model.predict(X)
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(y, y_pred, labels=class_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel("Predicted fault type")
        ax.set_ylabel("True fault type")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        st.dataframe(classification_report_table(y, y_pred, class_labels), placeholder="")


def classification_report_table(y, y_pred, class_labels: list[str]):
    report = classification_report(
        y, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    accuracy = report.pop("accuracy")
    table = pd.DataFrame(report).T
    accuracy_row = pd.DataFrame(
        {"precision": [math.nan], "recall": [math.nan], "f1-score": [accuracy], "support": [len(y)]},
        index=["accuracy"],
    )
    # Same row order as the printed report: classes, accuracy, then the averages.
    table = pd.concat([table.loc[class_labels], accuracy_row, table.loc[["macro avg", "weighted avg"]]])
    table.columns = ["Precision", "Recall", "F1", "Support"]
    table["Support"] = table["Support"].astype(int)
    return table.style.format("{:.4f}", subset=["Precision", "Recall", "F1"], na_rep="")


def show_comparison(models: dict, X, y) -> None:
    st.subheader("All models on the uploaded data")
    table = pd.DataFrame(
        {name: compute_metrics(model, X, y) for name, model in models.items()}
    ).T[METRIC_NAMES]
    table.index.name = "Model"
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, props=HIGHLIGHT_STYLE),
        placeholder="N/A",
    )
    st.caption("The best value in each column is highlighted.")

    metric = st.radio("Metric to chart", METRIC_NAMES, index=METRIC_NAMES.index("F1"), horizontal=True)
    values = table[metric].dropna()
    if values.empty:
        st.info(f"{metric} is not available for the uploaded data, so there is nothing to chart.")
        return
    st.altair_chart(metric_bar_chart(values, metric))


def metric_bar_chart(values: pd.Series, metric: str) -> alt.LayerChart:
    """Horizontal bars sorted from highest to lowest, with the top model in a different colour."""
    data = values.rename("Value").reset_index()
    data["Top"] = data["Value"] == data["Value"].max()
    # Leave room on the right for the value labels.
    x_scale = alt.Scale(domain=[min(0.0, data["Value"].min()), max(1.0, data["Value"].max() + 0.1)], nice=False)
    base = alt.Chart(data).encode(
        y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X("Value:Q", title=metric, scale=x_scale),
    )
    bars = base.mark_bar().encode(
        color=alt.condition(alt.datum.Top, alt.value(TOP_BAR_COLOUR), alt.value(OTHER_BAR_COLOUR))
    )
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("Value:Q", format=".4f"))
    return (bars + labels).properties(title=f"{metric} by model", height=60 * len(data))


def show_about() -> None:
    st.subheader("About this app")
    st.markdown(
        "**Dataset:** [Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) "
        "from the UCI Machine Learning Repository. Each row describes a surface defect on a "
        "stainless steel plate using 27 numeric features: its position, size, shape, "
        "brightness (luminosity), and the type and thickness of the steel. The task is to "
        "predict which of 7 fault types the defect belongs to."
    )
    st.markdown("**The 7 fault types** (number of plates in the full dataset of 1,941):")
    st.markdown("\n".join(f"- {fault}: {count}" for fault, count in FAULT_COUNTS.items()))
    st.markdown(
        "**How the models were built:** all 5 models were trained offline with "
        "`train_models.py` on the training split only, tuned with 5-fold cross-validation "
        "using macro F1, and saved to the `model/` folder. This app only loads those saved "
        "models and uses them to make predictions. It never trains or retrains a model."
    )
    st.markdown(
        "**Uploaded file format:** a CSV with the 27 feature columns and a "
        f"`{TARGET_COLUMN}` column holding the true fault type. Use "
        "**Download sample test data** in the sidebar for an example."
    )


def main() -> None:
    st.set_page_config(page_title="Steel Plates Fault Classification", layout="wide")
    st.title("Steel Plates Fault Classification")
    st.caption(
        "Predicts the fault type of a steel plate using the UCI Steel Plates Faults dataset, "
        "and compares 5 models: Logistic Regression, Decision Tree, KNN, Naive Bayes and "
        "Support Vector Machine."
    )

    try:
        models = load_models()
        class_labels = load_class_labels()
    except FileNotFoundError as exc:
        st.error(f"A saved model file is missing, so the app cannot run: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Test data")
        if SAMPLE_DATA_PATH.exists():
            st.download_button(
                "Download sample test data",
                data=SAMPLE_DATA_PATH.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
            )
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        st.header("Model")
        model_name = st.selectbox("Choose a model", list(MODEL_FILES))
        st.divider()
        st.caption(
            "**Shinde Shantanu Ganesh**  \n"
            "**BITS ID: 2026AK05458**  \n"
            "Machine Learning - Assignment 2  \n"
            "M.Tech AI/ML, BITS Pilani WILP"
        )

    data = None
    if uploaded_file is None:
        st.info(
            "Upload a CSV file in the sidebar to evaluate the models. If you do not have "
            "one, click **Download sample test data** in the sidebar first, then upload it."
        )
    else:
        df = read_csv(uploaded_file)
        if df is not None:
            st.success(f"Loaded {len(df)} rows from **{uploaded_file.name}**.")
            st.markdown("**Preview of the first 5 rows**")
            st.dataframe(df.head(5))
            data = validate(df, class_labels)

    selected_tab, compare_tab, about_tab = st.tabs(["Selected Model", "Compare All Models", "About"])
    with selected_tab:
        if data is None:
            st.info("Results appear here once a valid CSV file is uploaded.")
        else:
            show_selected_model(model_name, models[model_name], *data, class_labels)
    with compare_tab:
        if data is None:
            st.info("Results appear here once a valid CSV file is uploaded.")
        else:
            show_comparison(models, *data)
    with about_tab:
        show_about()


main()
