# ML Classification Assignment

BITS Pilani WILP, M.Tech, Machine Learning Assignment 2.

- **Name:** SHINDE SHANTANU GANESH
- **BITS Student ID:** 2026ak05458

## a. Problem Statement

Steel plates are checked for surface defects during manufacturing. Identifying the type of each defect by hand is slow and inconsistent, so the aim of this project is to classify defects automatically from measurements taken of each one.

Given the 27 measured features of a defect (its position, size, shape, brightness, and the type and thickness of the steel), the task is to predict which of 7 fault types it belongs to: Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps or Other_Faults. This is a multi-class classification problem.

Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes and Support Vector Machine) are trained on the same data and compared on Accuracy, AUC, Precision, Recall, F1 and MCC to find which works best for this dataset. The trained models are then made available in a Streamlit app, where a user can upload test data and see the predictions and evaluation metrics.

## b. Dataset Description

**Dataset:** Steel Plates Faults, from the UCI Machine Learning Repository (dataset id 198): https://archive.ics.uci.edu/dataset/198/steel+plates+faults

**Problem:** Each row describes a surface defect found on a stainless steel plate. The features describe the defect's position, size, shape, brightness (luminosity) and the type and thickness of the steel. The goal is to predict which type of fault the defect is.

**Size:** 1,941 rows and 27 numeric input features. There are no missing values.

**Target:** In the original data the fault type is stored as 7 separate 0/1 columns. Every row has exactly one of them set to 1, so `prepare_data.py` combines them into a single text column called `Fault_Type`. This makes it a **multi-class classification problem** with 7 classes:

| Fault_Type | Count | Percent |
|---|---|---|
| Other_Faults | 673 | 34.7% |
| Bumps | 402 | 20.7% |
| K_Scratch | 391 | 20.1% |
| Z_Scratch | 190 | 9.8% |
| Pastry | 158 | 8.1% |
| Stains | 72 | 3.7% |
| Dirtiness | 55 | 2.8% |

The classes are imbalanced, so the data is split 80/20 into train (1,552 rows) and test (389 rows) sets using a stratified split with `random_state=42`. The test set is saved as `test_data.csv` at the repo root. The exploratory analysis is in `notebooks/01_eda.ipynb`.

## c. GitHub Repository Link

https://github.com/shaanshinde/2026AK05458-ml-classification-assignment

## d. Models Used

TODO: Fill in the metrics for the remaining models after running `train_models.py`.

All metrics are measured on the test set (`test_data.csv`, 389 rows). Precision, Recall and F1 are macro-averaged over the 7 classes, and AUC is one-vs-rest macro-averaged.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7532 | 0.8583 | 0.7739 | 0.7597 | 0.7647 | 0.6811 |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline with 72.8% accuracy, macro F1 of 0.74 and MCC of 0.65. AUC is high (0.94), so even when its first guess is wrong, the correct fault is usually ranked near the top. It does very well on K_Scratch, Z_Scratch and Stains, which are easy to separate using shape and size features. It struggles with Pastry, Bumps and Other_Faults, which overlap with each other and need curved boundaries that a linear model can't draw. |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| Overall, Winner for your dataset? | |

## e. Streamlit App Link

TODO: Add the Streamlit Community Cloud app URL.

## f. How to Run Locally

TODO: Finalise the steps below once the app is built.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`test_data.csv` (to be added at the repo root) is a sample test file to upload in the app.
