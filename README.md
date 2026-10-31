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
| KNN | 0.7558 | 0.9251 | 0.7453 | 0.7687 | 0.7546 | 0.6884 |
| Naive Bayes | 0.6195 | 0.9240 | 0.6354 | 0.7212 | 0.6379 | 0.5609 |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Overall:** Accuracy 72.8%, macro F1 0.74, MCC 0.65. A solid linear baseline.<br>**Strong on:** K_Scratch, Z_Scratch and Stains, which are easy to separate using shape and size features.<br>**Weak on:** Pastry, Bumps and Other_Faults, which overlap and need curved boundaries that a linear model can't draw.<br>**Key point:** High AUC (0.94), so the correct fault is usually ranked near the top even when the first guess is wrong. |
| Decision Tree | **Overall:** Accuracy 75.3%, macro F1 0.76, MCC 0.68. Better than Logistic Regression on 5 of the 6 metrics.<br>**Strong on:** Bumps, Z_Scratch and Dirtiness, as its step-by-step rules can follow overlapping regions.<br>**Weak on:** Pastry (13 of 32 found). AUC is lower (0.86) because almost every prediction is made with 100% confidence.<br>**Key point:** Clearly overfitted (training F1 0.99 against test F1 0.76), but cross-validation and test scores match closely, so 0.76 is an honest estimate. |
| KNN | **Overall:** Accuracy 75.6%, recall 0.77, MCC 0.69, slightly ahead of the Decision Tree on these three, though macro F1 (0.75) is a little lower.<br>**Strong on:** K_Scratch (F1 0.96, only 3 of 78 missed), as scratch plates form a tight group once scaled. AUC is high (0.93) since probabilities come from a weighted vote of 7 neighbours.<br>**Weak on:** Pastry (13 of 32 found), as its plates get outvoted by neighbours from the larger Bumps and Other_Faults classes.<br>**Key point:** Scaling is essential. Without it, cross-validation F1 drops from 0.77 to 0.33, because Y_Minimum, measured in millions, dominates the distance. |
| Naive Bayes | **Overall:** Accuracy 62.0%, macro F1 0.64, MCC 0.56, below Logistic Regression, Decision Tree and KNN on all three.<br>**Strong on:** K_Scratch, Z_Scratch and Stains. AUC stays high (0.92) because the true fault is usually ranked near the top.<br>**Weak on:** Other_Faults (only 38 of 135 found, mostly labelled Bumps or Pastry), since this catch-all class doesn't fit the single bell curve per feature that Gaussian Naive Bayes assumes.<br>**Key point:** Several features measure the same thing (for example defect size), so the independence assumption counts the same evidence many times. Training and test F1 are both about 0.63, so it underfits rather than overfits. Reshaping skewed features with PowerTransformer improved cross-validation F1 by only 0.01. |
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

## g. Additional Analysis: F1 by Fault Type

This table shows how well each model identifies each fault type on the test data (F1 score per fault type), along with the number of test plates for each fault type.

| Fault Type | Test Plates | Logistic Regression | Decision Tree | KNN | Naive Bayes |
|---|---|---|---|---|---|
| Bumps | 81 | 0.6216 | 0.7186 | 0.6864 | 0.5951 |
| Dirtiness | 11 | 0.7368 | 0.8182 | 0.8182 | 0.4737 |
| K_Scratch | 78 | 0.9281 | 0.8571 | 0.9615 | 0.9150 |
| Other_Faults | 135 | 0.6761 | 0.7143 | 0.7000 | 0.4086 |
| Pastry | 32 | 0.5373 | 0.4643 | 0.4483 | 0.4348 |
| Stains | 14 | 0.8571 | 0.8571 | 0.8000 | 0.8000 |
| Z_Scratch | 38 | 0.8354 | 0.9231 | 0.8675 | 0.8378 |
