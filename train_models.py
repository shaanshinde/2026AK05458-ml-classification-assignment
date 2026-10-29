"""Train and evaluate the classification models for ML Assignment 2.

Planned responsibilities:
- Load the raw dataset from data/.
- Preprocess features (encoding, scaling) and split into train/test sets.
- Train five models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM.
- Evaluate each model on Accuracy, AUC, Precision, Recall, F1 and MCC.
- Save the fitted models plus scaler/encoder artifacts to model/ for app.py.
- Export a held-out test split as test_data.csv for use in the Streamlit app.
"""
