from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn_crfsuite import CRF
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "preprocessed_data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MECHANISTIC DATA
# ============================================================

print("=" * 70)
print("CRF - MECHANISTIC DATASET")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "mechanistic_X_train.csv")
X_val = pd.read_csv(DATA_DIR / "mechanistic_X_validation.csv")
X_test = pd.read_csv(DATA_DIR / "mechanistic_X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "mechanistic_y_train.csv"
).squeeze()

y_val = pd.read_csv(
    DATA_DIR / "mechanistic_y_validation.csv"
).squeeze()

y_test = pd.read_csv(
    DATA_DIR / "mechanistic_y_test.csv"
).squeeze()

print(f"\nTraining data:   {X_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Testing data:    {X_test.shape}")

# ============================================================
# CONVERT TABULAR DATA TO CRF SEQUENCES
# ============================================================
#
# Each patient is represented as one sequence.
# Each feature becomes one observation in that sequence.
#
# Example:
# Patient -> [age, BMI, BP, Albumin, UACR, KIM-1, NGAL, ...]
#
# The complete sequence receives the patient's binary target.
# ============================================================

def create_crf_sequences(X, y):
    X_sequences = []
    y_sequences = []

    feature_names = list(X.columns)

    for i in range(len(X)):
        patient_sequence = []

        for feature in feature_names:
            value = X.iloc[i][feature]

            patient_sequence.append({
                "feature": feature,
                "value": str(round(float(value), 6))
            })

        X_sequences.append(patient_sequence)

        label = "High" if int(y.iloc[i]) == 1 else "Low"

        y_sequences.append([label] * len(patient_sequence))

    return X_sequences, y_sequences


print("\nConverting data into CRF sequences...")

X_train_seq, y_train_seq = create_crf_sequences(
    X_train,
    y_train
)

X_val_seq, y_val_seq = create_crf_sequences(
    X_val,
    y_val
)

X_test_seq, y_test_seq = create_crf_sequences(
    X_test,
    y_test
)

print("CRF sequence conversion completed.")

# ============================================================
# CREATE CRF MODEL
# ============================================================

print("\nCreating CRF model...")

crf = CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining CRF model...")
print("Please wait...")

crf.fit(
    X_train_seq,
    y_train_seq
)

print("Training completed successfully.")

# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

predicted_sequences = crf.predict(X_test_seq)

# ============================================================
# CONVERT SEQUENCE OUTPUT TO PATIENT LABEL
# ============================================================

y_pred = []

for sequence in predicted_sequences:

    high_count = np.sum(sequence == "High")
    low_count = np.sum(sequence == "Low")

    if high_count >= low_count:
        y_pred.append(1)
    else:
        y_pred.append(0)

# ============================================================
# CRF PROBABILITY ESTIMATE
# ============================================================
#
# CRF provides marginal probabilities for labels.
# We use the average High probability across the
# patient's sequence as the patient-level probability.
# ============================================================

marginal_probabilities = crf.predict_marginals(X_test_seq)

y_probability = []

for patient_marginals in marginal_probabilities:

    high_probabilities = []

    for position in patient_marginals:
        high_probabilities.append(
            position.get("High", 0.0)
        )

    average_high_probability = (
        sum(high_probabilities) /
        len(high_probabilities)
    )

    y_probability.append(
        average_high_probability
    )

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CRF RESULTS - MECHANISTIC DATASET")
print("=" * 70)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low Risk",
            "High Risk"
        ],
        zero_division=0
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "crf_mechanistic.joblib"

joblib.dump(
    crf,
    model_path
)

print("\nModel saved to:")
print(model_path)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["CRF"],
    "Dataset": ["Mechanistic"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

results_path = (
    RESULT_DIR /
    "crf_mechanistic_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Risk_Probability": y_probability
})

prediction_path = (
    RESULT_DIR /
    "crf_mechanistic_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved to:")
print(prediction_path)

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("CRF MECHANISTIC EXPERIMENT COMPLETED")
print("=" * 70)