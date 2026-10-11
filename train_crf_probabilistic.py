import os
import numpy as np
import pandas as pd
import joblib

from sklearn_crfsuite import CRF
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD PROBABILISTIC DATA
# ============================================================

print("=" * 70)
print("CRF - PROBABILISTIC DATASET")
print("=" * 70)

print("\nLoading preprocessed probabilistic dataset...")

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_X_train.csv")
)

X_validation = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_X_validation.csv")
)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_y_train.csv")
)

y_validation = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_y_validation.csv")
)

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "probabilistic_y_test.csv")
)


# ============================================================
# EXTRACT TARGET
# ============================================================

y_train = y_train.iloc[:, 0].astype(int).values
y_validation = y_validation.iloc[:, 0].astype(int).values
y_test = y_test.iloc[:, 0].astype(int).values


print("\nDataset shapes:")
print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)
print("X_test       :", X_test.shape)

print("\nTarget distribution:")
print("Train:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nValidation:")
print(pd.Series(y_validation).value_counts().sort_index())

print("\nTest:")
print(pd.Series(y_test).value_counts().sort_index())


# ============================================================
# CONVERT TABULAR DATA INTO CRF SEQUENCES
# ============================================================

def create_sequences(X, y):
    """
    Convert each patient's tabular feature vector into
    a sequence of observations for the CRF.

    Each feature becomes one observation in the sequence.
    """

    sequences = []
    labels = []

    feature_names = list(X.columns)

    for row_index in range(len(X)):

        row = X.iloc[row_index]

        sequence = []

        for feature_name in feature_names:

            value = row[feature_name]

            observation = {
                "feature": feature_name,
                "value": str(value)
            }

            sequence.append(observation)

        sequences.append(sequence)

        # Same patient-level target is assigned to
        # every observation in the sequence.
        patient_label = "High" if y[row_index] == 1 else "Low"

        patient_labels = [
            patient_label
            for _ in range(len(sequence))
        ]

        labels.append(patient_labels)

    return sequences, labels


print("\nConverting data into CRF sequences...")

X_train_seq, y_train_seq = create_sequences(
    X_train,
    y_train
)

X_validation_seq, y_validation_seq = create_sequences(
    X_validation,
    y_validation
)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test
)


print("CRF sequence conversion completed.")

print("\nNumber of training sequences :", len(X_train_seq))
print("Number of validation sequences:", len(X_validation_seq))
print("Number of test sequences      :", len(X_test_seq))


# ============================================================
# TRAIN CRF
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CRF")
print("=" * 70)

crf = CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(
    X_train_seq,
    y_train_seq
)

print("\nCRF training completed.")


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating on validation data...")

validation_predictions = crf.predict(X_validation_seq)

print("Validation evaluation completed.")


# ============================================================
# TEST PREDICTION
# ============================================================

print("\nGenerating test predictions...")

test_predictions = crf.predict(X_test_seq)

print("Test prediction completed.")


# ============================================================
# CONVERT SEQUENCE PREDICTION TO PATIENT PREDICTION
# ============================================================

def sequence_to_patient_prediction(sequence):
    """
    Convert CRF sequence prediction into one
    patient-level binary prediction.

    Majority voting:
    High = 1
    Low  = 0
    """

    high_count = np.sum(
        np.array(sequence) == "High"
    )

    low_count = np.sum(
        np.array(sequence) == "Low"
    )

    if high_count >= low_count:
        return 1
    else:
        return 0


patient_predictions = []

for sequence in test_predictions:

    prediction = sequence_to_patient_prediction(
        sequence
    )

    patient_predictions.append(prediction)

patient_predictions = np.array(
    patient_predictions
)


# ============================================================
# CRF PROBABILITY ESTIMATION
# ============================================================

print("\nCalculating CRF probabilities...")

marginal_probabilities = crf.predict_marginals(
    X_test_seq
)

patient_probabilities = []

for sequence_probabilities in marginal_probabilities:

    high_probabilities = []

    for observation_probability in sequence_probabilities:

        high_probability = observation_probability.get(
            "High",
            0.0
        )

        high_probabilities.append(
            high_probability
        )

    # Average probability across the patient's
    # sequence of observations.
    patient_probability = np.mean(
        high_probabilities
    )

    patient_probabilities.append(
        patient_probability
    )

patient_probabilities = np.array(
    patient_probabilities
)


# ============================================================
# EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    patient_predictions
)

precision = precision_score(
    y_test,
    patient_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    patient_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    patient_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    patient_probabilities
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CRF PROBABILISTIC TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "crf_probabilistic.joblib"
)

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
    "Dataset": ["Probabilistic"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

results_path = os.path.join(
    RESULT_DIR,
    "crf_probabilistic_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": patient_predictions,
    "Predicted_Probability": patient_probabilities
})

predictions_path = os.path.join(
    RESULT_DIR,
    "crf_probabilistic_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)

print("\nPredictions saved to:")
print(predictions_path)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CRF PROBABILISTIC EXPERIMENT COMPLETED")
print("=" * 70)