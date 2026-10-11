import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 20260903

np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "preprocessed_data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MECHANISTIC DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED MECHANISTIC DATA")
print("=" * 70)

X_train = pd.read_csv(
    DATA_DIR / "mechanistic_X_train.csv"
)

X_val = pd.read_csv(
    DATA_DIR / "mechanistic_X_validation.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "mechanistic_X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "mechanistic_y_train.csv"
).iloc[:, 0].values

y_val = pd.read_csv(
    DATA_DIR / "mechanistic_y_validation.csv"
).iloc[:, 0].values

y_test = pd.read_csv(
    DATA_DIR / "mechanistic_y_test.csv"
).iloc[:, 0].values


print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = X_train.values.astype(np.float32)
X_val = X_val.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)


# ============================================================
# RESHAPE FOR 1-D CNN
# ============================================================
#
# CNN expects:
#
# samples × timesteps × channels
#
# Here:
# 50,000 patient records
# 30 features
# 1 channel
#
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


print("\nCNN input shape:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# BUILD 1-D CNN
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 1-D CNN MODEL")
print("=" * 70)

model = Sequential([

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], 1)
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING 1-D CNN")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=256,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("\nTraining completed successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("1-D CNN RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_prediction
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_prediction,
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

model_path = (
    MODEL_DIR /
    "cnn_1d_mechanistic.keras"
)

model.save(
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Model": [
        "1-D CNN"
    ],

    "Dataset": [
        "Mechanistic"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ],

    "ROC_AUC": [
        roc_auc
    ]
})


results_path = (
    RESULTS_DIR /
    "cnn_1d_mechanistic_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_prediction,

    "Risk_Probability": y_probability
})


predictions_path = (
    RESULTS_DIR /
    "cnn_1d_mechanistic_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)


print("\nResults saved to:")
print(results_path)

print("\nPredictions saved to:")
print(predictions_path)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("1-D CNN MECHANISTIC EXPERIMENT COMPLETED")
print("=" * 70)