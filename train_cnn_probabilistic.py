import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
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
print("1-D CNN - PROBABILISTIC DATASET")
print("=" * 70)

print("\nLoading data...")

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
# CONVERT TARGET TO NUMPY
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
# CONVERT TO NUMPY
# ============================================================

X_train = X_train.values.astype(np.float32)
X_validation = X_validation.values.astype(np.float32)
X_test = X_test.values.astype(np.float32)


# ============================================================
# RESHAPE FOR 1-D CNN
# ============================================================

# Original shape:
# (samples, features)
#
# CNN requires:
# (samples, features, channels)

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_validation = X_validation.reshape(
    X_validation.shape[0],
    X_validation.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nCNN input shape:")
print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)
print("X_test       :", X_test.shape)


# ============================================================
# BUILD 1-D CNN MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 1-D CNN")
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

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nModel summary:")
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
        X_validation,
        y_validation
    ),
    epochs=30,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)


print("\nCNN training completed.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# CALCULATE METRICS
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
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("1-D CNN PROBABILISTIC TEST RESULTS")
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
    "cnn_1d_probabilistic.keras"
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
    "Model": ["1-D CNN"],
    "Dataset": ["Probabilistic"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

results_path = os.path.join(
    RESULT_DIR,
    "cnn_1d_probabilistic_results.csv"
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
    "Predicted": y_prediction,
    "Predicted_Probability": y_probability
})

predictions_path = os.path.join(
    RESULT_DIR,
    "cnn_1d_probabilistic_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)

print("\nPredictions saved to:")
print(predictions_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("1-D CNN PROBABILISTIC EXPERIMENT COMPLETED")
print("=" * 70)