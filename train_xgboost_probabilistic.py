from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
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
# LOAD PROBABILISTIC DATA
# ============================================================

print("=" * 70)
print("XGBOOST - PROBABILISTIC DATASET")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "probabilistic_X_train.csv")
X_val = pd.read_csv(DATA_DIR / "probabilistic_X_validation.csv")
X_test = pd.read_csv(DATA_DIR / "probabilistic_X_test.csv")

y_train = pd.read_csv(DATA_DIR / "probabilistic_y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "probabilistic_y_validation.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "probabilistic_y_test.csv").squeeze()

print(f"\nTraining data:   {X_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Testing data:    {X_test.shape}")

# ============================================================
# CALCULATE CLASS WEIGHT
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass distribution:")
print(f"Low risk  (0): {negative}")
print(f"High risk (1): {positive}")
print(f"Scale positive weight: {scale_pos_weight:.4f}")

# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=20260903,
    n_jobs=-1
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost model...")
print("Please wait...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training completed successfully.")

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("XGBOOST RESULTS - PROBABILISTIC DATASET")
print("=" * 70)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Risk", "High Risk"]
))

# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "xgboost_probabilistic.joblib"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Dataset": ["Probabilistic"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

results_path = RESULT_DIR / "xgboost_probabilistic_results.csv"

results.to_csv(results_path, index=False)

print("\nResults saved to:")
print(results_path)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Risk_Probability": y_probability
})

prediction_path = RESULT_DIR / "xgboost_probabilistic_predictions.csv"

predictions.to_csv(prediction_path, index=False)

print("Predictions saved to:")
print(prediction_path)

print("\n" + "=" * 70)
print("XGBOOST PROBABILISTIC EXPERIMENT COMPLETED")
print("=" * 70)