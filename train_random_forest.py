from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
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
print("RANDOM FOREST - MECHANISTIC DATASET")
print("=" * 70)

X_train = pd.read_csv(DATA_DIR / "mechanistic_X_train.csv")
X_val = pd.read_csv(DATA_DIR / "mechanistic_X_validation.csv")
X_test = pd.read_csv(DATA_DIR / "mechanistic_X_test.csv")

y_train = pd.read_csv(DATA_DIR / "mechanistic_y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "mechanistic_y_validation.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "mechanistic_y_test.csv").squeeze()

print(f"\nTraining data:   {X_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Testing data:    {X_test.shape}")

# ============================================================
# CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=20260903,
    n_jobs=-1
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest model...")
print("Please wait...")

model.fit(X_train, y_train)

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
print("RANDOM FOREST RESULTS - MECHANISTIC DATASET")
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

model_path = MODEL_DIR / "random_forest_mechanistic.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to:")
print(model_path)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Dataset": ["Mechanistic"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

results_path = RESULT_DIR / "random_forest_mechanistic_results.csv"

results.to_csv(results_path, index=False)

print(f"\nResults saved to:")
print(results_path)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Risk_Probability": y_probability
})

prediction_path = RESULT_DIR / "random_forest_mechanistic_predictions.csv"

predictions.to_csv(prediction_path, index=False)

print(f"Predictions saved to:")
print(prediction_path)

print("\n" + "=" * 70)
print("RANDOM FOREST MECHANISTIC EXPERIMENT COMPLETED")
print("=" * 70)