import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# RESULT FILES
# ============================================================

result_files = [
    "xgboost_mechanistic_results.csv",
    "xgboost_probabilistic_results.csv",
    "lightgbm_mechanistic_results.csv",
    "lightgbm_probabilistic_results.csv",
    "random_forest_mechanistic_results.csv",
    "random_forest_probabilistic_results.csv",
    "crf_mechanistic_results.csv",
    "crf_probabilistic_results.csv",
    "cnn_1d_mechanistic_results.csv",
    "cnn_1d_probabilistic_results.csv",
]


# ============================================================
# LOAD ALL RESULTS
# ============================================================

all_results = []

for file_name in result_files:
    file_path = RESULTS_DIR / file_name

    if file_path.exists():
        df = pd.read_csv(file_path)
        all_results.append(df)
        print(f"Loaded: {file_name}")
    else:
        print(f"WARNING: File not found: {file_name}")


if not all_results:
    raise FileNotFoundError("No model result files were found.")


comparison = pd.concat(all_results, ignore_index=True)


# ============================================================
# CLEAN / ROUND VALUES
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

for column in metric_columns:
    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce"
    )

comparison = comparison.sort_values(
    by=["Dataset", "Accuracy"],
    ascending=[True, False]
).reset_index(drop=True)


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print("\n")
print("=" * 80)
print("ALL MODEL RESULTS")
print("=" * 80)

print(
    comparison[
        [
            "Model",
            "Dataset",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)


# ============================================================
# BEST MODEL - MECHANISTIC
# BASED ON ACCURACY
# ============================================================

mechanistic = comparison[
    comparison["Dataset"].str.lower() == "mechanistic"
]

if not mechanistic.empty:

    best_mechanistic = mechanistic.loc[
        mechanistic["Accuracy"].idxmax()
    ]

    print("\n")
    print("=" * 80)
    print("BEST MODEL - MECHANISTIC DATASET (BASED ON ACCURACY)")
    print("=" * 80)

    print(f"Model     : {best_mechanistic['Model']}")
    print(f"Accuracy  : {best_mechanistic['Accuracy']:.4f}")
    print(f"Precision : {best_mechanistic['Precision']:.4f}")
    print(f"Recall    : {best_mechanistic['Recall']:.4f}")
    print(f"F1 Score  : {best_mechanistic['F1_Score']:.4f}")
    print(f"ROC-AUC   : {best_mechanistic['ROC_AUC']:.4f}")


# ============================================================
# BEST MODEL - PROBABILISTIC
# BASED ON ACCURACY
# ============================================================

probabilistic = comparison[
    comparison["Dataset"].str.lower() == "probabilistic"
]

if not probabilistic.empty:

    best_probabilistic = probabilistic.loc[
        probabilistic["Accuracy"].idxmax()
    ]

    print("\n")
    print("=" * 80)
    print("BEST MODEL - PROBABILISTIC DATASET (BASED ON ACCURACY)")
    print("=" * 80)

    print(f"Model     : {best_probabilistic['Model']}")
    print(f"Accuracy  : {best_probabilistic['Accuracy']:.4f}")
    print(f"Precision : {best_probabilistic['Precision']:.4f}")
    print(f"Recall    : {best_probabilistic['Recall']:.4f}")
    print(f"F1 Score  : {best_probabilistic['F1_Score']:.4f}")
    print(f"ROC-AUC   : {best_probabilistic['ROC_AUC']:.4f}")


# ============================================================
# OVERALL BEST MODEL
# BASED ON ACCURACY
# ============================================================

overall_best = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\n")
print("=" * 80)
print("OVERALL BEST MODEL (BASED ON ACCURACY)")
print("=" * 80)

print(f"Model     : {overall_best['Model']}")
print(f"Dataset   : {overall_best['Dataset']}")
print(f"Accuracy  : {overall_best['Accuracy']:.4f}")
print(f"Precision : {overall_best['Precision']:.4f}")
print(f"Recall    : {overall_best['Recall']:.4f}")
print(f"F1 Score  : {overall_best['F1_Score']:.4f}")
print(f"ROC-AUC   : {overall_best['ROC_AUC']:.4f}")


# ============================================================
# SAVE FINAL COMPARISON
# ============================================================

output_file = RESULTS_DIR / "final_model_comparison.csv"

comparison.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 80)
print("FINAL COMPARISON SAVED")
print("=" * 80)

print(output_file)

print("\nComparison completed successfully.")