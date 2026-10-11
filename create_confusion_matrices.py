import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CREATE CONFUSION MATRIX FIGURES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
COMPARISON_DIR = BASE_DIR / "comparison"

COMPARISON_DIR.mkdir(exist_ok=True)


# ============================================================
# PREDICTION FILES
# ============================================================

prediction_files = [
    ("XGBoost", "Mechanistic", "xgboost_mechanistic_predictions.csv"),
    ("XGBoost", "Probabilistic", "xgboost_probabilistic_predictions.csv"),

    ("LightGBM", "Mechanistic", "lightgbm_mechanistic_predictions.csv"),
    ("LightGBM", "Probabilistic", "lightgbm_probabilistic_predictions.csv"),

    ("Random Forest", "Mechanistic", "random_forest_mechanistic_predictions.csv"),
    ("Random Forest", "Probabilistic", "random_forest_probabilistic_predictions.csv"),

    ("CRF", "Mechanistic", "crf_mechanistic_predictions.csv"),
    ("CRF", "Probabilistic", "crf_probabilistic_predictions.csv"),

    ("1-D CNN", "Mechanistic", "cnn_1d_mechanistic_predictions.csv"),
    ("1-D CNN", "Probabilistic", "cnn_1d_probabilistic_predictions.csv"),
]


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for name in possible_names:

        normalized_name = name.lower().replace(" ", "_")

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


# ============================================================
# CONVERT LABEL TO 0 / 1
# ============================================================

def convert_label(value):

    if pd.isna(value):
        return None

    # Numeric labels
    try:

        numeric_value = float(value)

        if numeric_value == 0:
            return 0

        if numeric_value == 1:
            return 1

    except (ValueError, TypeError):
        pass

    # Text labels
    text = str(value).strip().lower()

    if text in [
        "0",
        "low",
        "negative",
        "no",
        "false",
        "low risk",
        "no risk"
    ]:
        return 0

    if text in [
        "1",
        "high",
        "positive",
        "yes",
        "true",
        "high risk"
    ]:
        return 1

    return None


# ============================================================
# CALCULATE CONFUSION MATRIX WITHOUT SCIKIT-LEARN
# ============================================================

def calculate_confusion_matrix(y_true, y_pred):

    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)

    # True Negative
    tn = np.sum(
        (true_array == 0) &
        (pred_array == 0)
    )

    # False Positive
    fp = np.sum(
        (true_array == 0) &
        (pred_array == 1)
    )

    # False Negative
    fn = np.sum(
        (true_array == 1) &
        (pred_array == 0)
    )

    # True Positive
    tp = np.sum(
        (true_array == 1) &
        (pred_array == 1)
    )

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


# ============================================================
# GENERATE MATRICES
# ============================================================

successful = 0

print()
print("=" * 80)
print("CONFUSION MATRIX GENERATION")
print("=" * 80)


for model, dataset, file_name in prediction_files:

    print()
    print("-" * 80)
    print(f"Model   : {model}")
    print(f"Dataset : {dataset}")
    print(f"File    : {file_name}")

    file_path = RESULTS_DIR / file_name

    if not file_path.exists():

        print("WARNING: Prediction file not found.")
        continue


    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    print(f"Rows loaded: {len(df)}")


    # --------------------------------------------------------
    # FIND ACTUAL LABEL
    # --------------------------------------------------------

    actual_column = find_column(
        df,
        [
            "y_true",
            "true",
            "actual",
            "actual_label",
            "actual_binary",
            "true_label",
            "ground_truth",
            "target",
            "y"
        ]
    )


    # --------------------------------------------------------
    # FIND PREDICTED LABEL
    # --------------------------------------------------------

    predicted_column = find_column(
        df,
        [
            "y_pred",
            "predicted",
            "prediction",
            "predicted_label",
            "predicted_binary",
            "pred_label",
            "prediction_label"
        ]
    )


    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    if actual_column is None or predicted_column is None:

        print("ERROR: Could not identify actual/predicted columns.")

        print("Available columns:")
        print(list(df.columns))

        continue


    print(f"Actual column    : {actual_column}")
    print(f"Predicted column : {predicted_column}")


    # --------------------------------------------------------
    # CONVERT LABELS
    # --------------------------------------------------------

    y_true = df[actual_column].apply(convert_label)
    y_pred = df[predicted_column].apply(convert_label)


    # Remove invalid rows
    valid_rows = (
        y_true.notna() &
        y_pred.notna()
    )

    y_true = y_true[valid_rows].astype(int)
    y_pred = y_pred[valid_rows].astype(int)


    if len(y_true) == 0:

        print("ERROR: No valid prediction rows found.")
        continue


    # --------------------------------------------------------
    # CALCULATE CONFUSION MATRIX
    # --------------------------------------------------------

    cm = calculate_confusion_matrix(
        y_true,
        y_pred
    )


    print()
    print("Confusion Matrix:")
    print(cm)


    # ========================================================
    # SAFE FILE NAMES
    # ========================================================

    safe_model_name = (
        model
        .lower()
        .replace("-", "")
        .replace(" ", "_")
    )

    safe_dataset_name = dataset.lower()


    # ========================================================
    # SAVE CONFUSION MATRIX CSV
    # ========================================================

    csv_name = (
        f"confusion_matrix_"
        f"{safe_model_name}_"
        f"{safe_dataset_name}.csv"
    )

    cm_dataframe = pd.DataFrame(
        cm,
        index=[
            "Actual Low",
            "Actual High"
        ],
        columns=[
            "Predicted Low",
            "Predicted High"
        ]
    )

    cm_dataframe.to_csv(
        COMPARISON_DIR / csv_name
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )


    # Add values inside cells
    for row in range(2):

        for column in range(2):

            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                fontsize=16
            )


    # Axis labels
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Low", "High"],
        yticklabels=["Low", "High"],
        xlabel="Predicted Label",
        ylabel="Actual Label"
    )


    ax.set_title(
        f"{model} - {dataset} Dataset\n"
        f"Confusion Matrix",
        fontsize=14
    )


    # Color bar
    fig.colorbar(
        image,
        ax=ax
    )


    plt.tight_layout()


    # ========================================================
    # SAVE PNG
    # ========================================================

    figure_name = (
        f"confusion_matrix_"
        f"{safe_model_name}_"
        f"{safe_dataset_name}.png"
    )

    figure_path = (
        COMPARISON_DIR /
        figure_name
    )


    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Figure saved: {figure_path}"
    )


    successful += 1


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 80)
print("CONFUSION MATRIX GENERATION COMPLETED")
print("=" * 80)

print(
    f"Successfully generated: "
    f"{successful} confusion matrices"
)

print()
print("Files saved in:")
print(COMPARISON_DIR)

print()
print("Open the 'comparison' folder to view the PNG figures.")

print()
print("=" * 80)