import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 20260903

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "preprocessed_data"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

MECHANISTIC_FILE = DATA_DIR / "heart_disease_urine_biomarkers_mechanistic_50000 (1).csv"

PROBABILISTIC_FILE = DATA_DIR / "heart_disease_urine_biomarkers_probabilistic_50000.csv"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "age_years",
    "bmi_kg_m2",
    "physical_activity_min_week",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "resting_heart_rate_bpm",
    "urine_creatinine_mg_dL",
    "urine_albumin_mg_L",
    "uacr_mg_g",
    "urine_KIM1_ng_mL",
    "urine_NGAL_ng_mL",
    "urine_total_protein_mg_dL",
    "urine_glucose_mg_dL",
    "urine_specific_gravity",
    "urine_pH",
    "urine_sodium_mmol_L",
    "urine_potassium_mmol_L"
]

BINARY_FEATURES = [
    "family_history_heart_disease",
    "hypertension",
    "diabetes",
    "urine_blood",
    "urine_ketones"
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoking_status",
    "albuminuria_category"
]

TARGET = "heart_disease_risk_binary"


# ============================================================
# VALIDATE ORIGINAL DATASET
# ============================================================

def validate_original_dataset(df, name):

    print("\n" + "=" * 70)
    print(f"VALIDATING ORIGINAL {name.upper()} DATASET")
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Unique patient IDs:", df["patient_id"].nunique())

    print("Missing values:", df.isna().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    required_columns = (
        NUMERIC_FEATURES
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "patient_id",
            "heart_disease_risk_probability",
            "heart_disease_risk_binary",
            "heart_disease_risk"
        ]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {name}: {missing_columns}"
        )

    if len(df) != 50000:
        raise ValueError(
            f"{name} does not contain exactly 50,000 records."
        )

    if df["patient_id"].nunique() != 50000:
        raise ValueError(
            f"{name} contains duplicate patient IDs."
        )

    if df.isna().sum().sum() != 0:
        raise ValueError(
            f"{name} contains missing values."
        )

    print("Required columns: OK")
    print("Dataset validation: PASSED")


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_dataset(df, name):

    print("\n" + "=" * 70)
    print(f"PREPROCESSING {name.upper()} DATASET")
    print("=" * 70)

    data = df.copy()

    # --------------------------------------------------------
    # Preserve patient IDs separately
    # --------------------------------------------------------

    patient_ids = data["patient_id"].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = data[TARGET].astype(int).copy()

    # --------------------------------------------------------
    # Remove ID and target/leakage columns
    # --------------------------------------------------------

    leakage_columns = [
        "patient_id",
        "heart_disease_risk_probability",
        "heart_disease_risk_binary",
        "heart_disease_risk"
    ]

    X = data.drop(
        columns=leakage_columns
    )

    # --------------------------------------------------------
    # Convert categorical variables using one-hot encoding
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=int
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Handle missing values if any appear
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].isna().any():

            if pd.api.types.is_numeric_dtype(
                X[column]
            ):

                X[column] = X[column].fillna(
                    X[column].median()
                )

            else:

                X[column] = X[column].fillna(
                    X[column].mode()[0]
                )

    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=RANDOM_STATE,

        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=y_temp
    )

    # --------------------------------------------------------
    # Standardization
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(
        X_train[NUMERIC_FEATURES]
    )

    X_val_scaled[NUMERIC_FEATURES] = scaler.transform(
        X_val[NUMERIC_FEATURES]
    )

    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(
        X_test[NUMERIC_FEATURES]
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    prefix = name.lower()

    X_train_scaled.to_csv(
        OUTPUT_DIR / f"{prefix}_X_train.csv",
        index=False
    )

    X_val_scaled.to_csv(
        OUTPUT_DIR / f"{prefix}_X_validation.csv",
        index=False
    )

    X_test_scaled.to_csv(
        OUTPUT_DIR / f"{prefix}_X_test.csv",
        index=False
    )

    y_train.to_csv(
        OUTPUT_DIR / f"{prefix}_y_train.csv",
        index=False
    )

    y_val.to_csv(
        OUTPUT_DIR / f"{prefix}_y_validation.csv",
        index=False
    )

    y_test.to_csv(
        OUTPUT_DIR / f"{prefix}_y_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nOriginal dataset:", df.shape)

    print(
        "Training:",
        X_train_scaled.shape
    )

    print(
        "Validation:",
        X_val_scaled.shape
    )

    print(
        "Testing:",
        X_test_scaled.shape
    )

    print(
        "Number of model features:",
        X_train_scaled.shape[1]
    )

    print("\nTraining target:")
    print(y_train.value_counts())

    print("\nValidation target:")
    print(y_val.value_counts())

    print("\nTesting target:")
    print(y_test.value_counts())

    print(
        "\nMissing values after preprocessing:",
        X_train_scaled.isna().sum().sum()
    )

    return {
        "X_train": X_train_scaled,
        "X_validation": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_validation": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "patient_ids": patient_ids
    }


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

mechanistic = pd.read_csv(
    MECHANISTIC_FILE
)

probabilistic = pd.read_csv(
    PROBABILISTIC_FILE
)

print("Datasets loaded successfully.")


# ============================================================
# VALIDATE BOTH
# ============================================================

validate_original_dataset(
    mechanistic,
    "mechanistic"
)

validate_original_dataset(
    probabilistic,
    "probabilistic"
)


# ============================================================
# PREPROCESS BOTH
# ============================================================

mechanistic_processed = preprocess_dataset(
    mechanistic,
    "mechanistic"
)

probabilistic_processed = preprocess_dataset(
    probabilistic,
    "probabilistic"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print(" -", file.name)

print("\nReady for model development.")