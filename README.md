# Heart Disease Detection Using Urinary Biomarkers

A machine-learning research project investigating heart disease risk classification using urinary biomarkers and associated clinical features. The project compares mechanistic and probabilistic dataset approaches and evaluates multiple machine-learning and deep-learning models.

> **Research disclaimer:** This repository is an experimental research implementation using generated datasets. Its results are not clinical evidence and must not be used for medical diagnosis or treatment.

## Overview

The project explores whether patterns in urinary biomarkers and related features can support experimental heart disease risk classification. It includes data preprocessing, model training, evaluation, and comparative analysis.

Two dataset approaches are included:

- **Mechanistic dataset:** generated using predefined relationships between features and target labels.
- **Probabilistic dataset:** generated using probabilistic relationships between features and target labels.

The validity of any observed performance depends on the data-generation assumptions, label construction, preprocessing, and experimental design.

## Features

- Mechanistic and probabilistic dataset experiments.
- Data preprocessing and preparation.
- Random Forest classification.
- XGBoost and LightGBM experiments.
- One-dimensional CNN experiments.
- CRF-based experiments.
- Weighted Mean Swarm MLP experiment.
- Model comparison and reference-model evaluation scripts.
- Confusion-matrix generation and visual comparisons.
- Saved experiment metrics and predictions where generated.

## Technology Stack

| Component | Technologies |
|---|---|
| Language | Python |
| Data processing | NumPy, pandas, SciPy |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Deep learning | TensorFlow / Keras |
| Evaluation | scikit-learn metrics |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Dataset format | CSV |

## Project Structure

```text
Heart-Disease-Urine/
├── data/
│   ├── heart_disease_urine_biomarkers_mechanistic_50000 (1).csv
│   ├── heart_disease_urine_biomarkers_probabilistic_50000.csv
│   ├── preprocess.py
│   └── train_lightgbm.py
├── comparison/
│   └── Comparison figures and confusion matrices
├── results/
│   └── Saved experiment metrics and predictions
├── models/
│   └── Locally generated trained models
├── preprocessed_data/
│   └── Locally generated processed datasets
├── preprocess.py
├── train_random_forest.py
├── train_random_forest_probabilistic.py
├── train_xgboost.py
├── train_xgboost_probabilistic.py
├── train_lightgbm.py
├── train_lightgbm_probabilistic.py
├── train_cnn.py
├── train_cnn_probabilistic.py
├── train_crf.py
├── train_crf_probabilistic.py
├── train_swarm_mlp.py
├── compare_all_models.py
├── reference_model_comparison.py
├── create_confusion_matrices.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` and `preprocessed_data/` directories contain generated artifacts and may need to be created locally by running the relevant scripts. Not every experiment necessarily uses the same preprocessing or execution sequence.

## Prerequisites

Install:

- Python 3.10 or another version compatible with the selected dependency releases.
- pip.
- Git.
- Visual Studio Code or another Python development environment.

TensorFlow, LightGBM, and other scientific libraries may have platform-specific Python version requirements.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Pranav4656/Heart-Disease-Urine.git
cd Heart-Disease-Urine
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use the environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check the Python version and the specific package's platform requirements.

## Running Experiments

Run commands from the repository root. Start with one experiment and verify its input and output paths before running the next.

### 1. Data preprocessing

```powershell
python preprocess.py
```

Inspect the script to confirm the expected input files, output directories, and preprocessing operations.

### 2. Random Forest

```powershell
python train_random_forest.py
python train_random_forest_probabilistic.py
```

### 3. XGBoost

```powershell
python train_xgboost.py
python train_xgboost_probabilistic.py
```

### 4. LightGBM

```powershell
python train_lightgbm.py
python train_lightgbm_probabilistic.py
```

### 5. CNN

```powershell
python train_cnn.py
python train_cnn_probabilistic.py
```

### 6. CRF

```powershell
python train_crf.py
python train_crf_probabilistic.py
```

### 7. Weighted Mean Swarm MLP

```powershell
python train_swarm_mlp.py
```

### 8. Model comparison and confusion matrices

```powershell
python compare_all_models.py
python reference_model_comparison.py
python create_confusion_matrices.py
```

These are the repository's available script entry points, not a guarantee that every script runs independently or in the order listed. Inspect each script for dependencies on generated files, required model artifacts, command-line arguments, and output paths. Follow the execution sequence defined by the implementation.

## Evaluation

Use the evaluation scripts and generated artifacts to compare model performance across the two dataset approaches.

When interpreting results, examine:

- Accuracy, precision, recall, and F1-score where reported.
- Confusion matrices and class-wise errors.
- Differences between mechanistic and probabilistic experiments.
- Potential data leakage and dependence on synthetic label-generation rules.
- Reproducibility across seeds and train/validation/test partitions.

Do not compare results from different experiments without checking that they use compatible data splits, target definitions, and evaluation procedures.

## Reproducibility

For repeatable experiments:

1. Record the Python and library versions.
2. Record random seeds and dataset-generation settings.
3. Keep preprocessing and evaluation procedures consistent.
4. Record model configurations and dataset partitions.
5. Preserve experiment metrics separately from large model binaries.
6. Document any changes to labels, features, or data-generation assumptions.

## Limitations

- Generated datasets cannot establish clinical validity.
- High performance on synthetic data does not demonstrate performance on real patients.
- The experiments require independent evaluation using suitable real-world datasets before clinical conclusions can be considered.
- Urinary biomarkers should not be presented as a validated standalone diagnostic method on the basis of these experiments.
- CRF and neural-network approaches should be justified by the structure of the data and the research question, not only by their inclusion in a model comparison.

## Author

Pranav

## License

No license has been specified. Unless a license is added, standard copyright restrictions apply.
