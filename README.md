# Heart Disease Detection Using Urinary Biomarkers

A machine learning system for heart disease classification using urinary biomarkers and associated clinical features. The project implements and compares multiple machine learning and deep learning models using mechanistic and probabilistic datasets.

## Features

- Mechanistic and probabilistic dataset processing
- Data preprocessing and feature scaling
- Random Forest
- XGBoost
- LightGBM
- 1D Convolutional Neural Network (CNN)
- Conditional Random Field (CRF)
- Weighted Mean Swarm MLP
- Model performance comparison
- Confusion matrix generation
- Prediction and evaluation result generation

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | NumPy, pandas |
| Machine Learning | scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Deep Learning | TensorFlow / Keras |
| CRF | sklearn-crfsuite |
| Visualization | Matplotlib |
| Model Storage | Joblib |
| Dataset Format | CSV |

## Project Structure

```text
Heart-Disease-Urine/
├── data/
│   ├── heart_disease_urine_biomarkers_mechanistic_50000 (1).csv
│   ├── heart_disease_urine_biomarkers_probabilistic_50000.csv
│   ├── preprocess.py
│   └── train_lightgbm.py
├── comparison/
│   ├── Model comparison plots
│   └── Confusion matrices
├── models/
│   └── Trained model files
├── preprocessed_data/
│   └── Training, validation and test datasets
├── results/
│   └── Model metrics and predictions
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
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pranav4656/Heart-Disease-Urine.git
cd Heart-Disease-Urine
```

### 2. Create a Virtual Environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Execution

Run all commands from the project root directory.

### 1. Preprocessing

```bash
python preprocess.py
```

This generates the processed datasets used by the model training scripts.

### 2. Train Models

**Random Forest**
```bash
python train_random_forest.py
python train_random_forest_probabilistic.py
```

**XGBoost**
```bash
python train_xgboost.py
python train_xgboost_probabilistic.py
```

**LightGBM**
```bash
python train_lightgbm.py
python train_lightgbm_probabilistic.py
```

**CNN**
```bash
python train_cnn.py
python train_cnn_probabilistic.py
```

**CRF**
```bash
python train_crf.py
python train_crf_probabilistic.py
```

**Weighted Mean Swarm MLP**
```bash
python train_swarm_mlp.py
```

### 3. Compare Models

```bash
python compare_all_models.py
python reference_model_comparison.py
```

### 4. Generate Confusion Matrices

```bash
python create_confusion_matrices.py
```

## Outputs

- Trained models are saved in `models/`.
- Processed datasets are saved in `preprocessed_data/`.
- Evaluation metrics and predictions are saved in `results/`.
- Model comparison plots and confusion matrices are saved in `comparison/`.

## Author

Pranav
