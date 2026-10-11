# Heart Disease Detection Using Urinary Biomarkers

A machine learning system for heart disease classification using urinary biomarkers and associated clinical features.

## Problem

Heart disease classification involves analysing multiple clinical features. This project explores the use of urinary biomarkers alongside associated clinical features for machine learning-based classification.

## Solution

The system processes mechanistic and probabilistic datasets, trains multiple machine learning and deep learning models, and compares their classification performance.

## Key Workflow

1. **Dataset Input** — Mechanistic and probabilistic urinary biomarker datasets.
2. **Preprocessing** — Feature preparation, scaling, and dataset splitting.
3. **Model Training** — Train and evaluate Random Forest, XGBoost, LightGBM, CNN, CRF, and Weighted Mean Swarm MLP.
4. **Performance Evaluation** — Generate evaluation metrics and predictions.
5. **Model Comparison** — Compare results and generate confusion matrices and visualizations.

## Architecture

**Input Datasets → Preprocessing → Model Training → Evaluation → Model Comparison**

## Technology Stack

- Python
- NumPy and pandas
- scikit-learn
- XGBoost and LightGBM
- TensorFlow / Keras
- sklearn-crfsuite
- Matplotlib and Joblib

## Project Structure

- `data/` — Input datasets
- `preprocessed_data/` — Processed datasets
- `models/` — Trained models
- `results/` — Evaluation metrics and predictions
- `comparison/` — Comparison plots and confusion matrices

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Preprocess the datasets:

```bash
python preprocess.py
```

Run the model training scripts, followed by model comparison and confusion matrix generation.

