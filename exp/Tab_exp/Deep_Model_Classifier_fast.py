"""
Deep_Model_Classifier.py
========================
Binary classification benchmark for flight arrival delay prediction.

Goal: Predict whether a flight will be significantly delayed (|ARR_DELAY| > 15 min)
      using 7 deep tabular models from the Mambular library.

Models benchmarked: MLP, AutoInt, ResNet, FT-Transformer, Tangos, TabulaRNN, SAINT
Evaluation metrics: AUC (Area Under ROC Curve), Accuracy
Output: Classifier_results.csv — comparison table of all models on the test set.
"""

from random import randint, uniform

import numpy as np
import pandas as pd
import yaml
from mambular.models import (
    AutoIntClassifier,
    FTTransformerClassifier,
    MLPClassifier,
    TangosClassifier,
    ModernNCAClassifier,
    TabulaRNNClassifier,
    SAINTClassifier,
    ResNetClassifier,
)
from mambular.models.tabr import TabRClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)
from mambular.models import MambularClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import os
import shap

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ============================================================
# 1. DATA LOADING
# ============================================================
# Load the 2020 flight tabular dataset and drop rows with missing values.
MAIN_PATH = (
    "C:\\Users\\user\\Desktop\\tez\\git\\Delay-data\\Datasets\\Aeolus\\Flight_Tab\\Tab"
)
file_name = "Flight_tab_2024.csv"
df = pd.read_csv(os.path.join(MAIN_PATH, file_name))
df = df.dropna()

# ============================================================
# 2. TARGET VARIABLE — BINARY CLASSIFICATION
# ============================================================
# Convert continuous arrival delay (in minutes) to a binary label:
#   1 = "Delayed"  — absolute delay exceeds 15 minutes (FAA standard threshold)
#   0 = "On-time"  — within ±15 minutes of schedule
df["ARR_DELAY"] = df["ARR_DELAY"].apply(lambda x: 1 if abs(x) > 15.0 else 0)

# ============================================================
# 3. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# Split chronologically by day-of-month to prevent data leakage
# (no future flights leak into training data).
#   Train:      days 1–9
#   Validation: days 10–12
#   Test:       days 13+
df_train = df[df["FL_DAY"] <= 9].copy()
df_vaild = df[(df["FL_DAY"] > 9) & (df["FL_DAY"] <= 12)].copy()
df_test = df[df["FL_DAY"] > 12].copy()

df_list = [df_train, df_vaild, df_test]

# ============================================================
# 4. FEATURE METADATA FROM YAML
# ============================================================
# Load column type definitions (categorical vs continuous) from a config file.
with open(os.path.join(MAIN_PATH, "data_info_2020.yaml"), "r") as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info["columns_info"]["Categorical Features"]
continuous_columns = data_info["columns_info"]["Continuous Features"]

# Remove features that are not useful for prediction:
#   - FLIGHTS: not a meaningful predictor
#   - FL_YEAR, FL_MONTH: constant within a single year/month dataset
continuous_columns = [col for col in continuous_columns if col != "FLIGHTS"]
categorical_columns = [
    col for col in categorical_columns if col != "FL_YEAR" and col != "FL_MONTH"
]

# ============================================================
# 5. PREPROCESSING
# ============================================================
# Encode categorical features as integer labels and
# standardize continuous features (zero mean, unit variance).
for df_1 in df_list:
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    df_1[continuous_columns] = scaler.fit_transform(df_1[continuous_columns])

# Separate features (X) and target (y) for each split
X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train["ARR_DELAY"]
y_vaild = df_vaild["ARR_DELAY"]
y_test = df_test["ARR_DELAY"]

# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================
# Initialize results table
results_df = pd.DataFrame(
    columns=["Model", "Train AUC", "Train ACC", "Test AUC", "Test ACC"]
)

# Dictionary of 7 deep tabular classifiers from the Mambular library.
# Each model uses a different architecture for learning from tabular data:
#   - MLP:           Standard multi-layer perceptron
#   - AutoInt:       Attention-based automatic feature interaction learning
#   - ResNet:        ResNet-style skip connections adapted for tabular data
#   - FTTransformer: Feature Tokenizer + Transformer (state-of-the-art tabular model)
#   - Tangos:        Tangos architecture for tabular learning
#   - TabulaRNN:     RNN-based model for tabular data
#   - SAINT:         Self-Attention and Intersample Attention Transformer
models = {
    "MLP": MLPClassifier(d_model=128),
    "AutoInt": AutoIntClassifier(d_model=128, n_layers=4),
    "ResNet": ResNetClassifier(),
}

# Re-initialize results table (AUC + ACC on test set only)
results_df = pd.DataFrame(columns=["Model", "AUC", "ACC"])

# ============================================================
# 7. HYPERPARAMETER SEARCH SPACE
# ============================================================
# Search over model dimension, depth, and learning rate.
param_dist = {
    "d_model": [64, 128, 256],  # Hidden dimension size
    "n_layers": [2, 6, 10],  # Number of layers / blocks
    "lr": [1e-5, 1e-4, 1e-3],  # Learning rate
}

# ============================================================
# 8. TRAINING LOOP — FAST TWO-PHASE APPROACH
# ============================================================
CSV_FILE = "Classifier_results_fast.csv"

if __name__ == "__main__":
    # Resume from existing results if available (skip already-completed models)
    if os.path.exists(CSV_FILE):
        print(f"Found existing {CSV_FILE}, loading to resume progress...")
        results_df = pd.read_csv(CSV_FILE)
    else:
        results_df = pd.DataFrame(columns=["Model", "AUC", "ACC"])

    # Phase 1 uses 0.1% of training data for hyperparameter search (much faster!)
    X_search = X_train.sample(frac=0.1, random_state=42)
    y_search = y_train.loc[X_search.index]

    fit_params_phase1 = {
        "max_epochs": 100,
        "rebuild": True,
        "val_size": 0.5,
        "patience": 5,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "batch_size": 4096,
        "dataloader_kwargs": {"num_workers": 2},
    }

    fit_params_phase2 = {
        "max_epochs": 100,
        "rebuild": True,
        "X_val": X_vaild,
        "y_val": y_vaild,
        "patience": 3,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "batch_size": 4096,
        "dataloader_kwargs": {"num_workers": 2},
    }

    for model_name, model in models.items():
        # Skip models that are already in the CSV
        if model_name in results_df["Model"].values:
            print(f"Skipping {model_name}, already completed.")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # ----------------------------------------------------------
        # Phase 1: Hyperparameter search on 10% of the training data
        # ----------------------------------------------------------
        print(
            f"Phase 1: Searching hyperparams on 0.1% data slice ({len(X_search)} rows)..."
        )
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="accuracy",
            random_state=42,
        )
        random_search.fit(X_search, y_search, **fit_params_phase1)
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Best CV Score:   {random_search.best_score_:.4f}")

        # ----------------------------------------------------------
        # Phase 2: Retrain the best model on full 100% training data
        # ----------------------------------------------------------
        print(f"Phase 2: Retraining {model_name} on full data ({len(X_train)} rows)...")
        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train, **fit_params_phase2)

        # Evaluate the best model on the held-out test set
        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} - AUC: {auc:.4f}, ACC: {acc:.4f}")

        # Append results and save progressively (so partial results survive crashes)
        result = pd.DataFrame({"Model": [model_name], "AUC": [auc], "ACC": [acc]})
        results_df = pd.concat([results_df, result], ignore_index=True)
        results_df.to_csv(CSV_FILE, index=False)

    print("end")
