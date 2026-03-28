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
from mambular.models import AutoIntClassifier, FTTransformerClassifier, MLPClassifier, TangosClassifier, \
    ModernNCAClassifier, TabulaRNNClassifier, SAINTClassifier, ResNetClassifier
from mambular.models.tabr import TabRClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
from mambular.models import MambularClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ============================================================
# 1. DATA LOADING
# ============================================================
# Load the 2020 flight tabular dataset and drop rows with missing values.
MAIN_PATH = 'C:\\Users\\user\\Desktop\\tez\\git\\Delay-data\\Datasets\\Aeolus\\Flight_Tab\\Tab'
file_name = 'Flight_tab_2020.csv'
df = pd.read_csv(os.path.join(MAIN_PATH, file_name))
df = df.dropna()

# ============================================================
# 2. TARGET VARIABLE — BINARY CLASSIFICATION
# ============================================================
# Convert continuous arrival delay (in minutes) to a binary label:
#   1 = "Delayed"  — absolute delay exceeds 15 minutes (FAA standard threshold)
#   0 = "On-time"  — within ±15 minutes of schedule
df['ARR_DELAY'] = df['ARR_DELAY'].apply(lambda x: 1 if abs(x) > 15.0 else 0)

# ============================================================
# 3. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# Split chronologically by day-of-month to prevent data leakage
# (no future flights leak into training data).
#   Train:      days 1–9
#   Validation: days 10–12
#   Test:       days 13+
df_train = df[df['FL_DAY'] <= 9]
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)]
df_test = df[df['FL_DAY'] > 12]

df_list = [df_train, df_vaild, df_test]

# ============================================================
# 4. FEATURE METADATA FROM YAML
# ============================================================
# Load column type definitions (categorical vs continuous) from a config file.
with open(os.path.join(MAIN_PATH, 'data_info_2020.yaml'), 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

# Remove features that are not useful for prediction:
#   - FLIGHTS: not a meaningful predictor
#   - FL_YEAR, FL_MONTH: constant within a single year/month dataset
continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

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

y_train = df_train['ARR_DELAY']
y_vaild = df_vaild['ARR_DELAY']
y_test = df_test['ARR_DELAY']

# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================
# Initialize results table
results_df = pd.DataFrame(columns=['Model', 'Train AUC', 'Train ACC', 'Test AUC', 'Test ACC'])

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
    'MLP': MLPClassifier(d_model=128),
    'AutoInt': AutoIntClassifier(d_model=128, n_layers=4),
    'ResNet': ResNetClassifier(),
    'FTTransformer': FTTransformerClassifier(d_model=128, n_layers=4),
    'Tangos': TangosClassifier(d_model=128),
    'TabulaRNN': TabulaRNNClassifier(d_model=128),
    'SAINT': SAINTClassifier(d_model=128),
}

# Re-initialize results table (AUC + ACC on test set only)
results_df = pd.DataFrame(columns=['Model', 'AUC', 'ACC'])

# ============================================================
# 7. HYPERPARAMETER SEARCH SPACE
# ============================================================
# Search over model dimension, depth, and learning rate.
param_dist = {
    'd_model': [64, 128, 256],   # Hidden dimension size
    'n_layers': [2, 6, 10],      # Number of layers / blocks
    'lr': [1e-5, 1e-4, 1e-3]    # Learning rate
}

# ============================================================
# 8. TRAINING LOOP — RANDOMIZED SEARCH + EVALUATION
# ============================================================
for model_name, model in models.items():
    print(f"RandomizedSearchCV for {model_name}...")

    # Randomized hyperparameter search: tries 10 random combinations
    # with 3-fold cross-validation, optimizing for accuracy.
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        random_state=42
    )

    # Training parameters passed to mambular's .fit() method:
    #   - max_epochs=100:  upper bound on training epochs
    #   - patience=5:      early stopping — stop if no improvement for 5 epochs
    #   - rebuild=True:    rebuild model for each hyperparameter combination
    #   - X_val/y_val:     validation set for early stopping monitoring
    #   - accelerator=gpu: use GPU for faster training
    fit_params = {"max_epochs": 100, "rebuild": True, "X_val": X_vaild, "y_val": y_vaild,
                  "patience": 5,
                  "trainer_kwargs": {"accelerator": "gpu", "devices": 1}}
    random_search.fit(X_train, y_train, **fit_params)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    # Evaluate the best model on the held-out test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)   # Area Under ROC Curve
    acc = accuracy_score(y_test, y_pred)   # Classification accuracy
    print(f"{model_name} - AUC: {auc}, ACC: {acc}")

    # Append results and save progressively (so partial results survive crashes)
    result = pd.DataFrame({'Model': [model_name], 'AUC': [auc], 'ACC': [acc]})
    results_df = pd.concat([results_df, result], ignore_index=True)
    results_df.to_csv('Classifier_results.csv', index=False)

print('end')
