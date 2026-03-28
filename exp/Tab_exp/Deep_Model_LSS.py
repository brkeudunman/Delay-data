"""
Deep_Model_LSS.py
=================
Distributional (Location-Scale-Shape) benchmark for flight departure delay prediction.

Goal: Predict the full probability distribution of DEP_DELAY (not just a point estimate).
      Each model outputs a mean (location) and standard deviation (scale), enabling
      uncertainty quantification — i.e., "how confident is the model in its prediction?"

Models benchmarked: AutoInt, FT-Transformer, MLP, Tangos, TabulaRNN, SAINT, ResNet
Evaluation metrics:
  - NLL  (Negative Log-Likelihood): measures how well the predicted distribution fits the data
  - CRPS (Continuous Ranked Probability Score): measures calibration of predicted distributions
Output: LSS_results_DEP.csv — comparison table of all models on the test set.
"""

from random import randint, uniform

import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntLSS, FTTransformerLSS, MLPLSS, TangosLSS, ModernNCALSS, MambularLSS, TabulaRNNLSS, \
    SAINTLSS, ResNetLSS
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DATA LOADING
# ============================================================
# Load the flight delay dataset (arrival delay version).
file_name = 'D:\\project\\Delay_data\\Datasets\\arr_delay_data.csv'
df = pd.read_csv(file_name)

# ============================================================
# 2. FEATURE METADATA FROM YAML
# ============================================================
# Load column type definitions (categorical vs continuous) from a config file.
with open('D:\\project\\Delay_data\\Datasets\\arr_delay_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

# Remove features that are not useful for prediction:
#   - FLIGHTS: not a meaningful predictor
#   - FL_YEAR, FL_MONTH: constant within a single dataset slice
continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

# Standardize the target variable (DEP_DELAY) to have zero mean and unit variance.
# This helps LSS models learn the distribution parameters more stably.
scaler = StandardScaler()
df['DEP_DELAY'] = scaler.fit_transform(df[['DEP_DELAY']])

# ============================================================
# 3. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# Split chronologically by day-of-month to prevent data leakage.
#   Train:      days 1–9
#   Validation: days 10–12
#   Test:       days 13+
df_train = df[df['FL_DAY'] <= 9]
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)]
df_test = df[df['FL_DAY'] > 12]

# Separate features (X) and target (y) for each split.
# Target: DEP_DELAY — standardized departure delay.
X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['DEP_DELAY']
y_vaild = df_vaild['DEP_DELAY']
y_test = df_test['DEP_DELAY']

# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================
# Dictionary of 7 deep tabular LSS (distributional) models from Mambular.
# Unlike classifiers/regressors, these output distribution parameters
# (mean + std), not just a single prediction.
models = {
    "AutoInt": AutoIntLSS(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerLSS(d_model=64, n_layers=8),
    "MLP": MLPLSS(d_model=64),
    "Tangos": TangosLSS(d_model=64),
    'TabulaRNN': TabulaRNNLSS(d_model=64),
    'SAINT': SAINTLSS(d_model=64),
    'ResNet': ResNetLSS(),
}

# ============================================================
# 5. HYPERPARAMETER SEARCH SPACE
# ============================================================
# Search over model dimension, depth, and learning rate.
param_dist = {
    'd_model': [64, 128, 256],   # Hidden dimension size
    'n_layers': [2, 6, 10],      # Number of layers / blocks
    'lr': [1e-5, 1e-4, 1e-3]    # Learning rate
}

# Initialize results table
results_df = pd.DataFrame(columns=['Model', 'NLL', 'CRPS'])

# ============================================================
# 6. TRAINING LOOP — RANDOMIZED SEARCH + EVALUATION
# ============================================================
for model_name, model in models.items():
    # Randomized hyperparameter search: tries 10 random combinations
    # with 3-fold cross-validation, optimizing for negative MSE.
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42
    )

    # Training parameters:
    #   - max_epochs=100:  upper bound on training epochs
    #   - patience=5:      early stopping — stop if no improvement for 5 epochs
    #   - rebuild=True:    rebuild model for each hyperparameter combination
    #   - X_val/y_val:     validation set for early stopping monitoring
    fit_params = {"max_epochs": 100, "rebuild": True, "X_val": X_vaild, "y_val": y_vaild, "patience": 5}
    random_search.fit(X_train, y_train, **fit_params)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    # --------------------------------------------------------
    # LSS models return TWO outputs per sample:
    #   y_pred[:, 0] = predicted mean   (location parameter)
    #   y_pred[:, 1] = predicted std    (scale parameter)
    # --------------------------------------------------------
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_mean = y_pred[:, 0]  # Predicted mean delay
    y_pred_std = y_pred[:, 1]   # Predicted uncertainty (std deviation)

    # --------------------------------------------------------
    # METRIC 1: Negative Log-Likelihood (NLL)
    # --------------------------------------------------------
    # Assumes a Gaussian distribution: N(y_pred_mean, y_pred_std^2).
    # Lower NLL = better fit of predicted distribution to actual data.
    # Formula: NLL = 0.5 * mean[ (y - μ)²/σ² + log(σ²) + log(2π) ]
    delta = y_test - y_pred_mean
    sigma_sq = y_pred_std ** 2
    nll = 0.5 * np.mean(delta ** 2 / sigma_sq + np.log(sigma_sq) + np.log(2 * np.pi))
    print(f"NLL (calculated): {nll}")

    # --------------------------------------------------------
    # METRIC 2: Continuous Ranked Probability Score (CRPS)
    # --------------------------------------------------------
    # CRPS measures the quality of probabilistic forecasts.
    # It generalizes MAE to distributional predictions.
    # Lower CRPS = better calibrated uncertainty estimates.
    # Closed-form for Gaussian: CRPS = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    #   where z = (y - μ) / σ, φ = PDF, Φ = CDF of standard normal.
    from scipy.special import erf
    z = delta / y_pred_std
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)   # Standard normal PDF
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))               # Standard normal CDF
    crps_values = y_pred_std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    crps = crps_values.mean()
    print(f"CRPS: {crps}")

    print(f"{model_name} - NLL: {nll}, CRPS: {crps}")

    # Append results and save progressively (so partial results survive crashes)
    result = pd.DataFrame({'Model': [model_name], 'NLL': [nll], 'CRPS': [crps]})
    results_df = pd.concat([results_df, result], ignore_index=True)
    results_df.to_csv('LSS_results_DEP.csv', index=False)

print('end')
