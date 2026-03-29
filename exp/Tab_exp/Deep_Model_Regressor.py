"""
Deep_Model_Regressor.py
=======================
Regression benchmark for flight departure delay prediction.

Goal: Predict the exact departure delay in minutes (DEP_DELAY) using
      7 deep tabular models from the Mambular library.

Models benchmarked: AutoInt, FT-Transformer, MLP, Tangos, TabulaRNN, SAINT, ResNet
Evaluation metrics: MSE (Mean Squared Error), MAE (Mean Absolute Error)
Output: Regressor_results_DEP.csv — comparison table of all models on the test set.
"""

import pandas as pd
import yaml
from mambular.models import AutoIntRegressor, FTTransformerRegressor, MLPRegressor, TangosRegressor, ModernNCARegressor, \
    MambularRegressor, TabulaRNNRegressor, SAINTRegressor, ResNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
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

# Optional: standardize the target variable (currently disabled)
# scaler = StandardScaler()
# df['DEP_DELAY'] = scaler.fit_transform(df[['DEP_DELAY']])

# ============================================================
# 3. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# Split chronologically by day-of-month to prevent data leakage.
#   Train:      days 1–9
#   Validation: days 10–12
#   Test:       days 13+
df_train = df[df['FL_DAY'] <= 9].copy()
df_vaild = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)].copy()
df_test = df[df['FL_DAY'] > 12].copy()

# Separate features (X) and target (y) for each split.
# Target: DEP_DELAY — departure delay in minutes (continuous value).
X_train = df_train[categorical_columns + continuous_columns]
X_vaild = df_vaild[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['DEP_DELAY']
y_vaild = df_vaild['DEP_DELAY']
y_test = df_test['DEP_DELAY']

# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================
# Dictionary of 7 deep tabular regressors from the Mambular library.
# Each model uses a different architecture for learning from tabular data:
#   - AutoInt:       Attention-based automatic feature interaction learning
#   - FTTransformer: Feature Tokenizer + Transformer
#   - MLP:           Standard multi-layer perceptron
#   - Tangos:        Tangos architecture for tabular learning
#   - TabulaRNN:     RNN-based model adapted for tabular data
#   - SAINT:         Self-Attention and Intersample Attention Transformer
#   - ResNet:        ResNet-style skip connections for tabular data
models = {
    "AutoInt": AutoIntRegressor(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerRegressor(d_model=64, n_layers=8),
    "MLP": MLPRegressor(d_model=64),
    "Tangos": TangosRegressor(d_model=64),
    'TabulaRNN': TabulaRNNRegressor(d_model=64),
    'SAINT': SAINTRegressor(d_model=64),
    'ResNet': ResNetRegressor(),
}

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

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
results_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE'])

# ============================================================
# 6. TRAINING LOOP — RANDOMIZED SEARCH + EVALUATION
# ============================================================
if __name__ == '__main__':
    for model_name, model in models.items():
        print(f"RandomizedSearchCV for {model_name}...")

        # Randomized hyperparameter search: tries 10 random combinations
        # with 3-fold cross-validation, optimizing for negative MSE
        # (sklearn convention: higher is better, so neg_MSE is used).
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42
        )

        # Training parameters passed to mambular's .fit() method:
        #   - max_epochs=100:  upper bound on training epochs
        #   - patience=5:      early stopping — stop if no improvement for 5 epochs
        #   - rebuild=True:    rebuild model for each hyperparameter combination
        #   - X_val/y_val:     validation set for early stopping monitoring
        #   - num_workers=0:   prevent hanging on Windows
        fit_params = {"max_epochs": 100, "rebuild": True, "X_val": X_vaild, "y_val": y_vaild,
                      "patience": 5, "dataloader_kwargs": {"num_workers": 0}}

        random_search.fit(X_train, y_train, **fit_params)
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Evaluate the best model on the held-out test set
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)   # Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error

        print(f"{model_name} - MSE: {mse}, MAE: {mae}")

        # Append results and save progressively (so partial results survive crashes)
        result = pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'MAE': [mae]})

        results_df = pd.concat([results_df, result], ignore_index=True)
        results_df.to_csv('Regressor_results_DEP.csv', index=False)

    print('end')
