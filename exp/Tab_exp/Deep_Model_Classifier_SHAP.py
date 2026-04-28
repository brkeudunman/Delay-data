"""
Deep_Model_Classifier_SHAP.py
========================
Binary classification benchmark for flight arrival delay prediction using SHAP
Feature Selection and HUIM pattern generation.

Goal: Predict whether a flight will be significantly delayed (|ARR_DELAY| > 15 min)
      using deep tabular models (MLP, AutoInt, ResNet) with a customized feature set.
"""

import gc
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler

from mambular.models import AutoIntClassifier, MLPClassifier, ResNetClassifier

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAIN_PATH = r"C:\Users\user\Desktop\tez\git\Delay-data\Datasets\Aeolus\Flight_Tab\Tab"
EXP_PATH = r"C:\Users\user\Desktop\tez\git\Delay-data\exp\Tab_exp"
SHAP_PATH = os.path.join(EXP_PATH, "SHAP_HUIM")
CSV_FILE = os.path.join(EXP_PATH, "Classifier_results_SHAP_HUIM.csv")

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "d_model": [64, 128, 256],
    "n_layers": [2, 6, 10],
    "lr": [1e-5, 1e-4, 1e-3],
}

# Fraction of data used for each phase (set to 1.0 for full training)
SEARCH_FRAC = 0.1
TRAIN_FRAC = 0.1
VAL_FRAC = 0.1

BIN_LABELS_MAP = {"VeryLow": 0, "Low": 1, "Mid": 2, "High": 3, "VeryHigh": 4}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load the flight dataset and apply the binary delay label."""
    print("Loading datasets...")
    df = pd.read_csv(os.path.join(MAIN_PATH, "Flight_tab_2020.csv"))
    df = df.dropna()
    df["ARR_DELAY"] = (df["ARR_DELAY"].abs() > 15.0).astype(int)

    with open(os.path.join(MAIN_PATH, "data_info_2020.yaml"), "r") as f:
        data_info = yaml.load(f, Loader=yaml.FullLoader)

    cat_cols = [
        c
        for c in data_info["columns_info"]["Categorical Features"]
        if c not in ("FL_YEAR", "FL_MONTH")
    ]
    cont_cols = [
        c for c in data_info["columns_info"]["Continuous Features"] if c != "FLIGHTS"
    ]

    # Chronological split
    df_train = df[df["FL_DAY"] <= 9].copy()
    df_val = df[(df["FL_DAY"] > 9) & (df["FL_DAY"] <= 12)].copy()
    df_test = df[df["FL_DAY"] > 12].copy()

    return df_train, df_val, df_test, cat_cols, cont_cols


# ---------------------------------------------------------------------------
# 2. SHAP / HUIM config loading
# ---------------------------------------------------------------------------
def load_shap_huim_config():
    """Read top SHAP features and HUIM patterns from disk."""
    print("Extracting SHAP logic and HUIM patterns...")
    shap_df = pd.read_csv(os.path.join(SHAP_PATH, "shap_feature_importance.csv"))
    top_shap_features = shap_df.head(10)["Feature"].tolist()

    drivers_df = pd.read_csv(os.path.join(SHAP_PATH, "huim_delay_drivers.csv"))
    protectors_df = pd.read_csv(os.path.join(SHAP_PATH, "huim_delay_protectors.csv"))
    top_patterns = (
        drivers_df.head(20)["Pattern"].tolist()
        + protectors_df.head(20)["Pattern"].tolist()
    )

    return top_shap_features, top_patterns


# ---------------------------------------------------------------------------
# 3. HUIM feature engineering
# ---------------------------------------------------------------------------
def add_huim_features(split_df, top_patterns, cat_cols, cont_cols, discretizer):
    """Append binary HUIM pattern columns to *split_df* in-place."""
    cont_binned = discretizer.transform(split_df[cont_cols])

    for pat in top_patterns:
        pat_matches = np.ones(len(split_df), dtype=bool)
        for cond in (c.strip() for c in pat.split(" + ")):
            col, val = cond.split("=")
            if col in cat_cols:
                try:
                    val_float = float(val)
                    pat_matches &= (
                        pd.to_numeric(split_df[col], errors="coerce") == val_float
                    ).fillna(False)
                except ValueError:
                    pat_matches &= split_df[col].astype(str) == val
            elif col in cont_cols:
                bin_idx = BIN_LABELS_MAP[val]
                col_idx = cont_cols.index(col)
                pat_matches &= cont_binned[:, col_idx] == bin_idx

        pat_name = pat.replace(" + ", "_").replace("=", "_")
        split_df["HUIM_" + pat_name] = pat_matches.astype(int)


# ---------------------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------------------
def preprocess_splits(df_train, df_val, df_test, cat_cols, cont_cols, top_patterns):
    """Fit discretizer, add HUIM features, encode & scale all splits."""
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    discretizer.fit(df_train[cont_cols])

    for split in (df_train, df_val, df_test):
        add_huim_features(split, top_patterns, cat_cols, cont_cols, discretizer)

    return discretizer


def build_datasets(
    df_train, df_val, df_test, cat_cols, cont_cols, top_shap_features, top_patterns
):
    """Derive final feature columns, encode categoricals, scale continuous features."""
    kept_cat_cols = [c for c in cat_cols if c in top_shap_features]
    kept_cont_cols = [c for c in cont_cols if c in top_shap_features]
    huim_cols = [
        "HUIM_" + pat.replace(" + ", "_").replace("=", "_") for pat in top_patterns
    ]
    final_cat_cols = kept_cat_cols + huim_cols

    print(
        f"Features Generated: {len(kept_cont_cols)} Continuous SHAP, "
        f"{len(kept_cat_cols)} Categorical SHAP, {len(huim_cols)} Categorical HUIM"
    )

    # Fit scaler on train only; transform all splits
    scaler = StandardScaler()
    scaler.fit(df_train[kept_cont_cols]) if kept_cont_cols else None

    for split in (df_train, df_val, df_test):
        for col in final_cat_cols:
            le = LabelEncoder()
            split[col] = le.fit_transform(split[col].astype(str))
        if kept_cont_cols:
            split[kept_cont_cols] = scaler.transform(split[kept_cont_cols])

    final_features = final_cat_cols + kept_cont_cols

    X_train, y_train = df_train[final_features], df_train["ARR_DELAY"]
    X_val, y_val = df_val[final_features], df_val["ARR_DELAY"]
    X_test, y_test = df_test[final_features], df_test["ARR_DELAY"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def make_fit_params(X_val, y_val):
    return {
        "max_epochs": 100,
        "rebuild": True,
        "X_val": X_val,
        "y_val": y_val,
        "patience": 5,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "batch_size": 2048,
        "dataloader_kwargs": {"num_workers": 0, "pin_memory": False},
    }


def run_training_loop(X_train, y_train, X_val, y_val, X_test, y_test, results_df):
    """Run Phase-1 hyperparameter search + Phase-2 full retraining for every model."""

    # Sub-sample for fast iterations
    X_search = X_train.sample(frac=SEARCH_FRAC, random_state=42)
    y_search = y_train.loc[X_search.index]

    X_val_small = X_val.sample(frac=VAL_FRAC, random_state=42)
    y_val_small = y_val.loc[X_val_small.index]

    X_train_phase2 = X_train.sample(frac=TRAIN_FRAC, random_state=42)
    y_train_phase2 = y_train.loc[X_train_phase2.index]

    fit_params = make_fit_params(X_val_small, y_val_small)

    models = {
        "MLP": MLPClassifier(d_model=128),
        "AutoInt": AutoIntClassifier(d_model=128, n_layers=4),
        "ResNet": ResNetClassifier(),
    }

    for model_name, model in models.items():
        if model_name in results_df["Model"].values:
            print(f"Skipping {model_name}, already completed.")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name} (SHAP + HUIM features)")
        print(f"{'='*60}")

        # --- Phase 1: hyperparameter search ---
        print(f"Phase 1: Searching hyperparams on {len(X_search)} rows...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAM_DIST,
            n_iter=10,
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=1,  # Sequential to avoid Lightning/CUDA fork issues
        )
        search.fit(X_search, y_search, **fit_params)
        print(f"Best Parameters: {search.best_params_}")

        # --- Phase 2: retrain best model on full training data ---
        print(f"Phase 2: Retraining on {len(X_train_phase2)} rows...")
        best_model = search.best_estimator_
        best_model.fit(X_train_phase2, y_train_phase2, **fit_params)

        # --- Evaluation ---
        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} (SHAP-HUIM) — AUC: {auc:.4f}, ACC: {acc:.4f}")

        result = pd.DataFrame({"Model": [model_name], "AUC": [auc], "ACC": [acc]})
        results_df = pd.concat([results_df, result], ignore_index=True)
        results_df.to_csv(CSV_FILE, index=False)

        # Free GPU memory between iterations to prevent CUDA memory corruption
        torch.cuda.empty_cache()
        gc.collect()

    return results_df


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
def main():
    # Load raw data
    df_train, df_val, df_test, cat_cols, cont_cols = load_data()

    # Load SHAP / HUIM metadata
    top_shap_features, top_patterns = load_shap_huim_config()

    # Add HUIM binary features to every split
    preprocess_splits(df_train, df_val, df_test, cat_cols, cont_cols, top_patterns)

    # Encode, scale and build final X/y arrays
    X_train, y_train, X_val, y_val, X_test, y_test = build_datasets(
        df_train, df_val, df_test, cat_cols, cont_cols, top_shap_features, top_patterns
    )

    # Resume from existing results if available
    if os.path.exists(CSV_FILE):
        print(f"Found existing {CSV_FILE}, resuming...")
        results_df = pd.read_csv(CSV_FILE)
    else:
        results_df = pd.DataFrame(columns=["Model", "AUC", "ACC"])

    # Train all models
    results_df = run_training_loop(
        X_train, y_train, X_val, y_val, X_test, y_test, results_df
    )

    # Final comparison
    print("\n" + "=" * 60)
    print("  COMPARISON WITH BASELINE")
    print("=" * 60)
    baseline_csv = os.path.join(EXP_PATH, "Classifier_results_fast_2020.csv")
    if os.path.exists(baseline_csv):
        print("\nBASELINE (All features):")
        print(pd.read_csv(baseline_csv).to_string(index=False))

    print("\nSHAP + HUIM (Top 10 SHAP + 40 patterns):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
