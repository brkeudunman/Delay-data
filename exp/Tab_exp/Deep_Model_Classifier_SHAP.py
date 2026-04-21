"""
Deep_Model_Classifier_SHAP.py
========================
Binary classification benchmark for flight arrival delay prediction using SHAP
Feature Selection and HUIM pattern generation.

Goal: Predict whether a flight will be significantly delayed (|ARR_DELAY| > 15 min)
      using deep tabular models (MLP, AutoInt, ResNet) with a customized feature set.
"""

from random import randint, uniform
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import warnings

from mambular.models import (
    AutoIntClassifier,
    MLPClassifier,
    ResNetClassifier,
)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ============================================================
# 1. CONFIGURATION & DATA LOADING
# ============================================================
MAIN_PATH = (
    "C:\\Users\\user\\Desktop\\tez\\git\\Delay-data\\Datasets\\Aeolus\\Flight_Tab\\Tab"
)
EXP_PATH = "C:\\Users\\user\\Desktop\\tez\\git\\Delay-data\\exp\\Tab_exp"
SHAP_PATH = os.path.join(EXP_PATH, "SHAP_HUIM")

print("Loading datasets...")
df = pd.read_csv(os.path.join(MAIN_PATH, "Flight_tab_2020.csv"))
df = df.dropna()
df["ARR_DELAY"] = df["ARR_DELAY"].apply(lambda x: 1 if abs(x) > 15.0 else 0)

with open(os.path.join(MAIN_PATH, "data_info_2020.yaml"), "r") as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

cat_cols = [
    c
    for c in data_info["columns_info"]["Categorical Features"]
    if c not in ("FL_YEAR", "FL_MONTH")
]
cont_cols = [
    c for c in data_info["columns_info"]["Continuous Features"] if c != "FLIGHTS"
]

# Split chronologically
df_train = df[df["FL_DAY"] <= 9].copy()
df_vaild = df[(df["FL_DAY"] > 9) & (df["FL_DAY"] <= 12)].copy()
df_test = df[df["FL_DAY"] > 12].copy()

# ============================================================
# 2. SHAP SELECTION & HUIM FEATURE ENGINEERING
# ============================================================
print("Extracting SHAP logic and HUIM patterns...")
shap_df = pd.read_csv(os.path.join(SHAP_PATH, "shap_feature_importance.csv"))
top_shap_features = shap_df.head(10)["Feature"].tolist()

drivers_df = pd.read_csv(os.path.join(SHAP_PATH, "huim_delay_drivers.csv"))
protectors_df = pd.read_csv(os.path.join(SHAP_PATH, "huim_delay_protectors.csv"))
top_patterns = (
    drivers_df.head(20)["Pattern"].tolist() + protectors_df.head(20)["Pattern"].tolist()
)

bin_labels_map = {"VeryLow": 0, "Low": 1, "Mid": 2, "High": 3, "VeryHigh": 4}

discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
discretizer.fit(df_train[cont_cols])


def add_huim_features(split_df):
    cont_binned = discretizer.transform(split_df[cont_cols])

    for pat in top_patterns:
        pat_matches = np.ones(len(split_df), dtype=bool)
        conditions = [c.strip() for c in pat.split(" + ")]
        for cond in conditions:
            col, val = cond.split("=")
            if col in cat_cols:
                try:
                    val_float = float(val)
                    # Convert split_df to float just for comparison if possible
                    pat_matches &= (
                        pd.to_numeric(split_df[col], errors="coerce") == val_float
                    ).fillna(False)
                except ValueError:
                    pat_matches &= split_df[col].astype(str) == val
            elif col in cont_cols:
                bin_idx = bin_labels_map[val]
                col_idx = cont_cols.index(col)
                pat_matches &= cont_binned[:, col_idx] == bin_idx

        # Boolean to int category (0 or 1)
        pat_name = pat.replace(" + ", "_").replace("=", "_")
        split_df["HUIM_" + pat_name] = pat_matches.astype(int)


add_huim_features(df_train)
add_huim_features(df_vaild)
add_huim_features(df_test)

kept_cat_cols = [c for c in cat_cols if c in top_shap_features]
kept_cont_cols = [c for c in cont_cols if c in top_shap_features]
huim_cols = [
    "HUIM_" + pat.replace(" + ", "_").replace("=", "_") for pat in top_patterns
]

final_cat_cols = kept_cat_cols + huim_cols

print(
    f"Features Generated: {len(kept_cont_cols)} Continuous SHAP, {len(kept_cat_cols)} Categorical SHAP, {len(huim_cols)} Categorical HUIM"
)

# Label Encoding & Standardization
for df_1 in [df_train, df_vaild, df_test]:
    for col in final_cat_cols:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col].astype(str))

    scaler = StandardScaler()
    if kept_cont_cols:
        df_1[kept_cont_cols] = scaler.fit_transform(df_1[kept_cont_cols])

final_features = final_cat_cols + kept_cont_cols

X_train = df_train[final_features]
X_vaild = df_vaild[final_features]
X_test = df_test[final_features]

y_train = df_train["ARR_DELAY"]
y_vaild = df_vaild["ARR_DELAY"]
y_test = df_test["ARR_DELAY"]

# ============================================================
# 3. TRAINING LOOP — FAST TWO-PHASE APPROACH
# ============================================================
models = {
    "MLP": MLPClassifier(d_model=128),
    "AutoInt": AutoIntClassifier(d_model=128, n_layers=4),
    "ResNet": ResNetClassifier(),
}

param_dist = {
    "d_model": [64, 128, 256],
    "n_layers": [2, 6, 10],
    "lr": [1e-5, 1e-4, 1e-3],
}

CSV_FILE = os.path.join(EXP_PATH, "Classifier_results_SHAP_HUIM.csv")

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        print(f"Found existing {CSV_FILE}, loading to resume progress...")
        results_df = pd.read_csv(CSV_FILE)
    else:
        results_df = pd.DataFrame(columns=["Model", "AUC", "ACC"])

    X_search = X_train.sample(frac=0.1, random_state=42)
    y_search = y_train.loc[X_search.index]

    fit_params = {
        "max_epochs": 100,
        "rebuild": True,
        "X_val": X_vaild,
        "y_val": y_vaild,
        "patience": 5,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "batch_size": 4096,
        "dataloader_kwargs": {"num_workers": 2},
    }

    for model_name, model in models.items():
        if model_name in results_df["Model"].values:
            print(f"Skipping {model_name}, already completed.")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name} (SHAP + HUIM features)")
        print(f"{'='*60}")

        print(f"Phase 1: Searching hyperparams on ({len(X_search)} rows)...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="accuracy",
            random_state=42,
        )
        random_search.fit(X_search, y_search, **fit_params)
        print(f"Best Parameters: {random_search.best_params_}")

        print(f"Phase 2: Retraining on full data ({len(X_train)} rows)...")
        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train, **fit_params)

        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} (SHAP-HUIM) - AUC: {auc:.4f}, ACC: {acc:.4f}")

        result = pd.DataFrame({"Model": [model_name], "AUC": [auc], "ACC": [acc]})
        results_df = pd.concat([results_df, result], ignore_index=True)
        results_df.to_csv(CSV_FILE, index=False)

    print("\n" + "=" * 60)
    print("  COMPARISON WITH BASELINE FAST MODEL")
    print("=" * 60)
    baseline_csv = os.path.join(EXP_PATH, "Classifier_results_fast_2020.csv")
    if os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        print("\nBASELINE (All features):")
        print(baseline_df.to_string(index=False))

    print("\nSHAP + HUIM (Top 10 + 40 patterns):")
    print(results_df.to_string(index=False))
