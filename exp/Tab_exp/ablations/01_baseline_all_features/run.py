"""
01_baseline_all_features/run.py
================================
Ablation: All raw features, no SHAP selection, no HUIM.
2-phase training — mirrors Deep_Model_Classifier_fast.py.

Run:
    python ablations/01_baseline_all_features/run.py
"""
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from mambular.models import AutoIntClassifier, MLPClassifier, ResNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["paths"]["data_dir"]
CSV_FILE = SCRIPT_DIR / "results.csv"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    print(f"[{cfg['experiment']['name']}] Loading dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["data"]["file"]))
    df = df.dropna()
    df["ARR_DELAY"] = (df["ARR_DELAY"].abs() > 15.0).astype(int)

    with open(os.path.join(DATA_DIR, cfg["data"]["yaml_info"]), "r") as f:
        data_info = yaml.safe_load(f)

    exclude = cfg["features"]["exclude_cols"]
    cat_cols = [c for c in data_info["columns_info"]["Categorical Features"] if c not in exclude]
    cont_cols = [c for c in data_info["columns_info"]["Continuous Features"] if c not in exclude]

    tmax = cfg["data"]["train_days_max"]
    vmax = cfg["data"]["val_days_max"]
    df_train = df[df["FL_DAY"] <= tmax].copy()
    df_val   = df[(df["FL_DAY"] > tmax) & (df["FL_DAY"] <= vmax)].copy()
    df_test  = df[df["FL_DAY"] > vmax].copy()

    return df_train, df_val, df_test, cat_cols, cont_cols


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df_train, df_val, df_test, cat_cols, cont_cols):
    for split in (df_train, df_val, df_test):
        for col in cat_cols:
            le = LabelEncoder()
            split[col] = le.fit_transform(split[col].astype(str))
        scaler = StandardScaler()
        split[cont_cols] = scaler.fit_transform(split[cont_cols])

    features = cat_cols + cont_cols
    X_train = df_train[features];  y_train = df_train["ARR_DELAY"]
    X_val   = df_val[features];    y_val   = df_val["ARR_DELAY"]
    X_test  = df_test[features];   y_test  = df_test["ARR_DELAY"]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    df_train, df_val, df_test, cat_cols, cont_cols = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(
        df_train, df_val, df_test, cat_cols, cont_cols
    )

    t = cfg["training"]

    # Sub-samples
    X_search = X_train.sample(frac=t["phase1_frac"], random_state=42)
    y_search  = y_train.loc[X_search.index]

    p2_frac = t["phase2_frac"]
    X_train_p2 = X_train if p2_frac >= 1.0 else X_train.sample(frac=p2_frac, random_state=42)
    y_train_p2 = y_train if p2_frac >= 1.0 else y_train.loc[X_train_p2.index]

    v_frac = t["val_frac"]
    X_val_p2 = X_val if v_frac >= 1.0 else X_val.sample(frac=v_frac, random_state=42)
    y_val_p2 = y_val if v_frac >= 1.0 else y_val.loc[X_val_p2.index]

    dl_kwargs = {"num_workers": t.get("num_workers", 0), "pin_memory": False}

    # Phase 1: internal val split (val_size), no explicit X_val
    fit_p1 = {
        "max_epochs": t["max_epochs"],
        "rebuild": True,
        "val_size": t.get("phase1_val_size", 0.5),
        "patience": t["patience_phase1"],
        "accelerator": t["accelerator"],
        "devices": 1,
        "precision": t["precision"],
        "batch_size": t["batch_size"],
        "dataloader_kwargs": dl_kwargs,
    }

    # Phase 2: explicit val set
    fit_p2 = {
        "max_epochs": t["max_epochs"],
        "rebuild": True,
        "X_val": X_val_p2,
        "y_val": y_val_p2,
        "patience": t["patience_phase2"],
        "accelerator": t["accelerator"],
        "devices": 1,
        "precision": t["precision"],
        "batch_size": t["batch_size"],
        "dataloader_kwargs": dl_kwargs,
    }

    param_dist = {
        "d_model": t.get("d_model_options", [64, 128, 256]),
        "n_layers": t.get("n_layers_options", [2, 6, 10]),
        "lr": t.get("lr_options", [1e-5, 1e-4, 1e-3]),
    }

    models = {
        "MLP":     MLPClassifier(d_model=128),
        "AutoInt": AutoIntClassifier(d_model=128, n_layers=4),
        "ResNet":  ResNetClassifier(),
    }

    results_df = pd.read_csv(CSV_FILE) if CSV_FILE.exists() else pd.DataFrame(columns=["Model", "AUC", "ACC"])

    for model_name, model in models.items():
        if model_name in results_df["Model"].values:
            print(f"Skipping {model_name} — already done.")
            continue

        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

        print(f"Phase 1: search on {len(X_search)} rows...")
        search = RandomizedSearchCV(
            estimator=model, param_distributions=param_dist,
            n_iter=t["n_iter"], cv=t["cv"], scoring="accuracy",
            random_state=42, n_jobs=1,
        )
        search.fit(X_search, y_search, **fit_p1)
        print(f"Best params: {search.best_params_}")

        print(f"Phase 2: retrain on {len(X_train_p2)} rows...")
        best_model = search.best_estimator_
        best_model.fit(X_train_p2, y_train_p2, **fit_p2)

        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} — AUC: {auc:.4f}, ACC: {acc:.4f}")

        results_df = pd.concat(
            [results_df, pd.DataFrame({"Model": [model_name], "AUC": [auc], "ACC": [acc]})],
            ignore_index=True,
        )
        results_df.to_csv(CSV_FILE, index=False)
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nResults saved to {CSV_FILE}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
