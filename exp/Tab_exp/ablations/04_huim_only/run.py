"""
04_huim_only/run.py
====================
Ablation: Only HUIM binary pattern columns — no original features at all.
Tests whether mined combinatorial patterns alone can predict flight delay.

Run:
    python ablations/04_huim_only/run.py
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
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR        = cfg["paths"]["data_dir"]
CSV_FILE        = SCRIPT_DIR / "results.csv"
FEAT_CFG        = cfg["features"]
DRIVERS_FILE    = (SCRIPT_DIR / FEAT_CFG["drivers_file"]).resolve()
PROTECTORS_FILE = (SCRIPT_DIR / FEAT_CFG["protectors_file"]).resolve()

BIN_LABELS = {"VeryLow": 0, "Low": 1, "Mid": 2, "High": 3, "VeryHigh": 4}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    print(f"[{cfg['experiment']['name']}] Loading dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["data"]["file"]))
    df = df.dropna()
    df["ARR_DELAY"] = (df["ARR_DELAY"].abs() > 15.0).astype(int)

    with open(os.path.join(DATA_DIR, cfg["data"]["yaml_info"]), "r") as f:
        data_info = yaml.safe_load(f)

    exclude   = FEAT_CFG.get("exclude_cols", [])
    cat_cols  = [c for c in data_info["columns_info"]["Categorical Features"] if c not in exclude]
    cont_cols = [c for c in data_info["columns_info"]["Continuous Features"]  if c not in exclude]

    tmax = cfg["data"]["train_days_max"]
    vmax = cfg["data"]["val_days_max"]
    df_train = df[df["FL_DAY"] <= tmax].copy()
    df_val   = df[(df["FL_DAY"] > tmax) & (df["FL_DAY"] <= vmax)].copy()
    df_test  = df[df["FL_DAY"] > vmax].copy()

    return df_train, df_val, df_test, cat_cols, cont_cols


# ── HUIM binary feature engineering ───────────────────────────────────────────
def load_patterns():
    drivers_df    = pd.read_csv(DRIVERS_FILE)
    protectors_df = pd.read_csv(PROTECTORS_FILE)
    top_d = FEAT_CFG["top_drivers_n"]
    top_p = FEAT_CFG["top_protectors_n"]
    patterns = drivers_df.head(top_d)["Pattern"].tolist() + protectors_df.head(top_p)["Pattern"].tolist()
    print(f"Loaded {top_d} driver + {top_p} protector patterns = {len(patterns)} HUIM columns")
    return patterns


def add_huim_features(split_df, patterns, cat_cols, cont_cols, discretizer):
    cont_binned = discretizer.transform(split_df[cont_cols])
    for pat in patterns:
        matches = np.ones(len(split_df), dtype=bool)
        for cond in (c.strip() for c in pat.split(" + ")):
            col, val = cond.split("=")
            if col in cat_cols:
                try:
                    v = float(val)
                    matches &= (pd.to_numeric(split_df[col], errors="coerce") == v).fillna(False)
                except ValueError:
                    matches &= split_df[col].astype(str) == val
            elif col in cont_cols:
                bin_idx = BIN_LABELS[val]
                col_idx = cont_cols.index(col)
                matches &= cont_binned[:, col_idx] == bin_idx
        col_name = "HUIM_" + pat.replace(" + ", "_").replace("=", "_")
        split_df[col_name] = matches.astype(int)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df_train, df_val, df_test, cat_cols, cont_cols, patterns):
    # Discretizer is still needed to match HUIM continuous conditions
    discretizer = KBinsDiscretizer(
        n_bins=FEAT_CFG.get("n_bins", 5), encode="ordinal", strategy="quantile"
    )
    discretizer.fit(df_train[cont_cols])
    for split in (df_train, df_val, df_test):
        add_huim_features(split, patterns, cat_cols, cont_cols, discretizer)

    huim_cols = ["HUIM_" + p.replace(" + ", "_").replace("=", "_") for p in patterns]

    # mambular requires at least 1 numerical feature (num_features_list[0] crash otherwise).
    # Cast to float so columns are treated as numerical, not ordinal categorical.
    for split in (df_train, df_val, df_test):
        for col in huim_cols:
            split[col] = split[col].astype(float)

    # Zero-variance filter: float columns with std=0 → mambular's StandardScaler → NaN.
    # A pattern that never fires carries no information anyway.
    variances = df_train[huim_cols].var()
    valid_huim = [c for c in huim_cols if variances[c] > 1e-8]
    print(f"  Features: {len(valid_huim)}/{len(huim_cols)} HUIM columns "
          f"({len(huim_cols)-len(valid_huim)} zero-variance dropped)")

    X_train = df_train[valid_huim];  y_train = df_train["ARR_DELAY"]
    X_val   = df_val[valid_huim];    y_val   = df_val["ARR_DELAY"]
    X_test  = df_test[valid_huim];   y_test  = df_test["ARR_DELAY"]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    df_train, df_val, df_test, cat_cols, cont_cols = load_data()
    patterns = load_patterns()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(
        df_train, df_val, df_test, cat_cols, cont_cols, patterns
    )

    t = cfg["training"]
    X_search   = X_train.sample(frac=t["phase1_frac"], random_state=42)
    y_search   = y_train.loc[X_search.index]
    X_train_p2 = X_train.sample(frac=t["phase2_frac"], random_state=42)
    y_train_p2 = y_train.loc[X_train_p2.index]
    X_val_sm   = X_val.sample(frac=t["val_frac"], random_state=42)
    y_val_sm   = y_val.loc[X_val_sm.index]

    dl_kwargs = {"num_workers": t.get("num_workers", 0), "pin_memory": False}
    fit_params = {
        "max_epochs": t["max_epochs"], "rebuild": True,
        "X_val": X_val_sm, "y_val": y_val_sm,
        "patience": t["patience_phase1"],
        "accelerator": t["accelerator"], "devices": 1,
        "precision": t["precision"], "batch_size": t["batch_size"],
        "dataloader_kwargs": dl_kwargs,
    }
    fit_params_p2 = {**fit_params, "patience": t["patience_phase2"]}

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

    results_df = pd.read_csv(CSV_FILE) if CSV_FILE.exists() else pd.DataFrame(
        columns=["Model", "AUC", "ACC", "best_d_model", "best_n_layers", "best_lr"]
    )
    best_params_all = {}

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
        search.fit(X_search, y_search, **fit_params)
        bp = search.best_params_
        print(f"Best params: {bp}")

        print(f"Phase 2: retrain on {len(X_train_p2)} rows...")
        best_model = search.best_estimator_
        best_model.fit(X_train_p2, y_train_p2, **fit_params_p2)

        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} — AUC: {auc:.4f}, ACC: {acc:.4f}")

        row = pd.DataFrame({
            "Model": [model_name], "AUC": [auc], "ACC": [acc],
            "best_d_model": [bp.get("d_model")],
            "best_n_layers": [bp.get("n_layers")],
            "best_lr": [bp.get("lr")],
        })
        results_df = pd.concat([results_df, row], ignore_index=True)
        results_df.to_csv(CSV_FILE, index=False)

        # Save best params per model to JSON (survives across runs)
        best_params_all[model_name] = bp
        import json
        params_file = SCRIPT_DIR / "best_params.json"
        existing = json.loads(params_file.read_text()) if params_file.exists() else {}
        existing.update(best_params_all)
        params_file.write_text(json.dumps(existing, indent=2))

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nResults saved to {CSV_FILE}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
