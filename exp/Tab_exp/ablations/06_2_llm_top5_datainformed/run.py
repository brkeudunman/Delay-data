"""
06_2_llm_top5_datainformed/run.py
=================================
Ablation: Top-5 LLM-selected features only (LLM-Select / LLM-Score, qwen2.5:7b).

Identical to 02_shap_only/run.py apart from the name of the importance file it
reads — selection is still "take the first top_n rows of a Feature-ranked CSV".
The ranking comes from ablations/05_baseline_100_data instead of SHAP.

Everything else (split, encoding, two-phase training, model set, metric) is kept
byte-for-byte with 02/03/04 so the results are directly comparable. That includes
the metric wart: AUC is computed on hard predicted labels, not probabilities, so
it is really balanced accuracy. Do not "fix" it here alone — it would silently
break comparability with the ablations already in the table.

Run:
    uv run python exp/Tab_exp/ablations/06_2_llm_top5_datainformed/run.py
"""
import gc
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from mambular.models import AutoIntClassifier, MLPClassifier, ResNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR   = cfg["paths"]["data_dir"]
CSV_FILE   = SCRIPT_DIR / "results.csv"
FEAT_CFG   = cfg["features"]

# Resolve relative file paths against this script's directory
IMPORTANCE_FILE = (SCRIPT_DIR / FEAT_CFG["importance_file"]).resolve()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    print(f"[{cfg['experiment']['name']}] Loading dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["data"]["file"]))
    df = df.dropna()
    df["ARR_DELAY"] = (df["ARR_DELAY"].abs() > 15.0).astype(int)

    with open(os.path.join(DATA_DIR, cfg["data"]["yaml_info"]), "r") as f:
        data_info = yaml.safe_load(f)

    exclude   = FEAT_CFG["exclude_cols"]
    cat_cols  = [c for c in data_info["columns_info"]["Categorical Features"] if c not in exclude]
    cont_cols = [c for c in data_info["columns_info"]["Continuous Features"]  if c not in exclude]

    tmax = cfg["data"]["train_days_max"]
    vmax = cfg["data"]["val_days_max"]
    df_train = df[df["FL_DAY"] <= tmax].copy()
    df_val   = df[(df["FL_DAY"] > tmax) & (df["FL_DAY"] <= vmax)].copy()
    df_test  = df[df["FL_DAY"] > vmax].copy()

    return df_train, df_val, df_test, cat_cols, cont_cols


# ── Feature selection: top-N of a ranked importance CSV ───────────────────────
def select_top_features(cat_cols, cont_cols):
    imp_df = pd.read_csv(IMPORTANCE_FILE)
    top_n  = FEAT_CFG["top_n"]
    top    = imp_df.head(top_n)["Feature"].tolist()
    kept_cat  = [c for c in cat_cols  if c in top]
    kept_cont = [c for c in cont_cols if c in top]
    missing   = [f for f in top if f not in cat_cols + cont_cols]
    print(f"Ranking file: {IMPORTANCE_FILE.name}")
    print(f"Selection (top {top_n}): {len(kept_cont)} continuous, {len(kept_cat)} categorical")
    print(f"  kept: {kept_cat + kept_cont}")
    if missing:
        print(f"  !! {len(missing)} ranked features absent from the dataset schema: {missing}")
    return kept_cat, kept_cont


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df_train, df_val, df_test, cat_cols, cont_cols):
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    for split in (df_train, df_val, df_test):
        for col in cat_cols:
            le = LabelEncoder()
            split[col] = le.fit_transform(split[col].astype(str))
        if cont_cols:
            split[cont_cols] = scaler.transform(split[cont_cols])

    features = cat_cols + cont_cols
    X_train = df_train[features];  y_train = df_train["ARR_DELAY"]
    X_val   = df_val[features];    y_val   = df_val["ARR_DELAY"]
    X_test  = df_test[features];   y_test  = df_test["ARR_DELAY"]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    df_train, df_val, df_test, cat_cols, cont_cols = load_data()
    kept_cat, kept_cont = select_top_features(cat_cols, cont_cols)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(
        df_train, df_val, df_test, kept_cat, kept_cont
    )

    t = cfg["training"]

    X_search   = X_train.sample(frac=t["phase1_frac"], random_state=42)
    y_search   = y_train.loc[X_search.index]
    X_train_p2 = X_train.sample(frac=t["phase2_frac"], random_state=42)
    y_train_p2 = y_train.loc[X_train_p2.index]
    X_val_sm   = X_val.sample(frac=t["val_frac"], random_state=42)
    y_val_sm   = y_val.loc[X_val_sm.index]

    dl_kwargs = {"num_workers": t.get("num_workers", 0), "pin_memory": False}

    # Both phases use explicit X_val (no internal split)
    fit_params = {
        "max_epochs": t["max_epochs"],
        "rebuild": True,
        "X_val": X_val_sm,
        "y_val": y_val_sm,
        "patience": t["patience_phase1"],
        "accelerator": t["accelerator"],
        "devices": 1,
        "precision": t["precision"],
        "batch_size": t["batch_size"],
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
        search.fit(X_search, y_search, **fit_params)
        print(f"Best params: {search.best_params_}")

        print(f"Phase 2: retrain on {len(X_train_p2)} rows...")
        best_model = search.best_estimator_
        best_model.fit(X_train_p2, y_train_p2, **fit_params_p2)

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
