"""
run.py  —  SHAP + HUIM Extraction
===================================
Config-driven experiment: use SHAP values as utility values for
High Utility Itemset Mining (HUIM) to discover combinatorial
delay-driving and delay-protecting feature patterns.

Pipeline:
  1. Load sample_frac of data (year set in config.yaml)
  2. Train MLP classifier (binary: |ARR_DELAY| > 15 min)
  3. Compute SHAP values
  4. Discretize features → items
  5. Build HUIM transaction databases (positive & negative SHAP streams)
  6. Mine with PAMI EFIM
  7. Save outputs to csvs/ and txt/ subdirs

Run:
    python run.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION — read from config.yaml
# ============================================================
SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "config.yaml", "r") as _f:
    cfg = yaml.safe_load(_f)

MAIN_PATH         = cfg["paths"]["data_dir"]
FILE_NAME         = cfg["data"]["file"]
DATA_INFO_FILE    = cfg["data"]["yaml_info"]
DATA_FRAC         = cfg["data"]["sample_frac"]
SHAP_BG_SAMPLES   = cfg["shap"].get("bg_samples", 200)
SHAP_TEST_SAMPLES = cfg["shap"].get("test_samples", 1000)
N_BINS            = cfg["huim"]["n_bins"]
SHAP_SCALE        = cfg["huim"].get("shap_scale", 10000)
MIN_UTIL_PCTILE   = cfg["huim"].get("min_util_percentile", 90)

CSVS_DIR = SCRIPT_DIR / cfg["outputs"]["csvs_dir"]
TXT_DIR  = SCRIPT_DIR / cfg["outputs"]["txt_dir"]
CSVS_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)

print(f"Experiment : {cfg['experiment']['name']}")
print(f"Data file  : {FILE_NAME}  (sample_frac={DATA_FRAC})")

# ============================================================
# 2. DATA LOADING (10% sample)
# ============================================================
print("=" * 60)
print("  STEP 1: Loading data (10% sample)")
print("=" * 60)

df_full = pd.read_csv(os.path.join(MAIN_PATH, FILE_NAME))
df = df_full.sample(frac=DATA_FRAC, random_state=42).copy()
df = df.dropna()
del df_full  # free memory
print(f"  Loaded {len(df)} rows (10% sample)")

# ============================================================
# 3. FEATURE METADATA
# ============================================================
with open(os.path.join(MAIN_PATH, DATA_INFO_FILE), "r") as f:
    data_info = yaml.load(f, Loader=yaml.FullLoader)

categorical_columns = data_info["columns_info"]["Categorical Features"]
continuous_columns = data_info["columns_info"]["Continuous Features"]

# Remove non-useful features (same as main pipeline)
continuous_columns = [c for c in continuous_columns if c != "FLIGHTS"]
categorical_columns = [c for c in categorical_columns if c not in ("FL_YEAR", "FL_MONTH")]

all_feature_cols = categorical_columns + continuous_columns
print(f"  Features: {len(categorical_columns)} categorical + {len(continuous_columns)} continuous")

# ============================================================
# 4. TARGET VARIABLE — BINARY CLASSIFICATION
# ============================================================
df["ARR_DELAY"] = df["ARR_DELAY"].apply(lambda x: 1 if abs(x) > 15.0 else 0)
print(f"  Target distribution: {df['ARR_DELAY'].value_counts().to_dict()}")

# ============================================================
# 5. TIME-BASED SPLIT
# ============================================================
df_train = df[df["FL_DAY"] <= 9].copy()
df_valid = df[(df["FL_DAY"] > 9) & (df["FL_DAY"] <= 12)].copy()
df_test = df[df["FL_DAY"] > 12].copy()

print(f"  Split: train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}")

# ============================================================
# 6. PREPROCESSING
# ============================================================
# Stores the LabelEncoder for each categorical column (last fit wins per col).
# We keep them so we can later reverse-map encoded integers → original labels
# (e.g. 3 → "AA") when building human-readable HUIM item names.
label_encoders = {}
scalers = {}

for split_df in [df_train, df_valid, df_test]:
    # --- Categorical features ---
    # Convert string categories (e.g. "AA", "DL") to integers (0, 1, 2, …).
    # Mambular models and SHAP both expect numeric inputs.
    # NOTE: fitting a fresh encoder per split means the same label can get
    # a different integer in train vs. test — acceptable here because
    # each split is used independently, but worth knowing.
    for col in categorical_columns:
        le = LabelEncoder()
        split_df[col] = le.fit_transform(split_df[col])
        label_encoders[col] = le  # overwritten each split; keeps test's encoder

    # --- Continuous features ---
    # Standardize to zero-mean, unit-variance (z-score normalization).
    # Neural networks train more stably when inputs are on a similar scale.
    # Same caveat: fit per split, so each split has its own mean/std.
    scaler = StandardScaler()
    split_df[continuous_columns] = scaler.fit_transform(split_df[continuous_columns])
    scalers["last"] = scaler

# Separate feature matrices (X) and target vectors (y) for each split
X_train = df_train[all_feature_cols]
X_valid = df_valid[all_feature_cols]
X_test = df_test[all_feature_cols]
y_train = df_train["ARR_DELAY"]
y_valid = df_valid["ARR_DELAY"]
y_test = df_test["ARR_DELAY"]

# ============================================================
# 7. TRAIN MLP CLASSIFIER
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: Training MLP Classifier")
print("=" * 60)

from mambular.models import MLPClassifier

model = MLPClassifier(d_model=64)
model.fit(
    X_train, y_train,
    X_val=X_valid, y_val=y_valid,
    max_epochs=50,
    lr=1e-3,
    patience=5,
    batch_size=4096,
)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"  MLP Results — AUC: {auc:.4f}, ACC: {acc:.4f}")

# ============================================================
# 8. COMPUTE SHAP VALUES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: Computing SHAP values")
print("=" * 60)

# Subsample for speed
X_bg = X_train.sample(n=min(SHAP_BG_SAMPLES, len(X_train)), random_state=42)
X_explain = X_test.sample(n=min(SHAP_TEST_SAMPLES, len(X_test)), random_state=42)

print(f"  Background: {len(X_bg)}, Explaining: {len(X_explain)} instances")

explainer = shap.Explainer(model.predict, X_bg)
shap_result = explainer(X_explain)
shap_values = shap_result.values  # shape: (n_samples, n_features)

print(f"  SHAP values shape: {shap_values.shape}")
print(f"  SHAP range: [{shap_values.min():.4f}, {shap_values.max():.4f}]")

# Quick global feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": all_feature_cols,
    "Mean|SHAP|": mean_abs_shap
}).sort_values("Mean|SHAP|", ascending=False)
print("\n  Top 10 features by mean |SHAP|:")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# 9. DISCRETIZE CONTINUOUS FEATURES → ITEMS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: Discretizing features → items")
print("=" * 60)

# Discretize continuous features into quantile bins
X_explain_cont = X_explain[continuous_columns].values
discretizer = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
X_explain_disc = discretizer.fit_transform(X_explain_cont).astype(int)

# Build a mapping: (feature_index, bin_or_value) → item_id (integer)
# PAMI works best with integer item IDs
item_id_map = {}   # (col_name, bin/value) → integer ID
item_label_map = {} # integer ID → human-readable label
next_id = 1

# Categorical items
for feat_idx, col in enumerate(categorical_columns):
    for val in sorted(X_explain[col].unique()):
        key = (col, int(val))
        item_id_map[key] = next_id
        # Try to get original label
        if col in label_encoders:
            try:
                orig = label_encoders[col].inverse_transform([int(val)])[0]
                item_label_map[next_id] = f"{col}={orig}"
            except (ValueError, IndexError):
                item_label_map[next_id] = f"{col}={val}"
        else:
            item_label_map[next_id] = f"{col}={val}"
        next_id += 1

# Continuous items (binned)
bin_labels = {0: "VeryLow", 1: "Low", 2: "Mid", 3: "High", 4: "VeryHigh"}
for feat_idx, col in enumerate(continuous_columns):
    for b in range(N_BINS):
        key = (col, b)
        item_id_map[key] = next_id
        item_label_map[next_id] = f"{col}={bin_labels.get(b, f'B{b}')}"
        next_id += 1

print(f"  Total unique items: {next_id - 1}")

# ============================================================
# 10. BUILD HUIM TRANSACTION DATABASES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Building HUIM transaction databases")
print("=" * 60)


def build_utility_transactions(X_explain_df, X_explain_disc, shap_vals, mode="positive"):
    """
    Build PAMI-format utility transaction lines.

    Format per line: item1<tab>item2<tab>...:totalUtility:u1<tab>u2<tab>...

    Parameters
    ----------
    mode : 'positive' — only items with SHAP > 0 (delay drivers)
           'negative' — only items with SHAP < 0, using |SHAP| (delay protectors)
    """
    lines = []
    n_cat = len(categorical_columns)
    n_cont = len(continuous_columns)
    skipped = 0

    for row in range(len(X_explain_df)):
        items = []
        utilities = []

        # Categorical features
        for feat_idx, col in enumerate(categorical_columns):
            val = int(X_explain_df.iloc[row][col])
            sv = shap_vals[row, feat_idx]
            key = (col, val)

            if key not in item_id_map:
                continue

            if mode == "positive" and sv > 0:
                items.append(item_id_map[key])
                utilities.append(int(abs(sv) * SHAP_SCALE))
            elif mode == "negative" and sv < 0:
                items.append(item_id_map[key])
                utilities.append(int(abs(sv) * SHAP_SCALE))

        # Continuous features (discretized)
        for feat_idx, col in enumerate(continuous_columns):
            bin_val = X_explain_disc[row, feat_idx]
            sv = shap_vals[row, n_cat + feat_idx]
            key = (col, bin_val)

            if key not in item_id_map:
                continue

            if mode == "positive" and sv > 0:
                items.append(item_id_map[key])
                utilities.append(int(abs(sv) * SHAP_SCALE))
            elif mode == "negative" and sv < 0:
                items.append(item_id_map[key])
                utilities.append(int(abs(sv) * SHAP_SCALE))

        # Ensure non-zero utilities (HUIM requires positive utilities)
        utilities = [max(u, 1) for u in utilities]

        if items:
            trans_util = sum(utilities)
            item_str = "\t".join(str(i) for i in items)
            util_str = "\t".join(str(u) for u in utilities)
            lines.append(f"{item_str}:{trans_util}:{util_str}")
        else:
            skipped += 1

    print(f"    [{mode}] Transactions: {len(lines)}, Skipped (empty): {skipped}")
    return lines


# Build positive (delay drivers) and negative (delay protectors)
pos_lines = build_utility_transactions(X_explain, X_explain_disc, shap_values, mode="positive")
neg_lines = build_utility_transactions(X_explain, X_explain_disc, shap_values, mode="negative")

# Write to txt/ subdir
pos_file = str(TXT_DIR / "shap_utility_positive.txt")
neg_file = str(TXT_DIR / "shap_utility_negative.txt")

with open(pos_file, "w") as f:
    f.write("\n".join(pos_lines))
with open(neg_file, "w") as f:
    f.write("\n".join(neg_lines))

print(f"  Written: {pos_file}")
print(f"  Written: {neg_file}")

# ============================================================
# 11. MINE HIGH UTILITY ITEMSETS WITH PAMI EFIM
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: Mining High Utility Itemsets (EFIM)")
print("=" * 60)

from PAMI.highUtilityPattern.basic import EFIM


def mine_patterns(input_file, label, min_util_percentile=MIN_UTIL_PCTILE):
    """
    Run EFIM on a utility transaction file.
    Automatically determines minUtil from the data distribution.
    """
    # Read transaction utilities to set a data-driven threshold
    trans_utils = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) >= 2:
                trans_utils.append(int(parts[1]))

    if not trans_utils:
        print(f"  [{label}] No transactions found!")
        return pd.DataFrame()

    # Set minUtil as the Nth percentile of transaction utilities
    min_util = int(np.percentile(trans_utils, min_util_percentile))
    print(f"  [{label}] Transaction utility range: [{min(trans_utils)}, {max(trans_utils)}]")
    print(f"  [{label}] minUtil ({min_util_percentile}th percentile): {min_util}")

    # Run EFIM
    obj = EFIM.EFIM(input_file, min_util, "\t")
    obj.mine()

    patterns = obj.getPatterns()
    if not patterns:
        # Try with a lower threshold
        min_util = int(np.percentile(trans_utils, 75))
        print(f"  [{label}] No patterns at p90, retrying with p75 minUtil={min_util}")
        obj = EFIM.EFIM(input_file, min_util, "\t")
        obj.mine()
        patterns = obj.getPatterns()

    if not patterns:
        print(f"  [{label}] No patterns found even at p75 threshold.")
        return pd.DataFrame()

    # Convert to readable format
    results = []
    for pattern_str, utility in patterns.items():
        # pattern_str contains item IDs separated by tab
        item_ids = [int(x.strip()) for x in str(pattern_str).split("\t") if x.strip().isdigit()]
        readable_items = [item_label_map.get(iid, f"?{iid}") for iid in item_ids]
        results.append({
            "Pattern": " + ".join(readable_items),
            "ItemIDs": item_ids,
            "Utility": utility,
            "Size": len(item_ids),
        })

    results_df = pd.DataFrame(results).sort_values("Utility", ascending=False)

    print(f"  [{label}] Found {len(results_df)} high-utility patterns")
    print(f"\n  Top 15 {label} patterns:")
    print("  " + "-" * 70)
    for _, row in results_df.head(15).iterrows():
        print(f"  Utility={row['Utility']:>8}  |  {row['Pattern']}")
    print("  " + "-" * 70)

    return results_df


# Mine both streams
print("\n--- Delay DRIVERS (positive SHAP) ---")
drivers_df = mine_patterns(pos_file, "Delay Drivers")

print("\n--- Delay PROTECTORS (negative SHAP) ---")
protectors_df = mine_patterns(neg_file, "Delay Protectors")

# ============================================================
# 12. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: Saving results")
print("=" * 60)

if not drivers_df.empty:
    out = CSVS_DIR / "huim_delay_drivers.csv"
    drivers_df.to_csv(out, index=False)
    print(f"  Saved: {out} ({len(drivers_df)} patterns)")

if not protectors_df.empty:
    out = CSVS_DIR / "huim_delay_protectors.csv"
    protectors_df.to_csv(out, index=False)
    print(f"  Saved: {out} ({len(protectors_df)} patterns)")

# Save item label mapping for reference
label_df = pd.DataFrame([
    {"ItemID": k, "Label": v} for k, v in item_label_map.items()
]).sort_values("ItemID")
label_df.to_csv(CSVS_DIR / "huim_item_labels.csv", index=False)

# Save SHAP importance for reference
importance_df.to_csv(CSVS_DIR / "shap_feature_importance.csv", index=False)

print("\n" + "=" * 60)
print("  EXPERIMENT COMPLETE")
print("=" * 60)
print(f"  Model: MLP (AUC={auc:.4f}, ACC={acc:.4f})")
print(f"  SHAP computed on {len(X_explain)} instances")
print(f"  Delay driver patterns: {len(drivers_df) if not drivers_df.empty else 0}")
print(f"  Delay protector patterns: {len(protectors_df) if not protectors_df.empty else 0}")
