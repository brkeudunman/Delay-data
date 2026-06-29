"""
05_baseline_1000_data/run.py
============================
Export a small set of RAW (un-encoded) flight rows for LLM prompting.

No model training. Samples N rows from the train-day split (FL_DAY <= train_days_max),
keeps the original human-readable feature values (carrier codes, airport codes,
scheduled times, weather, coordinates), and appends the raw ARR_DELAY plus a binary
DELAYED label (|ARR_DELAY| > threshold). DEP_DELAY is excluded — it leaks the target.

Run:
    uv run python exp/Tab_exp/ablations/05_baseline_1000_data/run.py
"""
import os
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).parent
with open(SCRIPT_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["paths"]["data_dir"]


def main():
    d, e = cfg["data"], cfg["export"]

    with open(os.path.join(DATA_DIR, d["yaml_info"])) as f:
        info = yaml.safe_load(f)["columns_info"]

    exclude = set(cfg["features"]["exclude_cols"])
    target = e["target_col"]
    feat_cols = [c for c in info["Categorical Features"] + info["Continuous Features"]
                 if c not in exclude]

    # read only what we need: features + target + the split key
    need = list(dict.fromkeys(feat_cols + [target, "FL_DAY"]))
    df = pd.read_csv(os.path.join(DATA_DIR, d["file"]), usecols=need).dropna()

    # train-day split only — leave val/test days untouched for any downstream eval
    df = df[df["FL_DAY"] <= d["train_days_max"]]

    thr = e["delay_threshold"]
    df["DELAYED"] = (df[target].abs() > thr).astype(int)

    n = min(e["n_rows"], len(df))
    sample = df.sample(n=n, random_state=e["random_state"]).reset_index(drop=True)

    # column order: features, raw delay (the answer), binary label
    out_cols = feat_cols + [target, "DELAYED"]
    sample = sample[out_cols]

    out_path = SCRIPT_DIR / cfg["output"]["csv"]
    sample.to_csv(out_path, index=False)

    pos = int(sample["DELAYED"].sum())
    print(f"Wrote {len(sample)} rows -> {out_path}")
    print(f"  delayed (|{target}|>{thr}): {pos}  |  on-time: {len(sample) - pos}")
    print(f"  {len(feat_cols)} feature cols: {feat_cols}")


if __name__ == "__main__":
    main()
