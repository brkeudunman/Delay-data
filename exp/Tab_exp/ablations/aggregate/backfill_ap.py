"""
backfill_ap.py — add average precision (PR-AUC) to existing results.csv rows
=========================================================================
Runs from before `average_precision` was logged have their test probabilities in
<arm>/predictions/*.npy and the labels in _ytest/<target>__<test_hash>.npy. For each
results.csv row with an EMPTY `average_precision` this loads the matching predictions
file (match on model, seed, target, phase2_frac, extraction; file names parsed with
FNAME from paired_test.py), computes AP of the positive class, and writes it into that
row. Values already present are never overwritten; auc / accuracy / f1 are never touched.

Guard: AUC is recomputed from the same file and must equal the row's `auc` (float32
predictions -> tolerance AUC_TOL); otherwise the row is reported and left empty.

Reported: rows without a predictions file (predictions/ is gitignored), prediction files
without a results row, missing label files, AUC mismatches. 3class rows get NaN (AP is
defined for binary targets only).

    uv run python exp/Tab_exp/ablations/aggregate/backfill_ap.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ABLATIONS = HERE.parent
sys.path.insert(0, str(ABLATIONS))
from common.metrics import average_precision, positive_scores  # noqa: E402
from common.runner import RESULT_COLUMNS  # noqa: E402
from paired_test import FNAME  # noqa: E402

COL = "average_precision"
AUC_TOL = 1e-4


def index_predictions(arm_dir: Path) -> dict:
    out = {}
    for f in sorted((arm_dir / "predictions").glob("*.npy")):
        mt = FNAME.match(f.name)
        if not mt:
            print(f"  ?? unparsable prediction file name: {f.name}")
            continue
        key = (mt["model"], int(mt["seed"]), mt["target"], round(float(mt["p2"]), 6), mt["ext"])
        out[key] = f
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    a = ap.parse_args()

    labels = {}
    missing, orphans, mismatches, no_labels = [], [], [], []
    n_filled = n_kept = 0
    for res_file in sorted(ABLATIONS.glob("*/results.csv")):
        df = pd.read_csv(res_file)
        if "test_hash" not in df.columns:  # not a runner file
            continue
        arm = res_file.parent.name
        if COL not in df.columns:
            df[COL] = np.nan
        preds = index_predictions(res_file.parent)
        used = set()
        filled_here = 0
        for i, r in df.iterrows():
            key = (r.model, int(r.seed), r.target, round(float(r.phase2_frac), 6), r.extraction)
            used.add(key)
            if pd.notna(r[COL]):
                n_kept += 1
                continue
            if r.target == "3class":
                continue  # stays NaN
            f = preds.get(key)
            if f is None:
                missing.append((arm, *key))
                continue
            yf = ABLATIONS / "_ytest" / f"{r.target}__{r.test_hash}.npy"
            if not yf.exists():
                no_labels.append((arm, *key, yf.name))
                continue
            if yf not in labels:
                labels[yf] = np.load(yf).astype(int)
            y = labels[yf]
            proba = np.load(f).astype(np.float64)
            if len(proba) != len(y):
                mismatches.append((arm, *key, f"length {len(proba)} vs {len(y)} labels"))
                continue
            auc = roc_auc_score(y, positive_scores(proba))
            if abs(auc - r.auc) > AUC_TOL:
                mismatches.append((arm, *key, f"AUC from file {auc:.6f} vs results.csv {r.auc:.6f}"))
                continue
            df.at[i, COL] = average_precision(y, proba, n_classes=2)
            filled_here += 1
            print(f"  {arm} {key}: AP={df.at[i, COL]:.4f} (base rate {r.test_pos_rate:.4f}, "
                  f"AUC check Δ={auc - r.auc:+.1e})")
        orphans += [(arm, *k, f.name) for k, f in preds.items() if k not in used]
        n_filled += filled_here
        if filled_here and not a.dry_run:
            cols = [c for c in RESULT_COLUMNS if c in df.columns] + \
                   [c for c in df.columns if c not in RESULT_COLUMNS]
            df[cols].to_csv(res_file, index=False)
            print(f"-> {res_file.relative_to(ABLATIONS)}: {filled_here} rows filled")

    print(f"\nfilled {n_filled} rows, {n_kept} already had a value"
          + (" (dry run, nothing written)" if a.dry_run else ""))
    for title, items in (("rows WITHOUT a predictions file", missing),
                         ("rows whose label file is missing", no_labels),
                         ("rows skipped: predictions do not match the row", mismatches),
                         ("prediction files without a results.csv row", orphans)):
        print(f"\n{title}: {len(items)}")
        for it in items:
            print("  ", *it)


if __name__ == "__main__":
    main()
