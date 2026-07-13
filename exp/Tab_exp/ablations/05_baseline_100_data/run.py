"""
05_baseline_100_data/run.py
===========================
LLM-Select (LLM-Score) feature selection with a local Ollama model.

Method: taekb/llm-select — "LLM-Select: Feature Selection with Large Language
Models" (Jeong, Lipton & Ravikumar, TMLR 2025). LLM-Score asks the model for an
independent importance score in [0, 1] for each feature. We run two prompt
variants and score every feature n_repeats times, then aggregate mean +/- std:

  - data_free:     feature name + description + task context only (paper-faithful)
  - data_informed: same, plus the 100 sample rows from output/llm_sample_raw.csv

Outputs (in output/):
  - llm_feature_importance_datafree.csv       (Feature, LLM_Score, LLM_Score_std, Reasoning)
  - llm_feature_importance_datainformed.csv
  - raw_llm_outputs/<variant>/<feature>_rep<k>.json   (audit trail)

The importance CSVs mirror the shape of the SHAP file (Feature, score) so they
can be compared directly (see compare.py) or reused by future ablations.

Run:
    uv run python exp/Tab_exp/ablations/05_baseline_100_data/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).parent
HELPERS_DIR = SCRIPT_DIR / "helpers"
sys.path.insert(0, str(HELPERS_DIR))
from llm_client import OllamaScorer  # noqa: E402  (helpers/ added to path above)

with open(HELPERS_DIR / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
LS = CFG["llm_select"]


def resolve(rel: str) -> Path:
    """Resolve a config path relative to helpers/ (matches export convention)."""
    return (HELPERS_DIR / rel).resolve()


def build_data_block(sample_csv: Path, max_rows: int) -> str:
    """Compact CSV text of the sample rows for the data_informed variant.

    Drops the raw ARR_DELAY column (that IS the target) and keeps the binary
    DELAYED label, so the model sees feature->label relationships without the
    exact answer leaking in.
    """
    df = pd.read_csv(sample_csv)
    df = df.drop(columns=[c for c in ("ARR_DELAY",) if c in df.columns])
    df = df.head(max_rows)
    return df.to_csv(index=False).strip()


def score_variant(scorer: OllamaScorer, variant: str, features: dict,
                  data_block: str | None, n_repeats: int, raw_root: Path) -> pd.DataFrame:
    raw_dir = raw_root / variant
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (feat, meta) in enumerate(features.items(), 1):
        scores, reasoning = [], ""
        for rep in range(n_repeats):
            try:
                res = scorer.score_feature(feat, meta, data_block=data_block, seed=rep)
            except Exception as exc:  # noqa: BLE001 — keep the long run alive
                print(f"    ! {feat} rep{rep}: {exc}")
                continue
            scores.append(res.score)
            reasoning = reasoning or res.reasoning
            (raw_dir / f"{feat}_rep{rep}.json").write_text(
                json.dumps({"feature": feat, "rep": rep, "score": res.score,
                            "reasoning": res.reasoning, "raw": res.raw},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        mean = float(np.mean(scores)) if scores else float("nan")
        std = float(np.std(scores)) if scores else float("nan")
        rows.append({"Feature": feat, "LLM_Score": mean,
                     "LLM_Score_std": std, "Reasoning": reasoning})
        print(f"  [{variant}] {i:2d}/{len(features)}  {feat:<20} "
              f"score={mean:.3f} +/- {std:.3f}  (n={len(scores)})")

    return (pd.DataFrame(rows)
            .sort_values("LLM_Score", ascending=False, ignore_index=True))


def main():
    with open(resolve(LS["feature_desc_file"]), encoding="utf-8") as f:
        features = yaml.safe_load(f)          # ordered dict: name -> {type, unit, description}

    sample_csv = resolve(LS["sample_file"])
    out_dir = resolve(LS["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir / "raw_llm_outputs"

    scorer = OllamaScorer(
        model=LS["model"],
        host=LS["ollama_host"],
        temperature=LS["temperature"],
        timeout=LS.get("request_timeout", 120),
        target=LS["target_description"],
    )

    print(f"LLM-Score via Ollama model '{LS['model']}' — {len(features)} features, "
          f"{LS['n_repeats']} repeats/feature")
    print(f"Variants: {LS['variants']}\n")

    for variant in LS["variants"]:
        data_block = None
        if variant == "data_informed":
            data_block = build_data_block(sample_csv, LS.get("max_data_rows", 100))
        print(f"=== variant: {variant} ===")
        df = score_variant(scorer, variant, features, data_block,
                           LS["n_repeats"], raw_root)
        out_path = out_dir / f"llm_feature_importance_{variant.replace('_', '')}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> wrote {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
