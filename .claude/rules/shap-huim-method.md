# The SHAP + HUIM method (core novelty)

Pipeline in `exp/Tab_exp/shap_huim_extractions/01_2024_baseline/run.py`:

1. Load `sample_frac` (10%) of training data; train an MLP binary classifier.
2. Compute SHAP values (`KernelExplainer`, `bg_samples=200`, `test_samples=1000`).
3. Discretize continuous features (`KBinsDiscretizer`, 5 quantile bins, fit on train only) → categorical "items".
4. Build two HUIM transaction streams: **positive-SHAP items = delay drivers**, **negative-SHAP items = delay protectors**. Utility = `int(|SHAP| × shap_scale)`, `shap_scale=10000` (SHAP values are small floats; scaling avoids integer-rounding loss in PAMI).
5. Mine each stream with **PAMI EFIM**, `minUtil` = 90th percentile of transaction utilities → `delay_drivers` / `delay_protectors`.

**Hand-off gotcha (important):** the ablations read these files from `exp/Tab_exp/SHAP_HUIM/` with fixed names — `shap_feature_importance.csv`, `huim_delay_drivers.csv`, `huim_delay_protectors.csv` (see `03_shap_plus_huim/config.yaml` → `features.shap_importance_file` etc., paths like `../../SHAP_HUIM/...`). The extraction script writes to its own local `csvs/`/`txt/` dirs, so **after extraction you must place/rename the outputs into `exp/Tab_exp/SHAP_HUIM/`** for ablations 02–04 to find them. That directory is not committed.
