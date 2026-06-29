# Data & training conventions (must preserve)

- **Two dataset years — don't conflate them.** The **ablations** train/evaluate on **2020** (`Flight_tab_2020.csv`, `data_info_2020.yaml`). The **SHAP/HUIM extraction** mines patterns from **2024** (`Flight_tab_2024.csv`, `year: 2024` in its config). Both files live in `Datasets/Aeolus/Flight_Tab/Tab/`.
- **Temporal split by day-of-month — never shuffle across days** (prevents leakage): train `FL_DAY ≤ 9`, val `10 ≤ FL_DAY ≤ 12`, test `FL_DAY > 12` (`train_days_max` / `val_days_max` in config).
- **Excluded columns:** `FLIGHTS`, `FL_YEAR`, `FL_MONTH` (not meaningful at flight level).
- **Encoding:** `LabelEncoder` for categoricals; `StandardScaler` for continuous, **fit on train split only**.
- **Target:** `ARR_DELAY` binarized as `|delay| > 15 min`.
- Feature lists (categorical vs. continuous) come from `data_info_<year>.yaml`, not hardcoded — verify the YAML matches the CSV columns when debugging load errors.

## Two-phase training (the `run.py` pattern)
- **Phase 1 — hyperparameter search:** 10% subsample (`phase1_frac`), `RandomizedSearchCV` (`cv=3`, `n_iter=10`) over `d_model ∈ {64,128,256}`, `n_layers ∈ {2,6,10}`, `lr ∈ {1e-5,1e-4,1e-3}`.
- **Phase 2 — final training:** best hyperparameters, retrain on `phase2_frac` of train, monitor on `val_frac` of val, tighter early stopping. GPU + `16-mixed` precision.
- **Evaluation:** AUC-ROC + Accuracy on the held-out test set → `results.csv`.

**Per-ablation configs differ** — don't assume the values are uniform. E.g. `01_baseline` uses `phase2_frac: 1.0`, `batch_size: 4096`, `patience_phase2: 3`; `03_shap_plus_huim` uses `phase2_frac: 0.1`, `val_frac: 0.1`, `batch_size: 2048`, `patience_phase2: 5`. Always read the target ablation's `config.yaml`.

## Config shape (`config.yaml`)
```yaml
experiment: { name, description }
data:       { year, file, yaml_info, train_days_max, val_days_max }   # + sample_frac for extraction
training:   { phase1_frac, phase2_frac, val_frac, max_epochs,
              patience_phase1, patience_phase2, batch_size, precision, accelerator,
              n_iter, cv, d_model_options, n_layers_options, lr_options }
features:   { mode: all|shap_only|shap_plus_huim|huim_only, exclude_cols,
              top_shap_n, top_drivers_n, top_protectors_n, n_bins,
              shap_importance_file, drivers_file, protectors_file }   # files only for shap/huim modes
paths:      { data_dir }      # absolute, points at Datasets/Aeolus/Flight_Tab/Tab
```
