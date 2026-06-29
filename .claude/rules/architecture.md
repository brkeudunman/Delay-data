# Architecture

## `Datasets/` — data construction (raw BTS CSV → model-ready representations)
- `Flight_tab.py` — tabular representation (categorical: carrier, airport indices, calendar; continuous: weather, coordinates, scheduled times).
- `Flight_chain.py` — flight chains linking consecutive flights by aircraft/crew, for temporal delay propagation.
- `Flight_networks.py` — DGL graphs: airports as nodes, flights as edges, for network modeling.
- `data_extract.py`, `data_pro.py` — ETL: cleaning, outlier removal, datetime→minutes, feature engineering.

## `exp/Tab_exp/` — tabular experiments (the actively developed area)
- `Deep_Model_Classifier.py` / `Deep_Model_Regressor.py` / `Deep_Model_LSS.py` — standalone benchmarks of all `mambular` models for classification / regression / distributional tasks.
- `ablations/` — four feature-set ablations (`01_baseline_all_features`, `02_shap_only`, `03_shap_plus_huim`, `04_huim_only`), each `config.yaml` + `run.py` + `results.csv`.
- `shap_huim_extractions/` — the SHAP+HUIM mining pipeline. `01_2024_baseline/` and `01_2024_v2/` are variants; each writes `csvs/`, `txt/`, `figures/`.

## `util/` — analysis & figures
- `check/shap_check.py` — recompute SHAP global feature-importance ranking.
- `check/mi.py` — mutual-information ranking. ⚠️ contains **hardcoded stale `D:\project\...` paths**; fix paths before use.
- `figures/` — plotting scripts: `propagation.py` (delay-propagation Sankey), `month_delay.py`, `geo_delay.py`, `ts_delay.py` (temporal/geographic trends), `Table_delay.py`, `airport.py`, `radar.py`. Output PNGs live in `util/figures/<subdir>/`.
