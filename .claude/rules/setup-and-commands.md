# Setup & Commands

Environment is managed with **uv** (Python 3.12+). The PyTorch CUDA 12.8 wheel index is pinned in `pyproject.toml`.

```bash
uv sync                 # install all dependencies into .venv
```

Run any script with `uv run python <script>` (or activate `.venv` first). Each experiment is a self-contained directory with its own `config.yaml` + `run.py`; **`run.py` resolves paths relative to its own location**, so it can be launched from anywhere:

```bash
# Tabular ablations (each reads its local config.yaml, writes results.csv)
uv run python exp/Tab_exp/ablations/01_baseline_all_features/run.py   # all raw features (control)
uv run python exp/Tab_exp/ablations/02_shap_only/run.py               # top-N SHAP features only
uv run python exp/Tab_exp/ablations/03_shap_plus_huim/run.py          # SHAP + HUIM patterns (proposed)
uv run python exp/Tab_exp/ablations/04_huim_only/run.py               # HUIM patterns only

# SHAP + HUIM pattern extraction (produces the CSVs the ablations consume)
uv run python exp/Tab_exp/shap_huim_extractions/01_2024_baseline/run.py
uv run python exp/Tab_exp/shap_huim_extractions/01_2024_baseline/visualize.py

# Other tabular benchmarks (not config-driven; constants near top of file)
uv run python exp/Tab_exp/Deep_Model_Classifier.py   # binary classification, all features
uv run python exp/Tab_exp/Deep_Model_Regressor.py    # regress exact DEP_DELAY (MSE/MAE)
uv run python exp/Tab_exp/Deep_Model_LSS.py          # distributional (location-scale) delay, NLL
```

There is **no test suite, linter, or build step** — this is a research repo. "Running" means executing an experiment `run.py` and inspecting its `results.csv` / generated figures.

Chain and network experiments are **Jupyter notebooks**, run them in Jupyter/VS Code:
- `exp/Chain_exp/`: `LSTM.ipynb`, `GRU.ipynb`, `CNN_LSTM.ipynb`, `MogrifierLSTM.ipynb`
- `exp/Network_exp/`: `AFM-node_embedding.ipynb`
