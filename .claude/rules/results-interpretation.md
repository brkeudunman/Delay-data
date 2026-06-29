# Notes for interpreting results

- Baseline AUC is modest (~0.55–0.58): the features are only moderately predictive and the binarized target is near-balanced. SHAP+HUIM typically adds ~1–2% AUC; HUIM-only underperforms (patterns alone miss individual feature effects).
- Checkpoints land in `lightning_logs/version_N/checkpoints/` (gitignored).
- `mambular` (PyTorch Lightning under the hood) provides all tabular model classes; `pami` is the itemset-mining library; `shap` for attributions.
