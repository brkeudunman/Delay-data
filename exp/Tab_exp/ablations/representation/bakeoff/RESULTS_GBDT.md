# Bake-off with GBDT rows

50k sample, split 14690/4985/30325, seeds [42, 43, 44]; mean ± std over seeds (sample std, ddof=1, as in `../../aggregate.py`). GBDTs use one fixed configuration with early stopping on validation AUC (no search, like the fixed MLP head). The TabFM row is copied from `../skrub_head/README.md`.

The `skrub + MLP (rerun)` stds are rescaled from the 4-decimal ddof=0 values (×√1.5; per-seed values were not saved), so they are exact only to ±0.0001.

| Representation / head | dim | AUC | F1-macro | Accuracy |
|---|---|---|---|---|
| skrub + TabFM (head ablation) | 95 | 0.6336 ± 0.0001 | 0.5451 ± 0.0005 | 0.6168 ± 0.0003 |
| raw + CatBoost | 19 | 0.6009 ± 0.0019 | 0.4776 ± 0.0427 | 0.6016 ± 0.0018 |
| skrub + LightGBM | 95 | 0.5994 ± 0.0008 | 0.4933 ± 0.0281 | 0.6029 ± 0.0004 |
| skrub | 95 | 0.5982 ± 0.0017 | 0.5371 ± 0.0073 | 0.6005 ± 0.0017 |
| skrub + MLP (rerun) | 95 | 0.5982 ± 0.0018 | 0.5370 ± 0.0061 | 0.6001 ± 0.0016 |
| D_skrub_plus_huim | 135 | 0.5919 ± 0.0014 | 0.5282 ± 0.0067 | 0.5991 ± 0.0019 |
| sentence_transformer | 384 | 0.5726 ± 0.0052 | 0.4698 ± 0.0506 | 0.5943 ± 0.0019 |
