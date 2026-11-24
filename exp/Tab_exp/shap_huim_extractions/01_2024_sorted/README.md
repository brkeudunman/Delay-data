# 01_2024_sorted — 2024 extraction, patterns ranked by numeric utility

Same mined patterns as [`01_2024_baseline`](../01_2024_baseline/), **not re-mined**:

- `shap_feature_importance.csv`, `huim_item_labels.csv`: byte-identical copies.
- `huim_delay_drivers.csv` (9,565 rows), `huim_delay_protectors.csv` (69,061 rows): the same
  rows, sorted by **numeric** `Utility` (descending, stable).

**Why.** PAMI returns utilities as strings, and `01_2024_baseline/run.py` sorted them as
text (`"99388"` > `"540144"`). The first 20 rows of its drivers file are the patterns ranked
54–73 by utility (the real #1, `O_LONGITUDE=VeryHigh` at 540,144, is at row 169); the
first 20 rows of its protectors file are ranked 8,024–8,040 of 69,061. Ablation runs with
`01_2024_baseline` before the loader fix therefore used those lower-ranked patterns, not
the top 20s.

`ablations/common/features.py` now sorts numerically on load, so the file order no longer
matters. The separate folder exists for the *name*: the runner's skip key and
`best_params.json` use the extraction folder name, so a new name gets fresh runs and keeps
the old `01_2024_baseline` rows apart. Those old rows stay in the tables as a **baseline**
("lower-ranked patterns"): same mining and training, so top-ranked vs lower-ranked
patterns isolates whether picking the highest-utility patterns matters.

Arm 02 reads only `shap_feature_importance.csv`, which is identical, so 02 is not rerun
for this extraction; `aggregate.py` uses the `01_2024_baseline` rows of 02 as its reference.
