# Best results — summary

Tabular flight-delay classification on **2020** (train `FL_DAY` 1–9, val 10–12, test 13–31;
2,547,088 test rows; target `|ARR_DELAY| > 15 min`, 42.4% positive). Mean ± std over seeds
42/43/44 from the shared runner. Full tables: [`aggregate/ABLATIONS.md`](aggregate/ABLATIONS.md);
what changed and why: [`../../../change_logs/2026-11-11.md`](../../../change_logs/2026-11-11.md).

## Headline

- **Best model: XGBoost, ≈0.683 AUC** with the final fit on all of train. The top five cells
  are all XGBoost and lie within 0.0012 of each other (≈ seed noise): the feature set barely
  matters for it.
- **Best deep model: FT-Transformer on the SHAP top-10, 0.674** (all of train).
- **Data volume beats feature engineering:** going from 10% to 100% of train in the final fit
  adds 2–5 AUC points; no feature set comes close to that for the strong models.
- **HUIM patterns add nothing measurable**: 03 (SHAP + HUIM) ≈ 02 (SHAP only) for the strong
  models; HUIM alone (04) is the weakest arm (≈0.59 with 2020 patterns, ≈0.55 with 2024 patterns).
- **SHAP top-10 is the most useful selection for deep models** with full data
  (FT-T +0.023, ResNet +0.029, MLP +0.024 over all features).

## Final fit on all of train (`phase2_frac` 1.0) — recommended primary setting

Best feature set per model:

| Model | Best feature set | AUC | F1-macro | Acc. | All features (01b) | Δ vs all |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | 02 SHAP top-10 (2024 ranking) | **0.6840** ± 0.0002 | 0.6248 | 0.6423 | 0.6829 | +0.0011 |
| CatBoost | 03 SHAP + HUIM (2024 patterns) | **0.6793** ± 0.0002 | 0.6210 | 0.6386 | 0.6768 | +0.0024 |
| LightGBM | 02 SHAP top-10 (2024 ranking) | **0.6701** ± 0.0002 | 0.6102 | 0.6326 | 0.6682 | +0.0019 |
| FT-Transformer | 02 SHAP top-10 (2020) | **0.6741** ± 0.0004 | 0.6157 | 0.6338 | 0.6509 | +0.0232 |
| AutoInt | 03 SHAP + HUIM (2020) | **0.6702** ± 0.0022 | 0.6091 | 0.6327 | 0.6631 | +0.0070 |
| ResNet | 02 SHAP top-10 (2020) | **0.6469** ± 0.0042 | 0.5892 | 0.6179 | 0.6177 | +0.0292 |
| MLP | 03 SHAP + HUIM (2020) | **0.6339** ± 0.0008 | 0.5631 | 0.6098 | 0.6012 | +0.0327 |

Where 03 (SHAP + HUIM) is the nominal best (MLP, AutoInt), it is only +0.009 / +0.004 above
02 (SHAP only, same features without patterns); for ResNet the same difference is −0.011. The
gain over all features comes from the SHAP selection, not from the patterns. For the GBDTs the
best set is within 0.001–0.003 of all features.

Top 5 cells overall:

| # | Model | Feature set | AUC |
|---:|---|---|---:|
| 1 | XGBoost | 02 SHAP top-10 (2024 ranking) | 0.6840 ± 0.0002 |
| 2 | XGBoost | 03 SHAP + HUIM (2024 patterns) | 0.6833 ± 0.0001 |
| 3 | XGBoost | 03 SHAP + HUIM (2020) | 0.6830 ± 0.0006 |
| 4 | XGBoost | 01b all features | 0.6829 ± 0.0003 |
| 5 | XGBoost | 02 SHAP top-10 (2020) | 0.6828 ± 0.0006 |

## Final fit on 10% of train (`phase2_frac` 0.1, the arms' default)

Best feature set per model:

| Model | Best feature set | AUC | F1-macro | Acc. | All features (01b) | Δ vs all |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | 02 SHAP top-10 (2024 ranking) | **0.6453** ± 0.0004 | 0.5895 | 0.6168 | 0.6443 | +0.0010 |
| CatBoost | 01b all features | **0.6459** ± 0.0005 | 0.5898 | 0.6172 | 0.6459 | +0.0000 |
| LightGBM | 02 SHAP top-10 (2024 ranking) | **0.6429** ± 0.0006 | 0.5879 | 0.6149 | 0.6417 | +0.0012 |
| FT-Transformer | 01b all features | **0.6286** ± 0.0007 | 0.5777 | 0.6067 | 0.6286 | +0.0000 |
| AutoInt | 01b all features | **0.6289** ± 0.0027 | 0.5699 | 0.6079 | 0.6289 | +0.0000 |
| ResNet | 03 SHAP + HUIM (2020) | **0.5995** ± 0.0100 | 0.5457 | 0.5920 | 0.5931 | +0.0064 |
| MLP | 03 SHAP + HUIM (2020) | **0.6076** ± 0.0058 | 0.5439 | 0.5961 | 0.5906 | +0.0170 |

## Representation ablation (separate setting, not comparable in absolute terms)

50k flights from 2016–2024, split 14,690 / 4,985 / 30,325, 3 seeds
([`representation/`](representation/README.md)):

| Representation + head | Test AUC |
|---|---:|
| **skrub + TabFM (pretrained, in-context)** | **0.6336 ± 0.0001** |
| raw columns + CatBoost | 0.6009 ± 0.0016 |
| skrub + LightGBM | 0.5994 ± 0.0006 |
| skrub + MLP | 0.5982 ± 0.0014 |
| skrub + HUIM patterns + MLP | 0.5919 ± 0.0012 |

TabFM beats every trained head by ≈3 points on this sample; it has not yet been run on the
2020 ablation split.

## Caveats

- Seed std is 0.001–0.010 (up to 0.016 for some all-of-train cells); differences smaller than
  that are not meaningful, even when per-seed DeLong p-values are tiny (SE ≈ 0.0002).
- Hyper-parameters are searched on 10% of train for every arm; only the final fit uses more data.
- FT-Transformer × 03 (2020 patterns) × all of train has only 2 seeds (one seed did not
  learn) and is excluded from this summary.
- Results produced before the shared runner (`results_legacy.csv`) are not comparable.
