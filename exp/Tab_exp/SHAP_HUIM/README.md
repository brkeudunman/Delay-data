# SHAP × HUIM Experiment: Flight Delay Pattern Discovery

## Overview

This experiment combines **SHAP** (SHapley Additive exPlanations) with **High Utility Itemset Mining (HUIM)** to discover multi-feature delay patterns from a trained MLP classifier. Rather than examining features individually, this approach uncovers **combinations of conditions** that consistently drive or prevent flight delays.

## Method

1. **Data**: 10% sample of Flight_tab_2024.csv (~616K rows)
2. **Model**: MLP Classifier (Mambular) — binary delay prediction (|ARR_DELAY| > 15 min)
3. **SHAP**: Computed on 1,000 test instances (200 background samples)
4. **Discretization**: Continuous features → 5 quantile bins (VeryLow/Low/Mid/High/VeryHigh)
5. **HUIM**: PAMI EFIM algorithm, split into:
   - **Positive SHAP stream** → delay-driving patterns
   - **Negative SHAP stream** → delay-protecting patterns

## Results Summary

| Metric | Value |
|--------|-------|
| MLP AUC | ~0.55 |
| MLP Accuracy | ~0.59 |
| SHAP instances explained | 1,000 |
| Delay driver patterns mined | 9,566 |
| Delay protector patterns mined | 69,062 |

### SHAP Global Feature Importance

| Rank | Feature | Mean |SHAP| |
|------|---------|---------------|
| 1 | O_LONGITUDE | 0.110 |
| 2 | OP_CARRIER | 0.065 |
| 3 | CRS_ELAPSED_TIME | 0.063 |
| 4 | O_TEMP | 0.050 |
| 5 | OP_CARRIER_FL_NUM | 0.047 |
| 6 | CRS_DEP_TIME_MIN | 0.044 |

### Key Delay Driver Patterns

Patterns pushing predictions **toward delay**:

| Pattern | Utility | Interpretation |
|---------|---------|----------------|
| O_Longitude=VeryHigh | 540K | Northeast US origins dominate delays |
| O_Lat=High + O_Lon=VeryHigh | 463K | NYC area airports |
| Origin=JFK + O_Lon=VeryHigh | 98K | JFK-specific delay pattern |
| Carrier=DL + O_Lon=VeryHigh | 98K | Delta at NE airports |
| Elapsed=VeryHigh + O_Wind=VeryHigh + O_Lon=VeryHigh | 96K | Long-haul + high wind from NE |
| DepTime=High + D_Temp=VeryHigh + O_Lon=VeryHigh | 95K | Late departure + heat + NE origin |

### Key Delay Protector Patterns

Patterns pushing predictions **away from delay**:

| Pattern | Utility | Interpretation |
|---------|---------|----------------|
| O_Temp=Mid | 96K | Moderate origin temperature |
| O_Longitude=High | 98K | Non-extreme longitude (Midwest) |
| Short flight + regional carrier (WN/OO) + Low precip | ~10K | Simple, short, clear-weather routes |
| VeryLow departure time + Low wind + Low precip | ~10K | Early morning, calm weather |
| Moderate climate + Low precipitation at both ends | ~10K | Clear weather at origin and destination |

## Key Findings

1. **Geography is the strongest signal**: Flights from Northeast US airports (O_LONGITUDE=VeryHigh) are the dominant delay driver, appearing in nearly every top pattern.

2. **Weather amplifies in combination**: Wind speed and temperature alone have moderate impact, but combined with geography and scheduling they significantly increase delay risk.

3. **Carrier-specific patterns emerge**: Delta (DL) and American (AA) at Northeast airports are delay drivers; Southwest (WN) and SkyWest (OO) on short regional routes are protectors — reflecting different hub strategies.

4. **Actionable size**: The most interpretable patterns contain 2–4 features — specific enough to be actionable, general enough to cover many flights.

5. **Positive vs negative asymmetry**: There are ~7× more protector patterns (69K) than driver patterns (9.5K), suggesting on-time performance has more diffuse causes while delay concentration is driven by fewer, stronger factors.

## Files

| File | Description |
|------|-------------|
| `huim_delay_drivers.csv` | All mined delay-driving patterns |
| `huim_delay_protectors.csv` | All mined delay-protecting patterns |
| `shap_feature_importance.csv` | SHAP global feature ranking |
| `huim_item_labels.csv` | Item ID → human-readable label mapping |
| `shap_utility_positive.txt` | HUIM transaction DB (positive SHAP) |
| `shap_utility_negative.txt` | HUIM transaction DB (negative SHAP) |
| `visualize_results.py` | Generates all figures |
| `figures/` | Generated visualization figures |
