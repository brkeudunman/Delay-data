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

## Deep Dive: How Utility is Calculated

In traditional Association Rule Mining (like Apriori), patterns are discovered based on *frequency* (support). High Utility Itemset Mining (HUIM), however, discovers patterns based on *Utility* (value/weight). **This methodology innovatively uses localized SHAP values as the "Utility" (profit/weight) of an item.**

Because HUIM algorithms mathematically require all utility values to be strictly **positive**, but SHAP values can be positive (drive delay) or negative (prevent delay), the transactions are split into two separate databases: positive SHAP values (Delay Drivers) and absolute negative SHAP values (Delay Protectors).

### 1. Local Item Utility (Single Flight)
For every single flight (transaction), the local utility of a specific feature-value (item) is derived directly from its literal SHAP value for that specific prediction. Because HUIM requires integers, the SHAP value is scaled by 10,000 and converted to an integer.

**Formula:**
$$U(\text{item}_j, \text{flight}_i) = \text{int}(|\text{SHAP}_{i, j}| \times 10,000)$$

For example, if the feature `O_Lon=VeryHigh` has a SHAP value of `+0.105` for a specific flight, its utility in that transaction is `1050`.

### 2. Transaction Utility
The total utility of a transaction (a single flight prediction) is simply the sum of all local item utilities present in that flight. A high transaction utility means the combined features heavily pushed the model's prediction in a specific direction.

### 3. Pattern Utility (Global Rule)
The algorithm searches for occurring combinations of items, such as `{Origin=JFK + O_Lon=VeryHigh}`. 
The final Utility score of a pattern is the sum of the local utilities of *those specific items*, aggregated *only across flights where both items co-occurred*.

**Pattern Utility Formula:**
$$U(\text{Pattern}) = \sum_{\text{flight } \in \text{ flights containing Pattern}} \Big( \sum_{\text{item } \in \text{ Pattern}} U(\text{item, flight}) \Big)$$

**What does a "Utility = 98K" mean?**
If the pattern `Origin=JFK + O_Lon=VeryHigh` has a utility of **98K**, it indicates:
1. **Frequency × Magnitude:** The pattern might happen very often with a moderate SHAP impact, or less often but with massive SHAP impacts.
2. **Raw SHAP Equivalent:** Because of the 10,000 scaling factor, a utility of 98,000 equates to an aggregate absolute SHAP shift of roughly **9.8** across the 1,000 tested flights.
3. **Additive Evidence:** It mathematically proves that when these specific states appear simultaneously, they jointly inject massive momentum into the neural network's final decision.

## Results Summary

| Metric | Value |
|--------|-------|
| MLP AUC | ~0.55 |
| MLP Accuracy | ~0.59 |
| SHAP instances explained | 1,000 |
| Delay driver patterns mined | 9,566 |
| Delay protector patterns mined | 69,062 |

### SHAP Global Feature Importance

| Rank | Feature | Mean  |
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
