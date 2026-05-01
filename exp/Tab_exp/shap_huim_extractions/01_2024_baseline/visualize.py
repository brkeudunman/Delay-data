"""
visualize_results.py
====================
Generate publication-ready figures for the SHAP → HUIM experiment.

Produces:
  1. SHAP global feature importance bar chart
  2. Top-15 delay driver patterns (horizontal bar)
  3. Top-15 delay protector patterns (horizontal bar)
  4. Pattern size distribution (drivers vs protectors)
  5. Combined side-by-side comparison figure
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
importance_df = pd.read_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"))
drivers_df = pd.read_csv(os.path.join(RESULTS_DIR, "huim_delay_drivers.csv"))
protectors_df = pd.read_csv(os.path.join(RESULTS_DIR, "huim_delay_protectors.csv"))

print(f"Loaded: {len(drivers_df)} driver patterns, {len(protectors_df)} protector patterns")

# ============================================================
# Helper: shorten pattern labels for readability
# ============================================================
def shorten_pattern(p, max_items=4):
    """Shorten long pattern strings for chart labels."""
    parts = p.split(" + ")
    # Shorten individual items: "CRS_DEP_TIME_MIN=VeryHigh" -> "DEP_TIME=VHigh"
    short = []
    for part in parts[:max_items]:
        part = part.replace("CRS_DEP_TIME_MIN", "DepTime")
        part = part.replace("CRS_ARR_TIME_MIN", "ArrTime")
        part = part.replace("CRS_ELAPSED_TIME", "Elapsed")
        part = part.replace("ORIGIN_INDEX", "Origin")
        part = part.replace("DEST_INDEX", "Dest")
        part = part.replace("OP_CARRIER", "Carrier")
        part = part.replace("O_LONGITUDE", "O_Lon")
        part = part.replace("O_LATITUDE", "O_Lat")
        part = part.replace("D_LONGITUDE", "D_Lon")
        part = part.replace("D_LATITUDE", "D_Lat")
        part = part.replace("O_TEMP", "O_Temp")
        part = part.replace("D_TEMP", "D_Temp")
        part = part.replace("O_WSPD", "O_Wind")
        part = part.replace("D_WSPD", "D_Wind")
        part = part.replace("O_PRCP", "O_Precip")
        part = part.replace("D_PRCP", "D_Precip")
        part = part.replace("FL_WEEK", "Week")
        part = part.replace("FL_DAY", "Day")
        part = part.replace("VeryHigh", "VH")
        part = part.replace("VeryLow", "VL")
        short.append(part)
    label = " + ".join(short)
    if len(parts) > max_items:
        label += f" +{len(parts) - max_items} more"
    return label


# ============================================================
# Figure 1: SHAP Feature Importance
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
data = importance_df.sort_values("Mean|SHAP|", ascending=True)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(data)))
ax.barh(data["Feature"], data["Mean|SHAP|"], color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("Global Feature Importance (SHAP)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_shap_feature_importance.png"))
plt.close()
print("  Saved: 01_shap_feature_importance.png")


# ============================================================
# Figure 2: Top-15 Delay Driver Patterns
# ============================================================
# Filter to multi-item patterns (size >= 2) for more interesting results
drivers_multi = drivers_df[drivers_df["Size"] >= 2].head(15).copy()
drivers_multi["Label"] = drivers_multi["Pattern"].apply(shorten_pattern)
drivers_multi = drivers_multi.sort_values("Utility", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_d = plt.cm.Reds(np.linspace(0.4, 0.85, len(drivers_multi)))
bars = ax.barh(drivers_multi["Label"], drivers_multi["Utility"], color=colors_d,
               edgecolor="white", linewidth=0.5)
ax.set_xlabel("Cumulative Utility (SHAP × 10,000)")
ax.set_title("Top 15 Delay Driver Patterns (SHAP + HUIM)", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add size annotations
for bar, size in zip(bars, drivers_multi["Size"]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f"k={size}", va="center", fontsize=9, color="#555")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_top15_delay_drivers.png"))
plt.close()
print("  Saved: 02_top15_delay_drivers.png")


# ============================================================
# Figure 3: Top-15 Delay Protector Patterns
# ============================================================
protectors_multi = protectors_df[protectors_df["Size"] >= 2].head(15).copy()
protectors_multi["Label"] = protectors_multi["Pattern"].apply(shorten_pattern)
protectors_multi = protectors_multi.sort_values("Utility", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_p = plt.cm.Blues(np.linspace(0.4, 0.85, len(protectors_multi)))
bars = ax.barh(protectors_multi["Label"], protectors_multi["Utility"], color=colors_p,
               edgecolor="white", linewidth=0.5)
ax.set_xlabel("Cumulative Utility (|SHAP| × 10,000)")
ax.set_title("Top 15 Delay Protector Patterns (SHAP + HUIM)", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, size in zip(bars, protectors_multi["Size"]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f"k={size}", va="center", fontsize=9, color="#555")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_top15_delay_protectors.png"))
plt.close()
print("  Saved: 03_top15_delay_protectors.png")


# ============================================================
# Figure 4: Pattern Size Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Drivers
size_counts_d = drivers_df["Size"].value_counts().sort_index()
axes[0].bar(size_counts_d.index, size_counts_d.values, color="#d32f2f", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Pattern Size (k)")
axes[0].set_ylabel("Number of Patterns")
axes[0].set_title("Delay Drivers", fontweight="bold", color="#d32f2f")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Protectors
size_counts_p = protectors_df["Size"].value_counts().sort_index()
axes[1].bar(size_counts_p.index, size_counts_p.values, color="#1565c0", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Pattern Size (k)")
axes[1].set_ylabel("Number of Patterns")
axes[1].set_title("Delay Protectors", fontweight="bold", color="#1565c0")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Pattern Size Distribution: Drivers vs Protectors", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_pattern_size_distribution.png"))
plt.close()
print("  Saved: 04_pattern_size_distribution.png")


# ============================================================
# Figure 5: Side-by-side top-10 comparison
# ============================================================
top_d = drivers_df[drivers_df["Size"] >= 2].head(10).copy()
top_p = protectors_df[protectors_df["Size"] >= 2].head(10).copy()
top_d["Label"] = top_d["Pattern"].apply(lambda x: shorten_pattern(x, 3))
top_p["Label"] = top_p["Pattern"].apply(lambda x: shorten_pattern(x, 3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Drivers (left, red)
top_d_sorted = top_d.sort_values("Utility", ascending=True)
ax1.barh(top_d_sorted["Label"], top_d_sorted["Utility"],
         color=plt.cm.Reds(np.linspace(0.35, 0.8, len(top_d_sorted))),
         edgecolor="white", linewidth=0.5)
ax1.set_xlabel("Utility")
ax1.set_title("⚠ Delay Drivers", fontweight="bold", fontsize=14, color="#c62828")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Protectors (right, blue)
top_p_sorted = top_p.sort_values("Utility", ascending=True)
ax2.barh(top_p_sorted["Label"], top_p_sorted["Utility"],
         color=plt.cm.Blues(np.linspace(0.35, 0.8, len(top_p_sorted))),
         edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Utility")
ax2.set_title("✓ Delay Protectors", fontweight="bold", fontsize=14, color="#1565c0")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("SHAP × HUIM: Delay Drivers vs Protectors (Top 10, k≥2)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_drivers_vs_protectors.png"))
plt.close()
print("  Saved: 05_drivers_vs_protectors.png")


# ============================================================
# Print summary stats
# ============================================================
print(f"\n{'='*60}")
print("  VISUALIZATION COMPLETE")
print(f"{'='*60}")
print(f"  Figures saved to: {FIG_DIR}")
print(f"  Total driver patterns:   {len(drivers_df):,}")
print(f"  Total protector patterns: {len(protectors_df):,}")
print(f"  Driver size range:   {drivers_df['Size'].min()}-{drivers_df['Size'].max()}")
print(f"  Protector size range: {protectors_df['Size'].min()}-{protectors_df['Size'].max()}")
