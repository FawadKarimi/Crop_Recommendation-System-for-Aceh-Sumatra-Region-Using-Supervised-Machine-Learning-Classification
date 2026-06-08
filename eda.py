"""
eda.py  —  Exploratory Data Analysis
MMAI1002 Final Project: Crop Recommendation for Aceh/Sumatra
--------------------------------------------------------------
Produces all figures needed for Section 3 (Dataset & Features) of the manuscript.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
FIG_DIR    = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEAT_LABELS = {
    "N":           "Nitrogen (mg/kg)",
    "P":           "Phosphorus (mg/kg)",
    "K":           "Potassium (mg/kg)",
    "temperature": "Temperature (°C)",
    "humidity":    "Humidity (%)",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall (mm/month)",
}
PALETTE = "Set2"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "aceh_crop_data.csv")
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(df["label"].value_counts().to_string())

# ── Figure 1: Class distribution bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
counts = df["label"].value_counts().sort_values(ascending=False)
colors = sns.color_palette(PALETTE, len(counts))
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=10)
ax.set_title("Figure 1 — Class distribution (crop labels)", fontsize=13, fontweight="bold")
ax.set_xlabel("Crop")
ax.set_ylabel("Number of samples")
ax.set_ylim(0, counts.max() * 1.15)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_class_distribution.png", dpi=200)
plt.close()
print("Saved: fig1_class_distribution.png")

# ── Figure 2: Feature distributions (violin plots) ───────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    ax.violinplot(
        [df[df["label"] == c][feat].values for c in counts.index],
        positions=range(len(counts)), widths=0.7,
        showmedians=True, showextrema=False
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(FEAT_LABELS[feat], fontsize=10)
    ax.set_ylabel(feat)
axes[-1].set_visible(False)
fig.suptitle("Figure 2 — Feature distributions per crop class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_feature_distributions.png", dpi=200)
plt.close()
print("Saved: fig2_feature_distributions.png")

# ── Figure 3: Correlation heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.4, ax=ax, square=True,
            xticklabels=[FEAT_LABELS[f].split("(")[0].strip() for f in FEATURES],
            yticklabels=[FEAT_LABELS[f].split("(")[0].strip() for f in FEATURES])
ax.set_title("Figure 3 — Feature correlation matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_correlation_heatmap.png", dpi=200)
plt.close()
print("Saved: fig3_correlation_heatmap.png")

# ── Figure 4: Aceh climate context (rainfall + temperature by month) ──────────
# Real BMKG/climate data for Banda Aceh (climatestotravel.com / weather-and-climate.com)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
rainfall_mm   = [255, 111, 135, 145, 120, 115,  95, 105, 125, 210, 362, 290]  # mm/month
temp_avg_c    = [26.9, 27.1, 27.5, 27.8, 28.0, 28.2, 27.9, 27.8, 27.6, 27.2, 26.8, 26.7]

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

bars = ax1.bar(months, rainfall_mm, color="#5DCAA5", alpha=0.75, label="Rainfall (mm)")
ax1.set_ylabel("Monthly rainfall (mm)", color="#0F6E56")
ax1.tick_params(axis="y", labelcolor="#0F6E56")
ax1.set_ylim(0, max(rainfall_mm)*1.25)

line, = ax2.plot(months, temp_avg_c, "o-", color="#D85A30", linewidth=2,
                  markersize=6, label="Avg temperature (°C)")
ax2.set_ylabel("Average temperature (°C)", color="#D85A30")
ax2.tick_params(axis="y", labelcolor="#D85A30")
ax2.set_ylim(24, 32)

handles = [
    mpatches.Patch(color="#5DCAA5", label="Rainfall (mm)"),
    plt.Line2D([0],[0], color="#D85A30", marker="o", label="Temperature (°C)")
]
ax1.legend(handles=handles, loc="upper left", fontsize=9)
ax1.set_title("Figure 4 — Banda Aceh monthly climate (BMKG data)\n"
              "Source: BMKG / climatestotravel.com",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Month")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_aceh_climate_context.png", dpi=200)
plt.close()
print("Saved: fig4_aceh_climate_context.png")

# ── Figure 5: Feature means radar chart per selected crop ─────────────────────
from matplotlib.patches import FancyArrowPatch

selected = ["rice", "coffee", "oilpalm", "coconut", "maize"]
norm_feats = ["N", "P", "K", "temperature", "humidity", "rainfall"]

# Normalise to 0–1 for radar
means = df.groupby("label")[norm_feats].mean()
mins  = df[norm_feats].min()
maxs  = df[norm_feats].max()
means_norm = (means - mins) / (maxs - mins)

N_feat = len(norm_feats)
angles = np.linspace(0, 2 * np.pi, N_feat, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_r = sns.color_palette(PALETTE, len(selected))
for i, crop in enumerate(selected):
    vals = means_norm.loc[crop].tolist() + [means_norm.loc[crop].tolist()[0]]
    ax.plot(angles, vals, "o-", linewidth=1.8, color=colors_r[i], label=crop)
    ax.fill(angles, vals, alpha=0.08, color=colors_r[i])

ax.set_thetagrids(np.degrees(angles[:-1]), norm_feats)
ax.set_title("Figure 5 — Normalised feature profile per crop\n"
             "(selected Aceh-relevant crops)", pad=20,
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_radar_crop_profiles.png", dpi=200)
plt.close()
print("Saved: fig5_radar_crop_profiles.png")

print(f"\nAll 5 figures saved to: {FIG_DIR}")

# ── Print summary statistics table for manuscript ─────────────────────────────
print("\n-- Feature summary statistics --")
print(df[FEATURES].describe().round(2).to_string())
