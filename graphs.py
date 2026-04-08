import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import optuna
from optuna.distributions import FloatDistribution, IntDistribution

# ── Setup Output Directory ─────────────────────────────────────────────────
OUTPUT_DIR = "v3_results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "savefig.facecolor": "white",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

ACCENT   = "#2563EB"   # blue
ORANGE   = "#F97316"   # orange
GREEN    = "#16A34A"
RED      = "#DC2626"
PURPLE   = "#7C3AED"
GREY     = "#6B7280"
CMAP     = "viridis"

df = pd.read_csv("v3_results/4player_optuna_results_v3.csv")

# Derived columns
df["best_so_far"] = df["Optuna_Composite_Score"].cummax()
total_games = df[["Heuristic_Wins", "Vanilla_Wins", "MCTS_VARIANT", "MCTS_FINAL", "Draws"]].sum(axis=1)
df["target_win_rate"] = df["MCTS_FINAL"] / total_games
df["score_check"] = (df["Avg_Survival_Score"] * 0.8 + df["Avg_Length_Score"] * 0.2) + df["target_win_rate"] * 0.2

PARAMS = ["C_PARAM", "DEPTH_LIMIT", "PB_WEIGHT", "TARGET_LENGTH"]
param_labels = {
    "C_PARAM": "C Parameter",
    "DEPTH_LIMIT": "Depth Limit",
    "PB_WEIGHT": "PB Weight",
    "TARGET_LENGTH": "Target Length",
}

# ── Reconstruct Study & Calculate fANOVA Importance ────────────────────────
print("Calculating fANOVA importances... (This might take a moment)")

# 1. Infer distributions from data bounds
distributions = {
    "C_PARAM": FloatDistribution(df["C_PARAM"].min(), df["C_PARAM"].max()),
    "DEPTH_LIMIT": IntDistribution(int(df["DEPTH_LIMIT"].min()), int(df["DEPTH_LIMIT"].max())),
    "PB_WEIGHT": FloatDistribution(df["PB_WEIGHT"].min(), df["PB_WEIGHT"].max()),
    "TARGET_LENGTH": IntDistribution(int(df["TARGET_LENGTH"].min()), int(df["TARGET_LENGTH"].max())),
}

# 2. Create a mock in-memory study
study = optuna.create_study(direction="maximize")

# 3. Populate study with trials from the CSV
for _, row in df.iterrows():
    trial = optuna.trial.create_trial(
        params={
            "C_PARAM": float(row["C_PARAM"]),
            "DEPTH_LIMIT": int(row["DEPTH_LIMIT"]),
            "PB_WEIGHT": float(row["PB_WEIGHT"]),
            "TARGET_LENGTH": int(row["TARGET_LENGTH"]),
        },
        distributions=distributions,
        value=float(row["Optuna_Composite_Score"]),
    )
    study.add_trial(trial)

# 4. Calculate fANOVA importances
fanova_evaluator = optuna.importance.FanovaImportanceEvaluator()
importances = optuna.importance.get_param_importances(study, evaluator=fanova_evaluator)


# ── Helpers ────────────────────────────────────────────────────────────────
def scatter_with_score(ax, x, y, score, xlabel, ylabel, title, annotate_best=True):
    norm = Normalize(score.min(), score.max())
    sc = ax.scatter(x, y, c=score, cmap=CMAP, s=55, alpha=0.85,
                    edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="Optuna Score", pad=0.02)
    if annotate_best:
        best_idx = score.idxmax()
        ax.annotate(f"Best\n#{best_idx}",
                    xy=(x[best_idx], y[best_idx]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7.5, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Optimisation History
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.scatter(df["Trial_Number"], df["Optuna_Composite_Score"],
           s=45, color=ACCENT, alpha=0.7, zorder=3, label="Trial Score")
ax.plot(df["Trial_Number"], df["best_so_far"],
        color=ORANGE, lw=2, label="Best So Far", zorder=4)
best_row = df.loc[df["Optuna_Composite_Score"].idxmax()]
ax.scatter(best_row["Trial_Number"], best_row["Optuna_Composite_Score"],
           s=120, color=ORANGE, edgecolors="black", linewidths=1.2,
           zorder=5, label=f"Best (Trial {int(best_row['Trial_Number'])})")
ax.set_xlabel("Trial Number")
ax.set_ylabel("Optuna Composite Score")
ax.set_title("Optimisation History")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_optimisation_history.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig1_optimisation_history.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Score Decomposition: Survival vs Length (bubble = composite)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
norm = Normalize(df["Optuna_Composite_Score"].min(), df["Optuna_Composite_Score"].max())
sc = ax.scatter(
    df["Avg_Survival_Score"], df["Avg_Length_Score"],
    c=df["Optuna_Composite_Score"], cmap=CMAP,
    s=df["Optuna_Composite_Score"] * 600,
    alpha=0.78, edgecolors="white", linewidths=0.6, zorder=3
)
plt.colorbar(sc, ax=ax, label="Optuna Composite Score")

top5 = df.nlargest(5, "Optuna_Composite_Score")
for _, row in top5.iterrows():
    ax.annotate(f"#{int(row['Trial_Number'])}",
                xy=(row["Avg_Survival_Score"], row["Avg_Length_Score"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8)

ax.set_xlabel("Avg Survival Score  (weight = 0.80)")
ax.set_ylabel("Avg Length Score  (weight = 0.20)")
ax.set_title("Score Decomposition\nSurvival vs Length (bubble size = Composite Score)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_score_decomposition.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig2_score_decomposition.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Hyperparameter vs Optuna Score (2×2)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, param in zip(axes.flat, PARAMS):
    ax.scatter(df[param], df["Optuna_Composite_Score"],
               c=df["Optuna_Composite_Score"], cmap=CMAP,
               s=50, alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)
    z = np.polyfit(df[param], df["Optuna_Composite_Score"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df[param].min(), df[param].max(), 200)
    ax.plot(xs, p(xs), color=ORANGE, lw=1.5, linestyle="--", alpha=0.8, label="Trend")
    r = np.corrcoef(df[param], df["Optuna_Composite_Score"])[0, 1]
    ax.set_title(f"{param_labels[param]}", fontsize=10, fontweight="bold")
    ax.set_xlabel(param_labels[param], fontsize=12)
    ax.set_ylabel("Optuna Composite Score", fontsize=12)
    ax.legend(fontsize=8)
fig.suptitle("Hyperparameters vs Objective Score", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_param_vs_score.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig3_param_vs_score.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════
cols_corr = PARAMS + ["Avg_Survival_Score", "Avg_Length_Score", "Optuna_Composite_Score"]
corr = df[cols_corr].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")
labels = [param_labels.get(c, c.replace("_", " ")) for c in cols_corr]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.5, color=color)
ax.set_title("Pearson Correlation Matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_correlation_heatmap.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig4_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Win Distribution Stacked Bar (top 20 trials by score)
# ══════════════════════════════════════════════════════════════════════════
top20 = df.nlargest(20, "Optuna_Composite_Score").sort_values("Trial_Number")
win_cols = ["MCTS_FINAL", "Heuristic_Wins", "Vanilla_Wins", "MCTS_VARIANT", "Draws"]
colors_bar = [ACCENT, GREEN, RED, PURPLE, GREY]
labels_bar = ["MCTS Final", "Heuristic", "Vanilla", "MCTS Variant", "Draws"]

fig, ax = plt.subplots(figsize=(14, 5))
bottoms = np.zeros(len(top20))
for col, color, label in zip(win_cols, colors_bar, labels_bar):
    ax.bar(top20["Trial_Number"].astype(str), top20[col],
           bottom=bottoms, color=color, label=label, width=0.7)
    bottoms += top20[col].values

ax.set_xlabel("Trial Number")
ax.set_ylabel("Games")
ax.set_title("Win Distribution — Top 20 Trials by Optuna Score")
ax.legend(loc="upper right", ncol=5, framealpha=0.95)
ax.xaxis.set_tick_params(rotation=45)
ax2 = ax.twinx()
ax2.plot(range(len(top20)), top20["Optuna_Composite_Score"].values,
         color=ORANGE, lw=2, marker="o", markersize=5, label="Composite Score")
ax2.set_ylabel("Optuna Composite Score", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.spines["right"].set_edgecolor(ORANGE)
ax2.set_ylim(0, top20["Optuna_Composite_Score"].max() * 1.5)
ax2.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig5_win_distribution.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig5_win_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — Hyperparameter Distributions (best vs rest)
# ══════════════════════════════════════════════════════════════════════════
threshold = df["Optuna_Composite_Score"].quantile(0.75)
df["tier"] = np.where(df["Optuna_Composite_Score"] >= threshold, "Top 25%", "Rest")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, param in zip(axes.flat, PARAMS):
    best = df[df["tier"] == "Top 25%"][param]
    rest = df[df["tier"] == "Rest"][param]
    bins = 12
    ax.hist(rest, bins=bins, color=GREY, alpha=0.55, label="Rest", density=True)
    ax.hist(best, bins=bins, color=ACCENT, alpha=0.75, label="Top 25%", density=True)
    ax.axvline(best.mean(), color=ACCENT, lw=2, linestyle="--",
               label=f"Top mean: {best.mean():.2f}")
    ax.axvline(rest.mean(), color=GREY, lw=2, linestyle="--",
               label=f"Rest mean: {rest.mean():.2f}")
    ax.set_xlabel(param_labels[param])
    ax.set_ylabel("Density")
    ax.set_title(param_labels[param])
    ax.legend(fontsize=7.5)
fig.suptitle("Hyperparameter Distributions: Top 25% vs Rest", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_param_distributions.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig6_param_distributions.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Parallel Coordinates (top 15 highlighted)
# ══════════════════════════════════════════════════════════════════════════
plot_cols = PARAMS + ["Avg_Survival_Score", "Avg_Length_Score", "Optuna_Composite_Score"]
pc_labels = [param_labels.get(c, c.replace("_", " ").title()) for c in plot_cols]

normed = df[plot_cols].copy()
for col in plot_cols:
    normed[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

top15_idx = df.nlargest(15, "Optuna_Composite_Score").index
norm_score = Normalize(df["Optuna_Composite_Score"].min(), df["Optuna_Composite_Score"].max())

fig, ax = plt.subplots(figsize=(13, 5.5))
xs = range(len(plot_cols))

for i, row in normed.iterrows():
    score = df.loc[i, "Optuna_Composite_Score"]
    is_top = i in top15_idx
    color = plt.cm.viridis(norm_score(score))
    lw = 2.2 if is_top else 0.6
    alpha = 0.9 if is_top else 0.2
    ax.plot(xs, row[plot_cols].values, color=color, lw=lw, alpha=alpha)

ax.set_xticks(xs)
ax.set_xticklabels(pc_labels, rotation=25, ha="right")
ax.set_ylabel("Normalised Value")
ax.set_title("Parallel Coordinates — Highlighted: Top 15 Trials")
sm = ScalarMappable(cmap=CMAP, norm=norm_score)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Optuna Composite Score", pad=0.01)
for x in xs:
    ax.axvline(x, color="#dddddd", lw=0.8, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig7_parallel_coords.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig7_parallel_coords.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 8 — Score Component Breakdown: Top 10 Trials
# ══════════════════════════════════════════════════════════════════════════
top10 = df.nlargest(10, "Optuna_Composite_Score").sort_values("Optuna_Composite_Score", ascending=True)
y = np.arange(len(top10))
bar_h = 0.28

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(y + bar_h, top10["Avg_Survival_Score"] * 0.8, bar_h,
        color=ACCENT, label="Survival × 0.8", alpha=0.9)
ax.barh(y,         top10["Avg_Length_Score"] * 0.2, bar_h,
        color=GREEN, label="Length × 0.2", alpha=0.9)
ax.barh(y - bar_h, top10["target_win_rate"] * 0.2, bar_h,
        color=ORANGE, label="Win Rate × 0.2", alpha=0.9)
ax.plot(top10["Optuna_Composite_Score"].values, y,
        "D", color="black", markersize=7, label="Composite Score", zorder=5)

ax.set_yticks(y)
ax.set_yticklabels([f"Trial {int(t)}" for t in top10["Trial_Number"]])
ax.set_xlabel("Score Contribution")
ax.set_title("Score Component Breakdown — Top 10 Trials")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig8_score_components.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig8_score_components.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 9 — Depth Limit & C_PARAM 2D heatmap (score as colour)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pairs = [
    ("C_PARAM",   "DEPTH_LIMIT", "C Parameter",  "Depth Limit"),
    ("PB_WEIGHT", "TARGET_LENGTH","PB Weight",    "Target Length"),
]
for ax, (px, py, lx, ly) in zip(axes, pairs):
    scatter_with_score(
        ax, df[px], df[py], df["Optuna_Composite_Score"],
        lx, ly,
        f"Optuna Score: {lx} vs {ly}"
    )

fig.suptitle("2D Hyperparameter Search Space (colour = Composite Score)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig9_2d_search_space.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig9_2d_search_space.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 10 — Summary dashboard (mini panels)
# ══════════════════════════════════════════════════════════════════════════
best = df.loc[df["Optuna_Composite_Score"].idxmax()]

fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

ax_hist = fig.add_subplot(gs[0, :])
ax_hist.scatter(df["Trial_Number"], df["Optuna_Composite_Score"],
                s=30, color=ACCENT, alpha=0.6, zorder=3, label="Trial")
ax_hist.plot(df["Trial_Number"], df["best_so_far"],
             color=ORANGE, lw=1.8, label="Best so far")
ax_hist.scatter(best["Trial_Number"], best["Optuna_Composite_Score"],
                s=100, color=ORANGE, edgecolors="black", lw=1, zorder=5)
ax_hist.set_title("Optimisation History")
ax_hist.set_xlabel("Trial")
ax_hist.set_ylabel("Score")
ax_hist.legend(fontsize=8)

ax_sl = fig.add_subplot(gs[1, :2])
sc = ax_sl.scatter(df["Avg_Survival_Score"], df["Avg_Length_Score"],
                   c=df["Optuna_Composite_Score"], cmap=CMAP, s=40, alpha=0.8,
                   edgecolors="white", lw=0.4)
plt.colorbar(sc, ax=ax_sl, label="Score", pad=0.02)
ax_sl.set_xlabel("Survival (w=0.8)")
ax_sl.set_ylabel("Length (w=0.2)")
ax_sl.set_title("Survival vs Length Score")

ax_pie = fig.add_subplot(gs[1, 2])
totals = [df[c].sum() for c in win_cols]
wedge_colors = [ACCENT, GREEN, RED, PURPLE, GREY]
ax_pie.pie(totals, labels=labels_bar, colors=wedge_colors,
           autopct="%1.0f%%", startangle=140,
           textprops={"fontsize": 7.5}, pctdistance=0.75)
ax_pie.set_title("Overall Win Breakdown")

ax_txt = fig.add_subplot(gs[1, 3])
ax_txt.axis("off")
txt = (
    f"Best Trial: #{int(best['Trial_Number'])}\n"
    f"────────────────\n"
    f"Composite:  {best['Optuna_Composite_Score']:.3f}\n"
    f"Survival:   {best['Avg_Survival_Score']:.3f}\n"
    f"Length:     {best['Avg_Length_Score']:.3f}\n"
    f"────────────────\n"
    f"C_PARAM:    {best['C_PARAM']:.3f}\n"
    f"DEPTH:      {int(best['DEPTH_LIMIT'])}\n"
    f"PB_WEIGHT:  {best['PB_WEIGHT']:.3f}\n"
    f"TARGET_LEN: {int(best['TARGET_LENGTH'])}"
)
ax_txt.text(0.08, 0.95, txt, transform=ax_txt.transAxes,
            va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff",
                      edgecolor=ACCENT, linewidth=1.2))
ax_txt.set_title("Best Trial Summary", fontsize=10, fontweight="bold")

for i, param in enumerate(PARAMS):
    ax_p = fig.add_subplot(gs[2, i])
    ax_p.scatter(df[param], df["Optuna_Composite_Score"],
                 s=25, color=ACCENT, alpha=0.65, edgecolors="white", lw=0.3)
    z = np.polyfit(df[param], df["Optuna_Composite_Score"], 1)
    xs = np.linspace(df[param].min(), df[param].max(), 100)
    ax_p.plot(xs, np.poly1d(z)(xs), color=ORANGE, lw=1.4, linestyle="--")
    ax_p.set_title(param_labels[param], fontsize=9)
    ax_p.set_xlabel(param, fontsize=8)
    ax_p.set_ylabel("Score" if i == 0 else "", fontsize=8)

fig.suptitle("Optuna Results — Summary Dashboard", fontsize=14, fontweight="bold", y=1.01)
fig.savefig(os.path.join(OUTPUT_DIR, "fig10_dashboard.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig10_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 11 — fANOVA Hyperparameter Importance
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

# Sort importances from lowest to highest for the horizontal bar chart
sorted_params = sorted(importances.keys(), key=lambda k: importances[k])
sorted_vals = [importances[k] for k in sorted_params]
sorted_labels = [param_labels[k] for k in sorted_params]

bars = ax.barh(sorted_labels, sorted_vals, color=PURPLE, alpha=0.85, edgecolor="white")

# Add the exact values next to the bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}", ha="left", va="center", fontsize=9, fontweight="bold", color="#333333")

ax.set_xlabel("Importance Score (fANOVA)")
ax.set_title("Hyperparameter Importance")
# Extend x-axis limit slightly so our text annotations don't get cut off
ax.set_xlim(0, max(sorted_vals) * 1.15) 

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig11_fanova_importance.png"))
plt.close(fig)
print(f"✓ {OUTPUT_DIR}/fig11_fanova_importance.png")

print("\nAll 11 figures saved successfully!")