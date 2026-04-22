"""
analyze_deaths.py
=================
Parses the existing game_summaries.csv to infer how each agent died 
based on their final recorded health. No JSON files required.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Missing: pip install matplotlib")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUMMARIES_CSV = Path("logs-1/game_summaries.csv")
FIG_DIR       = Path("logs-1/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "Snake1":   "Heuristic",
    "Snake1_b": "Heuristic",
    "Snake2":   "MCTS",
    "Snake2_b": "MCTS",
    "Snake3":   "MCTSVar",
    "Snake3_b": "MCTSVar",
    "Snake4":   "VanillaMCTS",
    "Snake4_b": "VanillaMCTS",
}

AGENT_ORDER = ["Heuristic", "MCTS", "MCTSVar", "VanillaMCTS"]

CAUSE_COLORS = {
    "Survived":            "#4CAF50",  # Green
    "Collision":           "#F44336",  # Red
    "Starvation / Hazard": "#FFC107",  # Amber
}

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.35,
})


def infer_death_cause(row):
    # Check if they survived the time limit
    if str(row.get("alive_at_end", False)).lower() == "true" or row["turns_survived"] >= 300:
        return "Survived"
    
    # If health is 15 or less, it's starvation or late-game hazard damage
    if row["final_health"] <= 15:
        return "Starvation / Hazard"
    
    # High health but dead means they crashed
    return "Collision"


def process_data() -> pd.DataFrame:
    if not SUMMARIES_CSV.exists():
        sys.exit(f"Could not find {SUMMARIES_CSV}. Make sure you are in the right folder.")

    df = pd.read_csv(SUMMARIES_CSV)
    
    # Deduplicate in case an agent logged twice for the same game
    df = (
        df.sort_values("turns_survived", ascending=False)
          .drop_duplicates(subset=["game_id", "snake_id"], keep="first")
          .reset_index(drop=True)
    )

    # Apply readable names and infer death
    df["agent"] = df["snake_name"].map(NAME_MAP).fillna(df["snake_name"])
    df["cause"] = df.apply(infer_death_cause, axis=1)

    print(f"Loaded {len(df)} snake records from {SUMMARIES_CSV}")
    return df


def plot_death_causes(df: pd.DataFrame, output_path: Path, percentage: bool = False):
    """Generates a stacked bar chart of death causes."""
    summary = df.groupby(['agent', 'cause']).size().unstack(fill_value=0)
    
    causes_ordered = ["Survived", "Collision", "Starvation / Hazard"]
    for c in causes_ordered:
        if c not in summary.columns:
            summary[c] = 0
    summary = summary[causes_ordered]

    agents_present = [a for a in AGENT_ORDER if a in summary.index]
    summary = summary.reindex(agents_present)

    if percentage:
        summary = summary.div(summary.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    
    bottom = np.zeros(len(summary))
    for cause in causes_ordered:
        values = summary[cause].values
        color = CAUSE_COLORS.get(cause, "#9E9E9E")
        
        bars = ax.bar(summary.index, values, bottom=bottom, color=color, 
                      edgecolor="white", linewidth=0.5, label=cause)
        
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if (percentage and val > 6) or (not percentage and val > (summary.values.max() * 0.05)):
                label_text = f"{val:.0f}%" if percentage else f"{val:.0f}"
                ax.text(i, bot + val / 2, label_text, ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=9)
                
        bottom += values

    ax.set_ylabel("Percentage (%)" if percentage else "Total Occurrences")
    ax.set_title("Causes of Elimination by Agent (Percentage)" if percentage else "Causes of Elimination by Agent (Count)")
    
    if percentage:
        ax.set_ylim(0, 105)
        
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    print("=" * 60)
    print("  BattleSnakes — Elimination Analysis (Fast Mode)")
    print("=" * 60)

    df = process_data()
    
    plot_death_causes(df, FIG_DIR / "fig9_death_causes_pct.png", percentage=True)
    plot_death_causes(df, FIG_DIR / "fig10_death_causes_count.png", percentage=False)

    print("\nGraphs generated successfully! Check logs-1/figures/")

if __name__ == "__main__":
    main()