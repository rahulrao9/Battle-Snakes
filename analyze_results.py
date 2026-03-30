"""
analyze_results.py
==================
Phase 4 statistical analysis pipeline.

Reads:
    logs/game_summaries.csv   — written by logger.py inside each agent
    logs/experiment_map.csv   — written by tournament_runner.py

Produces:
    1. TrueSkill ratings
    2. ELO ratings
    3. Win rates per experiment
    4. Four publication-quality figures in logs/figures/
    5. LaTeX table printed to terminal

AGENT NAME MAPPING:
    Snake1 → Heuristic
    Snake2 → MCTS
    Snake3 → MCTSVar
    Snake4 → VanillaMCTS
"""

import sys
from collections import defaultdict
from pathlib import Path

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

try:
    import trueskill
except ImportError:
    sys.exit("Missing: pip install trueskill")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUMMARIES_CSV = Path("logs/game_summaries.csv")
EXP_MAP_CSV   = Path("logs/experiment_map.csv")
FIG_DIR       = Path("logs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Agent name mapping  — Snake1/2/3/4 → readable names for plots
# ---------------------------------------------------------------------------

NAME_MAP = {
    "Snake1": "Heuristic",
    "Snake2": "MCTS",
    "Snake3": "MCTSVar",
    "Snake4": "VanillaMCTS",
}

AGENT_COLORS = {
    "Heuristic":   "#2196F3",   # blue
    "MCTS":        "#4CAF50",   # green
    "MCTSVar":     "#E91E63",   # pink
    "VanillaMCTS": "#FF9800",   # orange
}

def agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#607D8B")

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

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

# ===========================================================================
# 1. Load and clean data
# ===========================================================================

def load_data() -> pd.DataFrame:
    """
    Loads game_summaries.csv and experiment_map.csv.
    Deduplicates the 16-rows-per-game issue by keeping the row with
    the highest turns_survived for each (game_id, snake_id) pair.
    Merges experiment name from experiment_map.csv.
    Maps Snake1/2/3/4 to readable agent names.
    """

    # ── Check files exist ─────────────────────────────────────────────────
    if not SUMMARIES_CSV.exists():
        sys.exit(f"Not found: {SUMMARIES_CSV}\nMake sure agents have run at least one game.")
    if not EXP_MAP_CSV.exists():
        sys.exit(f"Not found: {EXP_MAP_CSV}\nMake sure tournament_runner.py has been run.")

    # ── Load raw summaries ────────────────────────────────────────────────
    df = pd.read_csv(SUMMARIES_CSV)
    print(f"Raw rows loaded       : {len(df)}")

    # ── Deduplicate: keep highest turns_survived per (game_id, snake_id) ─
    # This handles the 16-rows-per-game issue (4 agents each writing 4 rows)
    df = (
        df.sort_values("turns_survived", ascending=False)
          .drop_duplicates(subset=["game_id", "snake_id"], keep="first")
          .reset_index(drop=True)
    )
    print(f"After deduplication   : {len(df)} rows "
          f"({df['game_id'].nunique()} games, "
          f"{len(df)/df['game_id'].nunique():.1f} snakes/game avg)")

    # ── Load experiment map ───────────────────────────────────────────────
    exp_map = pd.read_csv(EXP_MAP_CSV)[["game_id", "experiment"]]
    df = df.merge(exp_map, on="game_id", how="left")
    df["experiment"] = df["experiment"].fillna("unknown")

    missing_exp = (df["experiment"] == "unknown").sum()
    if missing_exp > 0:
        print(f"WARNING: {missing_exp} rows have no experiment mapping.")

    # ── Map snake names ───────────────────────────────────────────────────
    df["agent"] = df["snake_name"].map(NAME_MAP).fillna(df["snake_name"])

    # ── Recompute placement cleanly after dedup ───────────────────────────
    # Sort within each game by the same logic as logger.py:
    # alive_at_end desc, turns_survived desc, max_length desc, final_health desc
    df = df.sort_values(
        ["game_id", "alive_at_end", "turns_survived", "max_length", "final_health"],
        ascending=[True, False, False, False, False],
    )
    df["placement"] = df.groupby("game_id").cumcount() + 1

    print(f"Experiments found     : {sorted(df['experiment'].unique())}")
    print(f"Agents found          : {sorted(df['agent'].unique())}")
    return df


# ===========================================================================
# 2. TrueSkill
# ===========================================================================

def compute_trueskill(df: pd.DataFrame) -> dict:
    env     = trueskill.TrueSkill(draw_probability=0.02)
    ratings = {}

    def get_rating(name):
        if name not in ratings:
            ratings[name] = env.create_rating()
        return ratings[name]

    games_processed = 0
    for game_id, game_df in df.groupby("game_id"):
        game_df = game_df.sort_values("placement")
        agents  = game_df["agent"].tolist()
        places  = game_df["placement"].tolist()

        if len(agents) < 2:
            continue

        rating_groups = [(get_rating(a),) for a in agents]
        rank_order    = [p - 1 for p in places]

        try:
            new_groups = env.rate(rating_groups, ranks=rank_order)
            for (new_r,), agent in zip(new_groups, agents):
                ratings[agent] = new_r
            games_processed += 1
        except Exception as e:
            pass

    print(f"TrueSkill processed   : {games_processed} games")
    return ratings


def conservative_score(r) -> float:
    return r.mu - 3 * r.sigma


# ===========================================================================
# 3. ELO
# ===========================================================================

def compute_elo(df: pd.DataFrame, k: int = 32, initial: int = 1000) -> dict:
    elo = defaultdict(lambda: initial)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for game_id, game_df in df.groupby("game_id"):
        placements = {
            row["agent"]: row["placement"]
            for _, row in game_df.iterrows()
        }
        agents = list(placements.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b   = agents[i], agents[j]
                pa, pb = placements[a], placements[b]

                if pa < pb:
                    sa, sb = 1.0, 0.0
                elif pa > pb:
                    sa, sb = 0.0, 1.0
                else:
                    sa = sb = 0.5

                ea     = expected(elo[a], elo[b])
                elo[a] += k * (sa - ea)
                elo[b] += k * (sb - (1.0 - ea))

    return dict(elo)


# ===========================================================================
# 4. Win rates per experiment
# ===========================================================================

def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winner per game = agent with placement 1.
    Computes win% per agent per experiment.
    """
    winners = df[df["placement"] == 1][["experiment", "game_id", "agent"]].copy()
    winners = winners.rename(columns={"agent": "winner"})

    total = (
        winners.groupby("experiment")["game_id"]
        .nunique()
        .rename("total_games")
        .reset_index()
    )
    wins = (
        winners.groupby(["experiment", "winner"])["game_id"]
        .count()
        .rename("wins")
        .reset_index()
    )
    result = wins.merge(total, on="experiment")
    result["win_pct"] = result["wins"] / result["total_games"] * 100
    return result


# ===========================================================================
# 5. Figures
# ===========================================================================

def fig_trueskill(ratings: dict, output: Path):
    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]), reverse=True)
    mus    = [ratings[a].mu    for a in agents]
    sigmas = [ratings[a].sigma for a in agents]
    errs   = [2 * s            for s in sigmas]
    colors = [agent_color(a)   for a in agents]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(
        agents, mus, xerr=errs, color=colors,
        capsize=5, edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.5},
    )
    for i, (mu, sigma, err) in enumerate(zip(mus, sigmas, errs)):
        ax.text(
            mu + err + 0.3, i,
            f"μ={mu:.1f}  σ={sigma:.1f}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("TrueSkill Rating  (μ ± 2σ)")
    ax.set_title("TrueSkill Ratings — All Agents")
    ax.axvline(25, color="grey", linestyle="--", linewidth=0.8, label="Prior μ=25")
    ax.legend(fontsize=9)
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


def fig_elo(elo: dict, output: Path):
    agents = sorted(elo, key=elo.get, reverse=True)
    scores = [elo[a] for a in agents]
    colors = [agent_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(agents, scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}", va="center", fontsize=9,
        )

    ax.set_xlabel("ELO Rating")
    ax.set_title("ELO Ladder — All Agents")
    ax.axvline(1000, color="grey", linestyle="--", linewidth=0.8, label="Starting ELO=1000")
    ax.legend(fontsize=9)
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


def fig_turns_survived(df: pd.DataFrame, output: Path):
    agents = sorted(df["agent"].unique())
    data   = [
        df.loc[df["agent"] == a, "turns_survived"].dropna().values
        for a in agents
    ]
    colors = [agent_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        data,
        tick_labels=agents,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Turns Survived")
    ax.set_title("Distribution of Turns Survived per Agent")
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


def fig_win_rates(win_df: pd.DataFrame, output: Path):
    experiments = sorted(win_df["experiment"].unique())
    n           = len(experiments)
    fig, axes   = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        sub    = win_df[win_df["experiment"] == exp].sort_values("win_pct", ascending=False)
        agents = sub["winner"].tolist()
        wpcts  = sub["win_pct"].tolist()
        colors = [agent_color(a) for a in agents]

        bars = ax.bar(agents, wpcts, color=colors, edgecolor="white", linewidth=0.5)
        for bar, wp in zip(bars, wpcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{wp:.1f}%", ha="center", fontsize=8,
            )

        ax.set_ylim(0, max(wpcts) * 1.25 + 5 if wpcts else 100)
        ax.set_ylabel("Win %")
        ax.set_title(exp.replace("_", " "), fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.grid(axis="y")

    plt.suptitle("Win Rates per Experiment", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


# ===========================================================================
# 6. Console tables
# ===========================================================================

def print_trueskill_table(ratings: dict):
    print("\n" + "=" * 65)
    print("  TRUESKILL RANKINGS")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Agent':<15} {'mu':>7} {'sigma':>7} {'mu-3sigma':>10}")
    print("  " + "-" * 48)
    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]), reverse=True)
    for rank, agent in enumerate(agents, 1):
        r  = ratings[agent]
        cs = conservative_score(r)
        print(f"  {rank:<5} {agent:<15} {r.mu:>7.2f} {r.sigma:>7.2f} {cs:>10.2f}")
    print("=" * 65)


def print_elo_table(elo: dict):
    print("\n" + "=" * 42)
    print("  ELO LADDER")
    print("=" * 42)
    print(f"  {'Rank':<5} {'Agent':<15} {'ELO':>8}")
    print("  " + "-" * 32)
    for rank, (agent, score) in enumerate(
        sorted(elo.items(), key=lambda x: x[1], reverse=True), 1
    ):
        print(f"  {rank:<5} {agent:<15} {score:>8.1f}")
    print("=" * 42)


def print_win_rate_table(win_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  WIN RATES PER EXPERIMENT")
    print("=" * 70)
    print(f"  {'Experiment':<35} {'Agent':<15} {'Wins':>5} {'Total':>6} {'Win%':>7}")
    print("  " + "-" * 60)
    for _, row in win_df.sort_values(
        ["experiment", "win_pct"], ascending=[True, False]
    ).iterrows():
        print(
            f"  {row['experiment']:<35} {row['winner']:<15} "
            f"{int(row['wins']):>5} {int(row['total_games']):>6} "
            f"{row['win_pct']:>6.1f}%"
        )
    print("=" * 70)


"""def print_latex_table(ratings: dict, elo: dict):
    print("\n% ── LaTeX table — paste into your report ───────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{clrrrr}")
    print(r"\toprule")
    print(r"Rank & Agent & $\mu$ & $\sigma$ & $\mu - 3\sigma$ & ELO \\")
    print(r"\midrule")
    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]), reverse=True)
    for rank, agent in enumerate(agents, 1):
        r  = ratings[agent]
        cs = conservative_score(r)
        e  = elo.get(agent, 1000)
        print(f"{rank} & {agent} & {r.mu:.2f} & {r.sigma:.2f} & {cs:.2f} & {e:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{TrueSkill and ELO rankings across all 7 experiments (350 games).}")
    print(r"\label{tab:rankings}")
    print(r"\end{table}")
    print("% ────────────────────────────────────────────────────────────────\n")"""


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 65)
    print("  BattleSnakes Phase 4 — Statistical Analysis")
    print("=" * 65)

    print("\n[1/5] Loading and cleaning data ...")
    df = load_data()

    print("\n[2/5] Computing TrueSkill ...")
    ratings = compute_trueskill(df)
    print_trueskill_table(ratings)

    print("\n[3/5] Computing ELO ...")
    elo = compute_elo(df)
    print_elo_table(elo)

    print("\n[4/5] Computing win rates ...")
    win_df = compute_win_rates(df)
    print_win_rate_table(win_df)

    print("\n[5/5] Saving figures ...")
    fig_trueskill(ratings,  FIG_DIR / "fig1_trueskill_ratings.png")
    fig_elo(elo,            FIG_DIR / "fig2_elo_ladder.png")
    fig_turns_survived(df,  FIG_DIR / "fig3_turns_survived.png")
    fig_win_rates(win_df,   FIG_DIR / "fig4_win_rates.png")

    #print_latex_table(ratings, elo)

    print(f"All figures saved to {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
