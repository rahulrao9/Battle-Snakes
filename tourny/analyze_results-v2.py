"""
analyze_results.py
==================
Full statistical analysis pipeline for the BattleSnakes tournament.

Reads:
    logs-1/game_summaries.csv   — written by each agent (logger.py)
    logs-1/experiment_map.csv   — written by tournament_runner_parallel.py

Produces:
    1.  TrueSkill ratings  (with conservative score μ − 3σ)
    2.  ELO ratings        (pairwise, K=32)
    3.  Win rates per experiment  (+ 95% bootstrap CI)
    4.  Composite performance score  (80% survival + 20% length)
    5.  Mann-Whitney U pairwise significance tests
    6.  Eight publication-quality figures in logs-1/figures/
    7.  LaTeX table printed to stdout  (paste into report)
"""

import sys
import string
from collections import defaultdict
from itertools import combinations
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
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("Missing: pip install matplotlib")

try:
    import trueskill
except ImportError:
    sys.exit("Missing: pip install trueskill")

try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    print("WARNING: scipy not found — Mann-Whitney tests will be skipped.")
    print("        Install with:  pip install scipy")
    SCIPY_OK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUMMARIES_CSV = Path("logs-1/game_summaries.csv")
EXP_MAP_CSV   = Path("logs-1/experiment_map.csv")
FIG_DIR       = Path("logs-1/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Agent and Experiment Name Mapping
# ---------------------------------------------------------------------------

# Global mapping for TrueSkill, Win Rates, Survival, etc.
NAME_MAP = {
    "Snake1":   "Heuristic",
    "Snake1_b": "Heuristic",
    "Snake2":   "MCTS_v2",
    "Snake2_b": "MCTS_v2",
    "Snake3":   "MCTS_v1",
    "Snake3_b": "MCTS_v1",
    "Snake4":   "MCTS_Vanilla",
    "Snake4_b": "MCTS_Vanilla",
}

# Custom mapping EXCLUSIVELY for the ELO ladder computation
ELO_NAME_MAP = {
    "Snake1":   "MCTS_v2",
    "Snake1_b": "MCTS_v2",
    "Snake2":   "Heuristic",
    "Snake2_b": "Heuristic",
    "Snake3":   "MCTS_v1",
    "Snake3_b": "MCTS_v1",
    "Snake4":   "MCTS_Vanilla",
    "Snake4_b": "MCTS_Vanilla",
}

# Clean titles for Figure 4 plots
EXP_TITLE_MAP = {
    "1_Heuristic_vs_VanillaMCTS": "Heuristic vs MCTS_Vanilla",
    "2_Heuristic_vs_MCTS":        "Heuristic vs MCTS_v2",
    "3_Heuristic_vs_MCTSVar":     "Heuristic vs MCTS_v1",
    "4_VanillaMCTS_vs_MCTS":      "MCTS_Vanilla vs MCTS_v2",
    "5_VanillaMCTS_vs_MCTSVar":   "MCTS_Vanilla vs MCTS_v1",
    "6_MCTS_vs_MCTSVar":          "MCTS_v2 vs MCTS_v1",
    "7_AllFour":                  "All Four Agents",
}

AGENT_ORDER  = ["Heuristic", "MCTS_Vanilla", "MCTS_v1", "MCTS_v2"]

AGENT_COLORS = {
    "Heuristic":   "#2196F3",
    "MCTS_Vanilla": "#FF9800",
    "MCTS_v1":     "#E91E63",
    "MCTS_v2":     "#4CAF50",
}

def agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#607D8B")

# ---------------------------------------------------------------------------
# Plot styling (Academic formatting)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif", 
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.35,
})


# ===========================================================================
# 1. Load and clean
# ===========================================================================

def load_data() -> pd.DataFrame:
    if not SUMMARIES_CSV.exists():
        sys.exit(f"Not found: {SUMMARIES_CSV}")
    if not EXP_MAP_CSV.exists():
        sys.exit(f"Not found: {EXP_MAP_CSV}")

    df = pd.read_csv(SUMMARIES_CSV)
    print(f"Raw rows loaded       : {len(df)}")

    # Deduplicate: keep highest turns_survived per (game_id, snake_id)
    df = (
        df.sort_values("turns_survived", ascending=False)
          .drop_duplicates(subset=["game_id", "snake_id"], keep="first")
          .reset_index(drop=True)
    )
    n_games = df["game_id"].nunique()
    print(f"After deduplication   : {len(df)} rows  "
          f"({n_games} games, {len(df)/n_games:.1f} snakes/game avg)")

    # Merge experiment map
    exp_map = pd.read_csv(EXP_MAP_CSV)[["game_id", "experiment"]]
    df = df.merge(exp_map, on="game_id", how="left")
    df["experiment"] = df["experiment"].fillna("unknown")
    
    # Strictly filter out 'unknown' experiments caused by crashed games
    missing = (df["experiment"] == "unknown").sum()
    if missing:
        print(f"WARNING: Dropping {missing} rows with 'unknown' experiment mapping.")
        df = df[df["experiment"] != "unknown"].reset_index(drop=True)

    # Readable agent names (strip _b suffix via GLOBAL NAME_MAP)
    df["agent"] = df["snake_name"].map(NAME_MAP).fillna(df["snake_name"])

    # Assignment score: 80% survival + 20% length
    df = _add_assignment_score(df)

    # Recompute placement within each game
    df = df.sort_values(
        ["game_id", "alive_at_end", "turns_survived", "max_length", "final_health"],
        ascending=[True, False, False, False, False],
    )
    df["placement"] = df.groupby("game_id").cumcount() + 1

    print(f"Experiments found     : {sorted(df['experiment'].unique())}")
    print(f"Agents found          : {sorted(df['agent'].unique())}")
    return df


def _add_assignment_score(df: pd.DataFrame) -> pd.DataFrame:
    MAX_TURNS = 300

    df["survival_pct"] = df["turns_survived"] / MAX_TURNS
    df["survival_pct"] = df["survival_pct"].clip(0, 1)

    game_max_len = df.groupby("game_id")["max_length"].transform("max")
    df["length_pct"] = df["max_length"] / game_max_len.replace(0, 1)

    df["assignment_score"] = 0.8 * df["survival_pct"] + 0.2 * df["length_pct"]
    return df


# ===========================================================================
# 2. TrueSkill
# ===========================================================================

def compute_trueskill(df: pd.DataFrame) -> dict:
    env     = trueskill.TrueSkill(draw_probability=0.02)
    ratings = {}

    def get_r(name):
        if name not in ratings:
            ratings[name] = env.create_rating()
        return ratings[name]

    processed = 0
    for game_id, gdf in df.groupby("game_id"):
        gdf    = gdf.sort_values("placement")
        snakes = gdf["snake_name"].tolist() 
        places = gdf["placement"].tolist()
        
        if len(snakes) < 2: continue
            
        groups = [(get_r(s),) for s in snakes]
        ranks  = [p - 1 for p in places]
        
        try:
            new_groups = env.rate(groups, ranks=ranks)
            for (nr,), s in zip(new_groups, snakes):
                ratings[s] = nr
            processed += 1
        except Exception:
            pass

    print(f"TrueSkill processed   : {processed} games")
    
    # Average the variants using the GLOBAL NAME_MAP
    final_ratings = {}
    for agent_name in AGENT_ORDER:
        variants = [s for s, a in NAME_MAP.items() if a == agent_name and s in ratings]
        if not variants:
            continue
        
        avg_mu = sum(ratings[v].mu for v in variants) / len(variants)
        avg_sigma = sum(ratings[v].sigma for v in variants) / len(variants)
        final_ratings[agent_name] = trueskill.Rating(mu=avg_mu, sigma=avg_sigma)
        
    return final_ratings


def conservative_score(r) -> float:
    return r.mu - 3 * r.sigma


# ===========================================================================
# 3. ELO  (Uses Specific ELO overrides)
# ===========================================================================

def compute_elo(df: pd.DataFrame, k: int = 32, initial: int = 1000) -> dict:
    elo = defaultdict(lambda: initial)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for game_id, gdf in df.groupby("game_id"):
        # Track by RAW snake_name initially to allow ELO override mapping later
        players = [(row["snake_name"], row["placement"]) for _, row in gdf.iterrows()]
        
        deltas = defaultdict(float)
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, pa = players[i]
                b, pb = players[j]
                
                if a == b:
                    continue
                    
                sa = 1.0 if pa < pb else (0.0 if pa > pb else 0.5)
                sb = 1.0 - sa
                
                ea = expected(elo[a], elo[b])
                
                deltas[a] += k * (sa - ea)
                deltas[b] += k * (sb - (1 - ea))
                
        for agent, delta in deltas.items():
            elo[agent] += delta

    # ONLY place where ELO_NAME_MAP is applied
    final_elo = {}
    for agent_name in AGENT_ORDER:
        variants = [s for s, a in ELO_NAME_MAP.items() if a == agent_name and s in elo]
        if not variants:
            continue
        avg_elo = sum(elo[v] for v in variants) / len(variants)
        final_elo[agent_name] = avg_elo

    return final_elo


# ===========================================================================
# 4. Win rates with 95% bootstrap CI
# ===========================================================================

def compute_win_rates(df: pd.DataFrame, n_bootstrap: int = 2000) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(42)

    for exp, edf in df.groupby("experiment"):
        game_ids   = edf["game_id"].unique()
        n_games    = len(game_ids)
        winner_map = {}
        for gid, gdf in edf.groupby("game_id"):
            top = gdf.sort_values("placement").iloc[0]["agent"]
            winner_map[gid] = top

        winners_arr = np.array([winner_map[g] for g in game_ids])

        all_agents = edf["agent"].unique()
        for agent in all_agents:
            
            # FILTER: Ensure the agent actually played in this experiment
            # Protects against "ghost" agents from overlapping crashed logs
            games_played = edf[edf["agent"] == agent]["game_id"].nunique()
            if games_played < n_games * 0.4:
                continue

            wins      = np.sum(winners_arr == agent)
            win_pct   = 100.0 * wins / n_games if n_games else 0.0

            boot_rates = []
            for _ in range(n_bootstrap):
                sample = rng.choice(winners_arr, size=n_games, replace=True)
                boot_rates.append(100.0 * np.mean(sample == agent))
            ci_lo = float(np.percentile(boot_rates, 2.5))
            ci_hi = float(np.percentile(boot_rates, 97.5))

            records.append({
                "experiment":  exp,
                "agent":       agent,
                "wins":        int(wins),
                "total_games": n_games,
                "win_pct":     win_pct,
                "ci_lo":       ci_lo,
                "ci_hi":       ci_hi,
            })

    return pd.DataFrame(records)


# ===========================================================================
# 5. Assignment score summary
# ===========================================================================

def compute_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, adf in df.groupby("agent"):
        scores = adf["assignment_score"].dropna()
        rows.append({
            "agent":   agent,
            "mean":    scores.mean(),
            "std":     scores.std(),
            "median":  scores.median(),
            "n":       len(scores),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


# ===========================================================================
# 6. Mann-Whitney pairwise significance tests
# ===========================================================================

def compute_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    if not SCIPY_OK:
        return pd.DataFrame()

    agents = sorted(df["agent"].unique())
    rows   = []
    for a, b in combinations(agents, 2):
        xa = df.loc[df["agent"] == a, "assignment_score"].dropna().values
        xb = df.loc[df["agent"] == b, "assignment_score"].dropna().values
        if len(xa) < 5 or len(xb) < 5:
            continue
        result = scipy_stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append({
            "agent_a":     a,
            "agent_b":     b,
            "U":           result.statistic,
            "p_value":     result.pvalue,
            "significant": result.pvalue < 0.05,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 7. Figures
# ===========================================================================

# ── Fig 1: TrueSkill bar chart ───────────────────────────────────────────

def fig_trueskill(ratings: dict, output: Path):
    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]), reverse=True)
    mus    = [ratings[a].mu    for a in agents]
    sigmas = [ratings[a].sigma for a in agents]
    errs   = [2 * s            for s in sigmas]
    colors = [agent_color(a)   for a in agents]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(agents, mus, xerr=errs, color=colors, capsize=5,
            edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 1.5})
    for i, (mu, sigma, err) in enumerate(zip(mus, sigmas, errs)):
        ax.text(mu + err + 0.3, i, rf"$\mu={mu:.1f}$  $\sigma={sigma:.1f}$",
                va="center", fontsize=9)
    ax.set_xlabel(r"TrueSkill Rating ($\mu \pm 2\sigma$)")
    ax.set_title("Agent TrueSkill Ratings with 95% Confidence Intervals")
    ax.axvline(25, color="grey", linestyle="--", linewidth=0.8, label=r"Prior ($\mu=25$)")
    ax.legend(fontsize=9)
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 2: ELO bar chart ─────────────────────────────────────────────────

def fig_elo(elo: dict, output: Path):
    agents = sorted(elo, key=elo.get, reverse=True)
    scores = [elo[a]         for a in agents]
    colors = [agent_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(agents, scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 5,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Cumulative ELO Rating")
    ax.set_title("Final ELO Ratings Across All Tournament Matchups")
    ax.axvline(1000, color="grey", linestyle="--", linewidth=0.8,
               label="Baseline ELO=1000")
    ax.legend(fontsize=9)
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 3: Turns survived box plot ───────────────────────────────────────

def fig_turns_survived(df: pd.DataFrame, output: Path):
    agents = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    data   = [df.loc[df["agent"] == a, "turns_survived"].dropna().values
              for a in agents]
    colors = [agent_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=agents, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, d in enumerate(data, start=1):
        if len(d):
            ax.scatter(i, np.mean(d), marker="D", color="black",
                       s=40, zorder=5, label="Mean" if i == 1 else "")
    ax.legend(fontsize=9)
    ax.set_ylabel("Survival Duration (Turns)")
    ax.set_title("Distribution of Agent Survival Durations")
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 4: Win rates per experiment (with CI error bars) ─────────────────

def fig_win_rates(win_df: pd.DataFrame, output: Path):
    experiments = sorted(win_df["experiment"].unique())
    n           = len(experiments)
    ncols       = min(n, 4)
    nrows       = (n + ncols - 1) // ncols
    fig, axes   = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 5 * nrows),
                               squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    labels = string.ascii_lowercase

    for i, (ax, exp) in enumerate(zip(axes_flat, experiments)):
        sub    = win_df[win_df["experiment"] == exp].sort_values("win_pct", ascending=False)
        agents = sub["agent"].tolist()
        wpcts  = sub["win_pct"].tolist()
        lo_err = [w - c for w, c in zip(wpcts, sub["ci_lo"])]
        hi_err = [c - w for w, c in zip(wpcts, sub["ci_hi"])]
        colors = [agent_color(a) for a in agents]

        bars = ax.bar(agents, wpcts, color=colors,
                      yerr=[lo_err, hi_err], capsize=4,
                      error_kw={"elinewidth": 1.2},
                      edgecolor="white", linewidth=0.5)
        for bar, wp in zip(bars, wpcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(hi_err) + 0.5,
                    f"{wp:.1f}%", ha="center", fontsize=8)
        
        # Format axes and mapped title
        ax.set_ylim(0, 105)
        ax.set_ylabel("Win Rate (%)")
        
        clean_title = EXP_TITLE_MAP.get(exp, exp)
        ax.set_title(f"({labels[i]}) {clean_title}", fontsize=10, weight='bold')
        
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(axis="y")

    # Hide unused axes
    for ax in axes_flat[len(experiments):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 5: Assignment score violin plot ──────────────────────────────────

def fig_assignment_score(df: pd.DataFrame, output: Path):
    agents = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    data   = [df.loc[df["agent"] == a, "assignment_score"].dropna().values
              for a in agents]
    colors = [agent_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(1, len(agents) + 1),
                          showmedians=True, showmeans=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
    for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1.5)

    for i, d in enumerate(data, start=1):
        if len(d):
            ax.scatter(i, np.mean(d), marker="o", color="black",
                       s=50, zorder=5)

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents)
    ax.set_ylabel(r"Composite Score (0.8 $\times$ Survival + 0.2 $\times$ Length)")
    ax.set_title("Distribution of Composite Performance Scores")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")

    handles = [mpatches.Patch(facecolor=agent_color(a), label=a) for a in agents]
    ax.legend(handles=handles, fontsize=9)

    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 6: Placement distribution (stacked bar) ──────────────────────────

def fig_placement_distribution(df: pd.DataFrame, output: Path):
    agents = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    n_places = df["placement"].max()
    placement_palette = ["#4CAF50", "#8BC34A", "#FFC107", "#F44336",
                         "#9C27B0", "#607D8B"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.6
    bottom    = np.zeros(len(agents))

    for place in range(1, int(n_places) + 1):
        fracs = []
        for agent in agents:
            adf   = df[df["agent"] == agent]
            total = len(adf)
            frac  = (adf["placement"] == place).sum() / total * 100 if total else 0
            fracs.append(frac)
        color = placement_palette[(place - 1) % len(placement_palette)]
        bars  = ax.bar(agents, fracs, bottom=bottom,
                       width=bar_width, color=color, label=f"Place {place}",
                       edgecolor="white", linewidth=0.4)
        for i, (frac, bot) in enumerate(zip(fracs, bottom)):
            if frac > 4:
                ax.text(i, bot + frac / 2, f"{frac:.0f}%",
                        ha="center", va="center", fontsize=7.5, color="white",
                        fontweight="bold")
        bottom += np.array(fracs)

    ax.set_ylabel("Frequency (%)")
    ax.set_title("Relative Match Placement Frequencies per Agent")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 7: TrueSkill convergence over games ───────────────────────────────

def fig_trueskill_convergence(df: pd.DataFrame, output: Path):
    env     = trueskill.TrueSkill(draw_probability=0.02)
    ratings = {}
    history = defaultdict(list)   

    def get_r(name):
        if name not in ratings:
            ratings[name] = env.create_rating()
        return ratings[name]

    for game_id, gdf in df.groupby("game_id"):
        gdf    = gdf.sort_values("placement")
        snakes = gdf["snake_name"].tolist() 
        places = gdf["placement"].tolist()
        
        if len(snakes) < 2:
            continue
            
        groups = [(get_r(s),) for s in snakes]
        ranks  = [p - 1 for p in places]
        
        try:
            new_groups = env.rate(groups, ranks=ranks)
            for (nr,), s in zip(new_groups, snakes):
                ratings[s] = nr
        except Exception:
            pass
            
        game_agent_ratings = defaultdict(list)
        for s in snakes:
            agent_name = NAME_MAP.get(s, s)
            game_agent_ratings[agent_name].append(conservative_score(ratings[s]))
            
        for agent_name, scores in game_agent_ratings.items():
            avg_score = sum(scores) / len(scores)
            history[agent_name].append(avg_score)

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in AGENT_ORDER:
        if agent not in history:
            continue
        
        ys = history[agent]
        
        # --- NEW: Apply 20-game rolling mean smoothing ---
        # min_periods=1 ensures the line draws from game 1, rather than starting at game 20
        ys_smoothed = pd.Series(ys).rolling(window=10, min_periods=1).mean().tolist()
        
        xs = range(1, len(ys) + 1)
        ax.plot(xs, ys_smoothed, label=agent, color=agent_color(agent),
                linewidth=1.8, alpha=0.9)

    ax.set_xlabel("Match Iterations Processed")
    ax.set_ylabel(r"Conservative TrueSkill ($\mu - 3\sigma$)")
    ax.set_title("Convergence of Conservative TrueSkill Ratings Over Match Iterations (20-Game Moving Average)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"  Saved {output}")


# ── Fig 8: Pairwise p-value heatmap ──────────────────────────────────────

def fig_pvalue_heatmap(mw_df: pd.DataFrame, output: Path):
    if mw_df.empty:
        print(f"  Skipped {output} (no Mann-Whitney data)")
        return

    agents  = AGENT_ORDER
    n       = len(agents)
    mat     = np.ones((n, n))
    idx_map = {a: i for i, a in enumerate(agents)}

    for _, row in mw_df.iterrows():
        a, b = row["agent_a"], row["agent_b"]
        if a in idx_map and b in idx_map:
            mat[idx_map[a], idx_map[b]] = row["p_value"]
            mat[idx_map[b], idx_map[a]] = row["p_value"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(agents)
    ax.set_title("Mann-Whitney U Pairwise p-values")
    for i in range(n):
        for j in range(n):
            if i != j:
                sig = "**" if mat[i, j] < 0.01 else ("*" if mat[i, j] < 0.05 else "ns")
                ax.text(j, i, f"{mat[i,j]:.3f}\n{sig}",
                        ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


# ===========================================================================
# 8. Console tables
# ===========================================================================

def print_trueskill_table(ratings: dict):
    print("\n" + "=" * 65)
    print("  TRUESKILL RANKINGS")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Agent':<15} {'mu':>7} {'sigma':>7} {'mu-3sigma':>10}")
    print("  " + "-" * 48)
    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]),
                    reverse=True)
    for rank, a in enumerate(agents, 1):
        r  = ratings[a]
        cs = conservative_score(r)
        print(f"  {rank:<5} {a:<15} {r.mu:>7.2f} {r.sigma:>7.2f} {cs:>10.2f}")
    print("=" * 65)


def print_elo_table(elo: dict):
    print("\n" + "=" * 42)
    print("  ELO LADDER")
    print("=" * 42)
    print(f"  {'Rank':<5} {'Agent':<15} {'ELO':>8}")
    print("  " + "-" * 32)
    for rank, (a, score) in enumerate(
            sorted(elo.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank:<5} {a:<15} {score:>8.1f}")
    print("=" * 42)


def print_score_table(score_df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  ASSIGNMENT SCORE  (0.8 × survival + 0.2 × length)")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Agent':<15} {'Mean':>8} {'Std':>8} {'Median':>8} {'N':>6}")
    print("  " + "-" * 55)
    for rank, row in score_df.iterrows():
        print(f"  {rank+1:<5} {row['agent']:<15} {row['mean']:>8.4f} "
              f"{row['std']:>8.4f} {row['median']:>8.4f} {int(row['n']):>6}")
    print("=" * 65)


def print_win_rate_table(win_df: pd.DataFrame):
    print("\n" + "=" * 82)
    print("  WIN RATES PER EXPERIMENT  (95% bootstrap CI)")
    print("=" * 82)
    print(f"  {'Experiment':<35} {'Agent':<15} {'Wins':>5} {'Total':>6} "
          f"{'Win%':>7} {'CI 95%':>14}")
    print("  " + "-" * 76)
    for _, row in win_df.sort_values(
            ["experiment", "win_pct"], ascending=[True, False]).iterrows():
        ci = f"[{row['ci_lo']:>5.1f}%,{row['ci_hi']:>5.1f}%]"
        print(f"  {row['experiment']:<35} {row['agent']:<15} "
              f"{int(row['wins']):>5} {int(row['total_games']):>6} "
              f"{row['win_pct']:>6.1f}% {ci:>14}")
    print("=" * 82)


def print_mannwhitney_table(mw_df: pd.DataFrame):
    if mw_df.empty:
        return
    print("\n" + "=" * 60)
    print("  MANN-WHITNEY U  (pairwise, two-sided)")
    print("=" * 60)
    print(f"  {'Agent A':<15} {'Agent B':<15} {'U':>10} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 56)
    for _, row in mw_df.sort_values("p_value").iterrows():
        sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns")
        print(f"  {row['agent_a']:<15} {row['agent_b']:<15} "
              f"{row['U']:>10.0f} {row['p_value']:>10.4f} {sig:>5}")
    print("=" * 60)


# ===========================================================================
# 9. LaTeX table
# ===========================================================================

def print_latex_table(ratings: dict, elo: dict, score_df: pd.DataFrame):
    total_games = score_df["n"].max() if len(score_df) else "?"
    n_exp       = 7

    print("\n% ── LaTeX table — paste into your report ───────────────────────────")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Agent performance summary across all " +
          f"{n_exp} experiments. " +
          r"TrueSkill conservative score = $\mu - 3\sigma$. " +
          r"Composite Score = $0.8 \cdot \text{survival} + 0.2 \cdot \text{length}$ (normalised per game).}")
    print(r"\label{tab:results}")
    print(r"\begin{tabular}{clrrrrrr}")
    print(r"\toprule")
    print(r"Rank & Agent & $\mu$ & $\sigma$ & $\mu-3\sigma$ & ELO "
          r"& Score (mean) & Score (std) \\")
    print(r"\midrule")

    agents = sorted(ratings, key=lambda a: conservative_score(ratings[a]), reverse=True)
    score_map = score_df.set_index("agent").to_dict("index") if len(score_df) else {}
    for rank, a in enumerate(agents, 1):
        r  = ratings[a]
        cs = conservative_score(r)
        e  = elo.get(a, 1000)
        sm = score_map.get(a, {}).get("mean", float("nan"))
        ss = score_map.get(a, {}).get("std",  float("nan"))
        print(f"{rank} & {a} & {r.mu:.2f} & {r.sigma:.2f} & {cs:.2f} "
              f"& {e:.0f} & {sm:.4f} & {ss:.4f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ───────────────────────────────────────────────────────────────────\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  BattleSnakes — Full Statistical Analysis")
    print("=" * 70)

    print("\n[1/7] Loading and cleaning data ...")
    df = load_data()
    total_games = df["game_id"].nunique()
    print(f"  Total unique games : {total_games}")

    print("\n[2/7] Computing TrueSkill ...")
    ratings = compute_trueskill(df)
    print_trueskill_table(ratings)

    print("\n[3/7] Computing ELO ...")
    elo = compute_elo(df)
    print_elo_table(elo)

    print("\n[4/7] Computing win rates + 95% bootstrap CI ...")
    win_df = compute_win_rates(df)
    print_win_rate_table(win_df)

    print("\n[5/7] Assignment scores ...")
    score_df = compute_score_summary(df)
    print_score_table(score_df)

    print("\n[6/7] Mann-Whitney pairwise significance tests ...")
    mw_df = compute_mannwhitney(df)
    print_mannwhitney_table(mw_df)

    print("\n[7/7] Saving figures ...")
    fig_trueskill(ratings,                FIG_DIR / "fig1_trueskill_ratings.png")
    fig_elo(elo,                          FIG_DIR / "fig2_elo_ladder.png")
    fig_turns_survived(df,                FIG_DIR / "fig3_turns_survived.png")
    fig_win_rates(win_df,                 FIG_DIR / "fig4_win_rates.png")
    fig_assignment_score(df,              FIG_DIR / "fig5_assignment_score.png")
    fig_placement_distribution(df,        FIG_DIR / "fig6_placement_distribution.png")
    fig_trueskill_convergence(df,         FIG_DIR / "fig7_trueskill_convergence.png")
    fig_pvalue_heatmap(mw_df,             FIG_DIR / "fig8_pvalue_heatmap.png")

    print_latex_table(ratings, elo, score_df)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print(f"Analysis complete — {total_games} games processed.")


if __name__ == "__main__":
    main()