"""
run_1v1_tournament.py
=====================
Runs a 1v1 tournament between MCTS and Heuristic agents using joblib.
Generates performance and death-cause graphs based on the engine's JSON logs.
"""

import os
import sys
import json
import time
import random
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np

THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR.parent

try:
    from joblib import Parallel, delayed
except ImportError:
    sys.exit("Missing library. Please run: pip install joblib")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing library. Please run: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Missing library. Please run: pip install matplotlib")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAMES_TOTAL = 100
N_WORKERS   = 4
BOARD_W     = 11
BOARD_H     = 11

LOGS_DIR = Path("logs-2v2-1")
FIG_DIR  = LOGS_DIR / "figures"

MCTS_AGENT_PATH      = ROOT_DIR / "main/mcts_agent-final.py"
HEURISTIC_AGENT_PATH = ROOT_DIR / "main/heuristic_agent.py"

AGENT_1 = {"name": "MCTS",      "file": MCTS_AGENT_PATH,      "color": "#4CAF50"}
AGENT_2 = {"name": "Heuristic", "file": HEURISTIC_AGENT_PATH, "color": "#2196F3"}

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def start_agent(filepath: str, port: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PORT"] = str(port)
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(filepath)], env=env, stdout=lf, stderr=lf
        )
    return proc


def wait_for_port(port: int, timeout: float = 10.0):
    try:
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5)
                return
            except Exception:
                time.sleep(0.2)
    except ImportError:
        time.sleep(2.0)


def worker_task(worker_id: int, game_indices: list):
    worker_dir = LOGS_DIR / f"worker_{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    port_mcts = 8000 + worker_id * 10
    port_heur = 8000 + worker_id * 10 + 1

    proc_mcts = start_agent(AGENT_1["file"], port_mcts, worker_dir / "mcts.log")
    proc_heur = start_agent(AGENT_2["file"], port_heur, worker_dir / "heuristic.log")

    wait_for_port(port_mcts)
    wait_for_port(port_heur)

    try:
        for idx in game_indices:
            game_json = worker_dir / f"game_{idx}.json"
            if game_json.exists():
                game_json.unlink()

            seed = random.randint(1, 999_999_999)

            competitors = [
                ("--name", AGENT_1["name"], "--url", f"http://127.0.0.1:{port_mcts}"),
                ("--name", AGENT_2["name"], "--url", f"http://127.0.0.1:{port_heur}"),
            ]
            random.shuffle(competitors)

            cmd = [
                "battlesnake", "play",
                "-W", str(BOARD_W), "-H", str(BOARD_H),
                "-g", "standard", "-m", "hz_hazard_pits",
                competitors[0][0], competitors[0][1], competitors[0][2], competitors[0][3],
                competitors[1][0], competitors[1][1], competitors[1][2], competitors[1][3],
                "--output", str(game_json),
                "--seed", str(seed),
                "--timeout", "1000",
            ]

            print(f"[Worker {worker_id}] Starting Game {idx} (Seed: {seed})...")
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                print(f"[Worker {worker_id}] Game {idx} hit 5-min failsafe. Terminating.")

    finally:
        if proc_mcts.poll() is None:
            proc_mcts.kill()
        if proc_heur.poll() is None:
            proc_heur.kill()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def parse_game_jsons() -> pd.DataFrame:
    """
    Parse every game_*.json file and return one row per (game, snake).

    Fix for KeyError: status_map is now seeded from ALL states, not just
    states[0]. Snakes that appear mid-game (edge-case engine behaviour) or
    whose IDs turn up in a collision check but were never in states[0] will
    still be handled gracefully.
    """
    records = []

    for game_file in sorted(LOGS_DIR.rglob("game_*.json")):
        # ── Load all NDJSON lines ──────────────────────────────────────────
        states = []
        try:
            with open(game_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "board" in obj:
                            states.append(obj)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

        if not states:
            continue

        # ── Build status_map from ALL snakes seen in ANY state ────────────
        # This prevents KeyError when a snake ID appears in a collision
        # event but wasn't present in states[0] (e.g. engine quirks).
        status_map: dict[str, dict] = {}
        for state in states:
            for s in state.get("board", {}).get("snakes", []):
                sid = s["id"]
                if sid not in status_map:
                    status_map[sid] = {
                        "name":  s.get("name", sid),
                        "cause": "Survived",          # default; may be overwritten
                        "turns": int(state.get("turn", 0)),
                    }

        if not status_map:
            continue

        # ── Step through states to detect deaths ──────────────────────────
        for i in range(1, len(states)):
            curr_ids = {s["id"] for s in states[i].get("board", {}).get("snakes", [])}
            prev_snakes = {s["id"]: s for s in states[i - 1].get("board", {}).get("snakes", [])}

            for sid, prev_s in prev_snakes.items():
                if sid not in curr_ids:
                    # Snake disappeared between turn i-1 and turn i → it died
                    if sid not in status_map:
                        # Safety net: add it now (should not normally happen)
                        status_map[sid] = {
                            "name":  prev_s.get("name", sid),
                            "cause": "Survived",
                            "turns": i,
                        }
                    if status_map[sid]["cause"] == "Survived":
                        # Only record the first death cause
                        health = prev_s.get("health", 100)
                        status_map[sid]["cause"] = (
                            "Starvation / Hazard" if health <= 15 else "Collision"
                        )
                        status_map[sid]["turns"] = i

        # ── Determine winner ──────────────────────────────────────────────
        last_snakes = states[-1].get("board", {}).get("snakes", [])
        if len(last_snakes) == 1:
            winner_name = last_snakes[0]["name"]
        elif len(last_snakes) == 0:
            # Both died on the same turn — pick whoever survived longer
            survived = sorted(
                status_map.values(), key=lambda x: x["turns"], reverse=True
            )
            winner_name = survived[0]["name"] if survived else "Draw"
        else:
            # Hit turn limit with multiple survivors; longest snake wins
            longest = max(last_snakes, key=lambda s: s.get("length", 0))
            winner_name = longest["name"]

        # ── Emit records ──────────────────────────────────────────────────
        for info in status_map.values():
            records.append({
                "game_file":      game_file.name,
                "agent":          info["name"],
                "cause":          info["cause"],
                "turns_survived": info["turns"],
                "is_winner":      info["name"] == winner_name,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def style_plot():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    })


def plot_win_rate(df: pd.DataFrame):
    wins   = df[df["is_winner"]]["agent"].value_counts()
    n_games = df["game_file"].nunique()
    draws  = n_games - wins.sum()

    labels = list(wins.index) + (["Draw"] if draws > 0 else [])
    sizes  = list(wins.values) + ([draws] if draws > 0 else [])
    colors = []
    for label in labels:
        if label == AGENT_1["name"]:
            colors.append(AGENT_1["color"])
        elif label == AGENT_2["name"]:
            colors.append(AGENT_2["color"])
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, wedgeprops={"edgecolor": "white"},
    )
    ax.set_title(f"1v1 Win Rate  ({n_games} games)")
    fig.savefig(FIG_DIR / "fig1_win_rate.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_win_rate.png")


def plot_death_causes(df: pd.DataFrame):
    summary = df.groupby(["agent", "cause"]).size().unstack(fill_value=0)
    causes_ordered = ["Survived", "Collision", "Starvation / Hazard"]
    for c in causes_ordered:
        if c not in summary.columns:
            summary[c] = 0
    summary = summary[causes_ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom      = np.zeros(len(summary))
    cause_colors = {
        "Survived":            "#4CAF50",
        "Collision":           "#F44336",
        "Starvation / Hazard": "#FFC107",
    }

    for cause in causes_ordered:
        vals = summary[cause].values
        ax.bar(
            summary.index, vals, bottom=bottom,
            label=cause, color=cause_colors[cause], edgecolor="white",
        )
        max_val = summary.values.max() or 1
        for i, val in enumerate(vals):
            if val > max_val * 0.05:
                ax.text(
                    i, bottom[i] + val / 2, str(val),
                    ha="center", va="center", color="white", fontweight="bold",
                )
        bottom += vals

    ax.set_ylabel("Number of Games")
    ax.set_title("Causes of Elimination by Agent")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_death_causes.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_death_causes.png")


def plot_turn_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    agents  = [AGENT_1["name"], AGENT_2["name"]]
    data    = [df[df["agent"] == a]["turns_survived"].values for a in agents]
    colors  = [AGENT_1["color"], AGENT_2["color"]]

    bp = ax.boxplot(data, tick_labels=agents, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Mean markers
    for i, d in enumerate(data, start=1):
        if len(d):
            ax.scatter(i, np.mean(d), marker="D", color="black", s=45,
                       zorder=5, label="Mean" if i == 1 else "")
    ax.legend(fontsize=9)
    ax.set_ylabel("Turns Survived")
    ax.set_title("Game Length / Survivability Distribution")
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_turn_distribution.png")
    plt.close(fig)
    print("  Saved fig3_turn_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  1v1 Tournament: {AGENT_1['name']} vs {AGENT_2['name']}")
    print(f"  Total Games: {GAMES_TOTAL} | Workers: {N_WORKERS}")
    print("=" * 60)

    indices = list(range(1, GAMES_TOTAL + 1))
    chunks  = [indices[i::N_WORKERS] for i in range(N_WORKERS)]

    t0 = time.time()
    Parallel(n_jobs=N_WORKERS)(
        delayed(worker_task)(i, chunks[i]) for i in range(N_WORKERS)
    )
    print(f"\nTournament complete in {time.time() - t0:.1f} seconds. Parsing logs...")

    style_plot()
    df = parse_game_jsons()

    if df.empty:
        print("No valid JSON logs found. Check if the engine produced output.")
        return

    n_games = df["game_file"].nunique()
    print(f"\nParsed {n_games} games, {len(df)} total snake records.")

    plot_win_rate(df)
    plot_death_causes(df)
    plot_turn_distribution(df)

    # ── Quick console summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    wins = df[df["is_winner"]]["agent"].value_counts()
    for agent, w in wins.items():
        pct = 100 * w / n_games
        print(f"  {agent:<15}  {w:>3} wins  ({pct:.1f}%)")
    draws = n_games - wins.sum()
    if draws:
        print(f"  {'Draw':<15}  {draws:>3}       ({100*draws/n_games:.1f}%)")
    print("=" * 50)
    print("\nGraphs saved to logs-2v2/figures/")


if __name__ == "__main__":
    main()