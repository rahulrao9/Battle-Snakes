"""
tune_rahul_optuna.py
====================
Parallelized Optuna tuner for rahul_mcts_tunable.py.

Opponent pool (anti-overfit)
----------------------------
Each game draws its opponent at random from three agents with very different
play styles:
  • heuristic_agent.py  — fast greedy heuristic, hugs food, no tree search
  • mcts_agent3.py      — vanilla MCTS, moderate strength
  • vanilla_agent.py    — simple rule-based, mostly moves away from walls

Tuning against a single opponent (previously only mcts_agent3) caused Optuna
to over-specialise weights toward that agent's quirks.  Using a diverse pool
ensures the discovered hyper-parameters generalise to the real arena.

Each trial still plays GAMES_PER_TRIAL games total; the opponent for each
individual game is drawn i.i.d. from the pool.  The per-game results are
pooled into a single scalar score so Optuna sees one number per trial.
"""

# ── std-lib ───────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import time
import random
import subprocess
import logging
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

from joblib import Parallel, delayed

# =============================================================================
# CONFIGURATION
# =============================================================================

GAMES_PER_TRIAL    = 20
BOARD_SIZE         = 11
N_TRIALS           = 60
N_WORKERS          = 8
# Worker k occupies a port block of 10:
#   BASE_PORT + k*10 + 0  → opponent
#   BASE_PORT + k*10 + 1  → RahulMCTS
BASE_PORT          = 8000

RAHUL_TIME_MS      = 800
TIMEOUT_BUFFER_MS  = 100
CLI_TIMEOUT_MS     = RAHUL_TIME_MS + TIMEOUT_BUFFER_MS

BATTLESNAKE_BIN    = "./battlesnake"

TARGET_SCRIPT = "rahul_mcts_tunable.py"
TARGET_NAME   = "RahulMCTS"

# ---------------------------------------------------------------------------
# Opponent pool — add or remove entries freely.
# Each entry is (display_name, script_filename).
# Games rotate through this list randomly so Optuna can't overfit to one style.
# ---------------------------------------------------------------------------
OPPONENT_POOL = [
    ("Heuristic",  "heuristic_agent.py"),
    ("MCTS3",      "mcts_agent3.py"),
    ("Vanilla",    "vanilla_mcts.py"),
]

RESULTS_DIR   = Path("rahul_results-smarter-tuning")
MATCH_LOG_DIR = RESULTS_DIR / "match_logs"
PLOT_DIR      = RESULTS_DIR / "plots"
CSV_PATH      = RESULTS_DIR / "rahul_results.csv"
DB_PATH       = RESULTS_DIR / "rahul_optuna.db"
STORAGE_URL   = f"sqlite:///{DB_PATH}"
STUDY_NAME    = "rahul_mcts"


# =============================================================================
# SETUP
# =============================================================================

def setup():
    RESULTS_DIR.mkdir(exist_ok=True)
    MATCH_LOG_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    print("── Pre-flight ────────────────────────────────────────────────────")
    ok = True

    for display_name, fname in OPPONENT_POOL:
        sym = "✓" if Path(fname).exists() else "✗"
        if sym == "✗":
            ok = False
        print(f"  {sym} {fname}  [{display_name}]")

    sym = "✓" if Path(TARGET_SCRIPT).exists() else "✗"
    if sym == "✗":
        ok = False
    print(f"  {sym} {TARGET_SCRIPT}  [target]")

    sym = "✓" if Path(BATTLESNAKE_BIN).exists() else "✗"
    if sym == "✗":
        ok = False
    print(f"  {sym} {BATTLESNAKE_BIN}  [engine]")

    opp_names = ", ".join(n for n, _ in OPPONENT_POOL)
    print(f"\n  CLI timeout : {CLI_TIMEOUT_MS} ms  (rahul budget: {RAHUL_TIME_MS} ms)")
    print(f"  Workers     : {N_WORKERS}")
    print(f"  Trials      : {N_TRIALS}  ×  {GAMES_PER_TRIAL} games each")
    print(f"  Opponent pool ({len(OPPONENT_POOL)}): {opp_names}")
    print(f"  Opponent per game: drawn i.i.d. from pool (anti-overfit)")
    print("─────────────────────────────────────────────────────────────────\n")

    if not ok:
        print("ERROR: missing files — fix paths and retry.")
        sys.exit(1)

    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.writer(f).writerow([
                "trial", "worker", "game", "opponent",
                "C_PARAM", "DEPTH_LIMIT", "PB_WEIGHT", "RAVE_K",
                "PHASE_LATE_TURN", "PHASE_LATE_LENGTH",
                "FOOD_WEIGHT_EARLY", "SPACE_WEIGHT_LATE",
                "CENTER_WEIGHT_LATE", "CORNER_WEIGHT_LATE", "KILL_WEIGHT_LATE",
                "result", "survival", "final_length", "optuna_score",
            ])


# =============================================================================
# MATCH LOG PARSER
# =============================================================================

def parse_match(log_path: Path, target_name: str):
    """
    Returns (survival_score, winner_name, final_length).
    survival_score ∈ [0, 1].
    winner_name is the display name of the winner, or "Draw".
    """
    if not log_path.exists():
        return 0.0, "Draw", 3

    max_turn    = 0
    last_alive  = {}
    max_lengths = {}

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    state = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn     = state.get("turn", 0)
                max_turn = max(max_turn, turn)
                for s in state.get("board", {}).get("snakes", []):
                    n = s["name"]
                    last_alive[n]  = turn
                    max_lengths[n] = max(max_lengths.get(n, 3), s.get("length", 3))
    except OSError:
        return 0.0, "Draw", 3

    if max_turn <= 1 or not last_alive:
        return 0.0, "Draw", 3

    highest = max(last_alive.values())
    winners = [n for n, t in last_alive.items() if t == highest]
    if len(winners) > 1:
        winners.sort(key=lambda w: max_lengths.get(w, 0), reverse=True)
        winner = (
            winners[0]
            if max_lengths.get(winners[0], 0) > max_lengths.get(winners[1], 0)
            else "Draw"
        )
    else:
        winner = winners[0]

    survival = last_alive.get(target_name, 0) / 300.0
    length   = max_lengths.get(target_name, 3)
    return survival, winner, length


# =============================================================================
# SINGLE-GAME RUNNER
# =============================================================================

def run_one_game(
    worker_id:    int,
    game_index:   int,
    trial_number: int,
    opp_name:     str,
    opp_script:   str,
    rahul_env:    dict,
    opp_port:     int,
    rahul_port:   int,
) -> dict:
    """
    Starts the opponent process, starts RahulMCTS, runs one game, parses result.
    Returns a dict with result metadata.
    """
    def _env(port: int, extra: dict = None) -> dict:
        e = os.environ.copy()
        e["PORT"] = str(port)
        if extra:
            e.update(extra)
        return e

    log_path = (
        MATCH_LOG_DIR
        / f"w{worker_id}_t{trial_number}_g{game_index}_{opp_name}.json"
    )
    log_path.unlink(missing_ok=True)

    opp_proc   = None
    rahul_proc = None

    try:
        opp_proc = subprocess.Popen(
            ["python", opp_script],
            env=_env(opp_port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rahul_proc = subprocess.Popen(
            ["python", TARGET_SCRIPT],
            env=rahul_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)   # let both servers bind

        cli = [
            BATTLESNAKE_BIN, "play",
            "-W", str(BOARD_SIZE), "-H", str(BOARD_SIZE),
            "-g", "standard",
            "-m", "hz_hazard_pits",
            "--timeout", str(CLI_TIMEOUT_MS),
            "--name", opp_name,     "--url", f"http://127.0.0.1:{opp_port}",
            "--name", TARGET_NAME,  "--url", f"http://127.0.0.1:{rahul_port}",
            "--output", str(log_path),
        ]
        try:
            subprocess.run(cli, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            pass

        survival, winner, final_len = parse_match(log_path, TARGET_NAME)

        result = "win" if winner == TARGET_NAME else (
            "draw" if winner == "Draw" else "loss"
        )

        return {
            "opponent":   opp_name,
            "result":     result,
            "survival":   survival,
            "length":     final_len,
        }

    finally:
        for p in (opp_proc, rahul_proc):
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    pass


# =============================================================================
# TRIAL RUNNER  (plays GAMES_PER_TRIAL games, rotating opponents)
# =============================================================================

def run_trial(worker_id: int, params: dict, trial_number: int) -> dict:
    """
    Plays GAMES_PER_TRIAL games against randomly selected opponents.
    Each game gets a fresh opponent process so port conflicts are avoided.

    Port layout for this worker:
      BASE_PORT + worker_id*10 + 0  → opponent
      BASE_PORT + worker_id*10 + 1  → RahulMCTS
    """
    pb         = BASE_PORT + worker_id * 10
    opp_port   = pb + 0
    rahul_port = pb + 1

    rahul_env = os.environ.copy()
    rahul_env.update({
        "PORT":                        str(rahul_port),
        "RAHUL_TIME_LIMIT_MS":         str(RAHUL_TIME_MS),
        "RAHUL_C_PARAM":               str(params["C_PARAM"]),
        "RAHUL_DEPTH_LIMIT":           str(params["DEPTH_LIMIT"]),
        "RAHUL_PB_WEIGHT":             str(params["PB_WEIGHT"]),
        "RAHUL_RAVE_K":                str(params["RAVE_K"]),
        "RAHUL_PHASE_LATE_TURN":       str(params["PHASE_LATE_TURN"]),
        "RAHUL_PHASE_LATE_LENGTH":     str(params["PHASE_LATE_LENGTH"]),
        "RAHUL_FOOD_WEIGHT_EARLY":     str(params["FOOD_WEIGHT_EARLY"]),
        "RAHUL_SPACE_WEIGHT_LATE":     str(params["SPACE_WEIGHT_LATE"]),
        "RAHUL_CENTER_WEIGHT_LATE":    str(params["CENTER_WEIGHT_LATE"]),
        "RAHUL_CORNER_WEIGHT_LATE":    str(params["CORNER_WEIGHT_LATE"]),
        "RAHUL_KILL_WEIGHT_LATE":      str(params["KILL_WEIGHT_LATE"]),
    })

    # Metrics accumulators — tracked globally and per-opponent
    wins = draws = losses = 0
    total_surv   = 0.0
    total_length = 0.0
    games_played = 0

    per_opp_wins: dict = {name: 0 for name, _ in OPPONENT_POOL}
    per_opp_games: dict = {name: 0 for name, _ in OPPONENT_POOL}

    game_records = []   # for CSV logging

    # Draw a fixed opponent sequence for this trial (reproducible given seed)
    rng      = random.Random(trial_number * 1000 + worker_id)
    opp_seq  = [rng.choice(OPPONENT_POOL) for _ in range(GAMES_PER_TRIAL)]

    for g_idx, (opp_name, opp_script) in enumerate(opp_seq):
        result_dict = run_one_game(
            worker_id    = worker_id,
            game_index   = g_idx,
            trial_number = trial_number,
            opp_name     = opp_name,
            opp_script   = opp_script,
            rahul_env    = rahul_env,
            opp_port     = opp_port,
            rahul_port   = rahul_port,
        )

        result   = result_dict["result"]
        survival = result_dict["survival"]
        length   = result_dict["length"]

        total_surv   += survival
        total_length += min(1.0, max(0.0, (length - 3) / 12.0))
        games_played += 1

        if result == "win":
            wins += 1
            per_opp_wins[opp_name]  += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1

        per_opp_games[opp_name] = per_opp_games.get(opp_name, 0) + 1

        game_records.append({
            "game":     g_idx,
            "opponent": opp_name,
            "result":   result,
            "survival": survival,
            "length":   length,
        })

        # Early stopping: clearly losing after 10 games (< 25% win rate)
        if g_idx == 10:
            if wins / games_played < 0.25:
                break

    n          = games_played
    avg_surv   = total_surv   / n
    avg_length = total_length / n
    win_rate   = wins         / n

    # Score = weighted combination of win rate, survival, and length
    # Win rate dominates (0.70) as it's the cleanest signal.
    score = avg_surv * 0.15 + avg_length * 0.15 + win_rate * 0.70

    return {
        "score":        score,
        "avg_surv":     avg_surv,
        "avg_length":   avg_length,
        "win_rate":     win_rate,
        "wins":         wins,
        "draws":        draws,
        "losses":       losses,
        "games":        n,
        "per_opp_wins": per_opp_wins,
        "per_opp_games":per_opp_games,
        "game_records": game_records,
    }


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(trial: optuna.Trial, worker_id: int) -> float:
    params = {
        "C_PARAM":            trial.suggest_float("C_PARAM",            0.5,    3.0),
        "DEPTH_LIMIT":        trial.suggest_int(  "DEPTH_LIMIT",        8,      24),
        "PB_WEIGHT":          trial.suggest_float("PB_WEIGHT",          2.0,    22.0),
        "RAVE_K":             trial.suggest_float("RAVE_K",             50.0,   1200.0),
        "PHASE_LATE_TURN":    trial.suggest_int(  "PHASE_LATE_TURN",    25,     80),
        "PHASE_LATE_LENGTH":  trial.suggest_int(  "PHASE_LATE_LENGTH",  5,      16),
        "FOOD_WEIGHT_EARLY":  trial.suggest_float("FOOD_WEIGHT_EARLY",  1.0,    7.0),
        "SPACE_WEIGHT_LATE":  trial.suggest_float("SPACE_WEIGHT_LATE",  0.5,    5.0),
        "CENTER_WEIGHT_LATE": trial.suggest_float("CENTER_WEIGHT_LATE", 0.0,    4.0),
        "CORNER_WEIGHT_LATE": trial.suggest_float("CORNER_WEIGHT_LATE", 0.0,    6.0),
        "KILL_WEIGHT_LATE":   trial.suggest_float("KILL_WEIGHT_LATE",   0.0,    7.0),
    }

    print(
        f"  [W{worker_id}|T{trial.number}] "
        f"C={params['C_PARAM']:.2f} D={params['DEPTH_LIMIT']} "
        f"PB={params['PB_WEIGHT']:.1f} RAVE={params['RAVE_K']:.0f} | "
        f"LT={params['PHASE_LATE_TURN']} LL={params['PHASE_LATE_LENGTH']} | "
        f"FWE={params['FOOD_WEIGHT_EARLY']:.1f} SWL={params['SPACE_WEIGHT_LATE']:.1f} "
        f"CWL={params['CENTER_WEIGHT_LATE']:.1f} "
        f"CorWL={params['CORNER_WEIGHT_LATE']:.1f} KWL={params['KILL_WEIGHT_LATE']:.1f}",
        flush=True,
    )

    metrics = run_trial(worker_id, params, trial.number)
    score   = metrics["score"]

    # ── per-opponent win rate breakdown ──────────────────────────────────────
    opp_summary = "  ".join(
        f"{n}={metrics['per_opp_wins'].get(n,0)}/{metrics['per_opp_games'].get(n,0)}"
        for n, _ in OPPONENT_POOL
    )
    print(
        f"  [W{worker_id}|T{trial.number}] "
        f"score={score:.4f} wr={metrics['win_rate']:.2f} "
        f"surv={metrics['avg_surv']:.3f} games={metrics['games']} | "
        f"W={metrics['wins']} D={metrics['draws']} L={metrics['losses']} | "
        f"by-opp: {opp_summary}",
        flush=True,
    )

    # ── CSV logging — one row per game ───────────────────────────────────────
    for _ in range(10):
        try:
            with CSV_PATH.open("a", newline="") as f:
                w = csv.writer(f)
                for rec in metrics["game_records"]:
                    row_score = (
                        rec["survival"] * 0.15
                        + min(1.0, max(0.0, (rec["length"] - 3) / 12.0)) * 0.15
                        + (1.0 if rec["result"] == "win" else 0.0) * 0.70
                    )
                    w.writerow([
                        trial.number,
                        worker_id,
                        rec["game"],
                        rec["opponent"],
                        round(params["C_PARAM"],            4),
                        params["DEPTH_LIMIT"],
                        round(params["PB_WEIGHT"],          4),
                        round(params["RAVE_K"],              1),
                        params["PHASE_LATE_TURN"],
                        params["PHASE_LATE_LENGTH"],
                        round(params["FOOD_WEIGHT_EARLY"],  4),
                        round(params["SPACE_WEIGHT_LATE"],  4),
                        round(params["CENTER_WEIGHT_LATE"], 4),
                        round(params["CORNER_WEIGHT_LATE"], 4),
                        round(params["KILL_WEIGHT_LATE"],   4),
                        rec["result"],
                        round(rec["survival"],              4),
                        rec["length"],
                        round(row_score,                    4),
                    ])
            break
        except OSError:
            time.sleep(0.1)

    return score


# =============================================================================
# JOBLIB WORKER
# =============================================================================

def _worker(worker_id: int, n_trials: int):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    study.optimize(
        lambda trial: objective(trial, worker_id),
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )


# =============================================================================
# ANALYTICS & PLOTS
# =============================================================================

def generate_plots(study: optuna.Study):
    print("\n── Generating analytics & plots ─────────────────────────────────")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("  No completed trials — skipping plots.")
        return

    scores     = [t.value for t in complete]
    trial_nums = [t.number for t in complete]
    params_all = {k: [t.params[k] for t in complete] for k in complete[0].params}

    # ── 1. Optimization history ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trial_nums, scores, "o-", alpha=0.55, markersize=4, label="Trial score")
    best_so_far = [max(scores[:i+1]) for i in range(len(scores))]
    ax.plot(trial_nums, best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Optuna score")
    ax.set_title("Optimization History — RahulMCTS vs Mixed Opponent Pool")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "optimization_history.png", dpi=150)
    plt.close(fig)
    print("  ✓ optimization_history.png")

    # ── 2. Score distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=min(20, len(scores)), color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(max(scores),            color="red",    linestyle="--",
               label=f"Best  {max(scores):.4f}")
    ax.axvline(float(np.mean(scores)), color="orange", linestyle="--",
               label=f"Mean  {np.mean(scores):.4f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ score_distribution.png")

    # ── 3. Per-opponent win rates from CSV ────────────────────────────────────
    if CSV_PATH.exists():
        try:
            rows = []
            with CSV_PATH.open() as f:
                for row in csv.DictReader(f):
                    rows.append(row)

            if rows:
                # Win rate over time (all opponents combined)
                trial_wr: dict = {}
                for row in rows:
                    t = int(row["trial"])
                    trial_wr.setdefault(t, {"wins": 0, "games": 0})
                    trial_wr[t]["games"] += 1
                    if row["result"] == "win":
                        trial_wr[t]["wins"] += 1
                sorted_t = sorted(trial_wr)
                wr_curve = [trial_wr[t]["wins"] / trial_wr[t]["games"] for t in sorted_t]

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(sorted_t, wr_curve, "g-o", markersize=3, alpha=0.8)
                ax.set_xlabel("Trial")
                ax.set_ylabel("Win rate (all opponents)")
                ax.set_title("Win Rate over Trials (Mixed Pool)")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(PLOT_DIR / "win_rate_over_trials.png", dpi=150)
                plt.close(fig)
                print("  ✓ win_rate_over_trials.png")

                # Per-opponent win rate bar chart (aggregated over all trials)
                opp_stats: dict = {}
                for row in rows:
                    opp = row["opponent"]
                    opp_stats.setdefault(opp, {"wins": 0, "games": 0})
                    opp_stats[opp]["games"] += 1
                    if row["result"] == "win":
                        opp_stats[opp]["wins"] += 1

                opp_names_plot = list(opp_stats.keys())
                opp_wr = [
                    opp_stats[n]["wins"] / max(1, opp_stats[n]["games"])
                    for n in opp_names_plot
                ]
                opp_games = [opp_stats[n]["games"] for n in opp_names_plot]

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(opp_names_plot, opp_wr,
                              color=["steelblue", "tomato", "seagreen"][:len(opp_names_plot)])
                for bar, g in zip(bars, opp_games):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"n={g}",
                        ha="center", va="bottom", fontsize=9,
                    )
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Win rate")
                ax.set_title("Win Rate vs Each Opponent (all trials)")
                ax.grid(alpha=0.3, axis="y")
                fig.tight_layout()
                fig.savefig(PLOT_DIR / "per_opponent_win_rate.png", dpi=150)
                plt.close(fig)
                print("  ✓ per_opponent_win_rate.png")

        except Exception as e:
            print(f"  ⚠ win-rate plots skipped: {e}")

    # ── 4. Parameter importances ──────────────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        names  = list(importances.keys())
        values = list(importances.values())
        colors = cm.viridis(np.linspace(0.2, 0.9, len(names)))
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlabel("Importance (Fanova)")
        ax.set_title("Parameter Importances — RahulMCTS")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "param_importances.png", dpi=150)
        plt.close(fig)
        print("  ✓ param_importances.png")
    except Exception as e:
        print(f"  ⚠ importances skipped: {e}")

    # ── 5. Slice plots ────────────────────────────────────────────────────────
    plotted = 0
    for pname, pvals in params_all.items():
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(pvals, scores, c=trial_nums, cmap="viridis",
                            alpha=0.7, s=25)
            plt.colorbar(sc, ax=ax, label="Trial #")
            ax.set_xlabel(pname)
            ax.set_ylabel("Score")
            ax.set_title(f"Score vs {pname}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"slice_{pname}.png", dpi=120)
            plt.close(fig)
            plotted += 1
        except Exception:
            pass
    print(f"  ✓ slice_*.png ({plotted} params)")

    # ── 6. Contour plots for top-4 pairs ──────────────────────────────────────
    try:
        top_params = list(optuna.importance.get_param_importances(study).keys())[:4]
        n_contour  = 0
        for i, p1 in enumerate(top_params):
            for p2 in top_params[i+1:]:
                fig, ax = plt.subplots(figsize=(7, 5))
                sc = ax.scatter(params_all[p1], params_all[p2],
                                c=scores, cmap="RdYlGn", s=50, alpha=0.8)
                plt.colorbar(sc, ax=ax, label="Score")
                ax.set_xlabel(p1)
                ax.set_ylabel(p2)
                ax.set_title(f"Contour — {p1} × {p2}")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(PLOT_DIR / f"contour_{p1}_vs_{p2}.png", dpi=120)
                plt.close(fig)
                n_contour += 1
        print(f"  ✓ contour_*.png ({n_contour} pairs)")
    except Exception as e:
        print(f"  ⚠ contour plots skipped: {e}")

    # ── 7. Parameter correlation heatmap ──────────────────────────────────────
    try:
        keys    = list(params_all.keys())
        mat     = np.array([params_all[k] for k in keys])
        corr    = np.corrcoef(mat)
        n       = len(keys)
        fig, ax = plt.subplots(figsize=(n + 2, n + 2))
        im      = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(keys, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(corr[i, j]) > 0.5 else "black")
        ax.set_title("Parameter Correlation Matrix")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "param_correlation.png", dpi=150)
        plt.close(fig)
        print("  ✓ param_correlation.png")
    except Exception as e:
        print(f"  ⚠ correlation heatmap skipped: {e}")

    print("─────────────────────────────────────────────────────────────────")


# =============================================================================
# PRINT + SAVE BEST PARAMS
# =============================================================================

def print_best(study: optuna.Study):
    bt = study.best_trial
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print( "║              BEST TRIAL                                      ║")
    print(f"║  Trial #{bt.number:<5}   Score: {bt.value:.6f}                        ║")
    print( "╠══════════════════════════════════════════════════════════════╣")
    for k, v in bt.params.items():
        line = f"║  {k:<25} = {v}"
        print(f"{line:<63}║")
    print( "╚══════════════════════════════════════════════════════════════╝")

    out = RESULTS_DIR / "best_params.txt"
    with out.open("w") as f:
        f.write(f"# Best Trial #{bt.number}  Score={bt.value:.6f}\n\n")
        f.write("# Paste these into rahul_mcts_tunable.py defaults\n")
        f.write("# or set as env vars before launching the server.\n\n")
        for k, v in bt.params.items():
            env_key = f"RAHUL_{k}"
            f.write(f"{env_key}={v}\n")
    print(f"\nBest params → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=15),
        pruner=MedianPruner(n_startup_trials=12, n_warmup_steps=5),
        load_if_exists=True,
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)

    if remaining == 0:
        print(f"Study already has {done} completed trials (≥ N_TRIALS={N_TRIALS}).")
        print("Delete rahul_results/rahul_optuna.db to start fresh.\n")
        generate_plots(study)
        print_best(study)
        return

    print(f"Resuming study: {done} done, {remaining} remaining → "
          f"distributing across {N_WORKERS} workers\n")

    trials_per_worker = [remaining // N_WORKERS] * N_WORKERS
    for i in range(remaining % N_WORKERS):
        trials_per_worker[i] += 1

    t0 = time.time()

    Parallel(n_jobs=N_WORKERS, prefer="processes", backend="loky", verbose=10)(
        delayed(_worker)(worker_id, n)
        for worker_id, n in enumerate(trials_per_worker)
    )

    elapsed = time.time() - t0
    print(f"\nTuning finished in {elapsed/60:.1f} min  ({elapsed:.0f} s total)",
          flush=True)

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    generate_plots(study)
    print_best(study)


if __name__ == "__main__":
    main()