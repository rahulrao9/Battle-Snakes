"""
tune_mcts3_optuna.py
====================
Parallelized Optuna tuner for mcts_agent3.py.

Opponent pool (anti-overfit)
----------------------------
Each game draws its opponent at random from three agents:
  • heuristic_agent.py      — fast greedy heuristic, no tree search
  • vanilla_mcts.py         — simple rule-based agent
  • rahul_mcts_tunable.py   — strongest opponent; MCTS with RAVE + PB

Tuning against a diverse pool prevents Optuna from over-specialising toward
any single opponent's quirks.

Parameters tuned
----------------
  C_PARAM                  — UCB1 exploration constant
  DEPTH_LIMIT              — rollout depth cap
  PB_WEIGHT                — progressive bias weight (root only)
  EARLY_GAME_TARGET_LENGTH — length threshold for food-chasing rollout guide

All four are injected into mcts_agent3.py via environment variables.
mcts_agent3.py must read them at module load time — see the patch note at the
bottom of this file if it doesn't already do so.
"""

# ── std-lib ───────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import time
import random
import subprocess
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
# Worker k occupies:
#   BASE_PORT + k*10 + 0  → opponent
#   BASE_PORT + k*10 + 1  → mcts3 (target)
BASE_PORT          = 9000     # offset from rahul tuner (8000) to avoid collisions

MCTS3_TIME_MS      = 650      # matches mcts_agent3 default budget
TIMEOUT_BUFFER_MS  = 100
CLI_TIMEOUT_MS     = MCTS3_TIME_MS + TIMEOUT_BUFFER_MS

BATTLESNAKE_BIN    = "./battlesnake"

TARGET_SCRIPT = "mcts_agent3.py"
TARGET_NAME   = "MCTS3"

# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------
OPPONENT_POOL = [
    ("Heuristic",  "heuristic_agent.py"),
    ("Vanilla",    "vanilla_mcts.py"),
    ("RahulMCTS",  "rahul_mcts_tunable.py"),
]

RESULTS_DIR   = Path("mcts3_tuning_results")
MATCH_LOG_DIR = RESULTS_DIR / "match_logs"
PLOT_DIR      = RESULTS_DIR / "plots"
CSV_PATH      = RESULTS_DIR / "mcts3_results.csv"
DB_PATH       = RESULTS_DIR / "mcts3_optuna.db"
STORAGE_URL   = f"sqlite:///{DB_PATH}"
STUDY_NAME    = "mcts3_tune"


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
    print(f"\n  CLI timeout : {CLI_TIMEOUT_MS} ms  (mcts3 budget: {MCTS3_TIME_MS} ms)")
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
                "C_PARAM", "DEPTH_LIMIT", "PB_WEIGHT", "EARLY_GAME_TARGET_LENGTH",
                "result", "survival", "final_length", "optuna_score",
            ])


# =============================================================================
# MATCH LOG PARSER  (identical logic to rahul tuner)
# =============================================================================

def parse_match(log_path: Path, target_name: str):
    """Returns (survival_score ∈ [0,1], winner_name, final_length)."""
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
    mcts3_env:    dict,
    opp_port:     int,
    mcts3_port:   int,
) -> dict:
    """
    Spawns opponent + mcts3, runs one game, returns result dict.
    RahulMCTS opponent needs no special env vars; it uses its baked-in defaults.
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
    mcts3_proc = None

    try:
        opp_proc = subprocess.Popen(
            ["python", opp_script],
            env=_env(opp_port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mcts3_proc = subprocess.Popen(
            ["python", TARGET_SCRIPT],
            env=mcts3_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)

        cli = [
            BATTLESNAKE_BIN, "play",
            "-W", str(BOARD_SIZE), "-H", str(BOARD_SIZE),
            "-g", "standard",
            "-m", "hz_hazard_pits",
            "--timeout", str(CLI_TIMEOUT_MS),
            "--name", opp_name,   "--url", f"http://127.0.0.1:{opp_port}",
            "--name", TARGET_NAME, "--url", f"http://127.0.0.1:{mcts3_port}",
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
            "opponent": opp_name,
            "result":   result,
            "survival": survival,
            "length":   final_len,
        }

    finally:
        for p in (opp_proc, mcts3_proc):
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    pass


# =============================================================================
# TRIAL RUNNER
# =============================================================================

def run_trial(worker_id: int, params: dict, trial_number: int) -> dict:
    """
    Plays GAMES_PER_TRIAL games against randomly selected opponents.

    Port layout:
      BASE_PORT + worker_id*10 + 0  → opponent
      BASE_PORT + worker_id*10 + 1  → mcts3 (target)
    """
    pb         = BASE_PORT + worker_id * 10
    opp_port   = pb + 0
    mcts3_port = pb + 1

    mcts3_env = os.environ.copy()
    mcts3_env.update({
        "PORT":                          str(mcts3_port),
        "MCTS3_TIME_LIMIT_MS":           str(MCTS3_TIME_MS),
        "MCTS3_C_PARAM":                 str(params["C_PARAM"]),
        "MCTS3_DEPTH_LIMIT":             str(params["DEPTH_LIMIT"]),
        "MCTS3_PB_WEIGHT":               str(params["PB_WEIGHT"]),
        "MCTS3_EARLY_GAME_TARGET_LENGTH": str(params["EARLY_GAME_TARGET_LENGTH"]),
    })

    wins = draws = losses = 0
    total_surv   = 0.0
    total_length = 0.0
    games_played = 0

    per_opp_wins:  dict = {name: 0 for name, _ in OPPONENT_POOL}
    per_opp_games: dict = {name: 0 for name, _ in OPPONENT_POOL}
    game_records:  list = []

    rng     = random.Random(trial_number * 1000 + worker_id)
    opp_seq = [rng.choice(OPPONENT_POOL) for _ in range(GAMES_PER_TRIAL)]

    for g_idx, (opp_name, opp_script) in enumerate(opp_seq):
        result_dict = run_one_game(
            worker_id    = worker_id,
            game_index   = g_idx,
            trial_number = trial_number,
            opp_name     = opp_name,
            opp_script   = opp_script,
            mcts3_env    = mcts3_env,
            opp_port     = opp_port,
            mcts3_port   = mcts3_port,
        )

        result   = result_dict["result"]
        survival = result_dict["survival"]
        length   = result_dict["length"]

        total_surv   += survival
        total_length += min(1.0, max(0.0, (length - 3) / 12.0))
        games_played += 1

        if result == "win":
            wins += 1
            per_opp_wins[opp_name] += 1
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

        # Early stopping: abort if clearly losing at midpoint
        if g_idx == 10 and wins / games_played < 0.25:
            break

    n          = games_played
    avg_surv   = total_surv   / n
    avg_length = total_length / n
    win_rate   = wins         / n
    score      = avg_surv * 0.15 + avg_length * 0.15 + win_rate * 0.70

    return {
        "score":         score,
        "avg_surv":      avg_surv,
        "avg_length":    avg_length,
        "win_rate":      win_rate,
        "wins":          wins,
        "draws":         draws,
        "losses":        losses,
        "games":         n,
        "per_opp_wins":  per_opp_wins,
        "per_opp_games": per_opp_games,
        "game_records":  game_records,
    }


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(trial: optuna.Trial, worker_id: int) -> float:
    params = {
        "C_PARAM":                  trial.suggest_float("C_PARAM",                  0.3,   3.0),
        "DEPTH_LIMIT":              trial.suggest_int(  "DEPTH_LIMIT",              3,     14),
        "PB_WEIGHT":                trial.suggest_float("PB_WEIGHT",                1.0,   20.0),
        "EARLY_GAME_TARGET_LENGTH": trial.suggest_int(  "EARLY_GAME_TARGET_LENGTH", 5,     25),
    }

    print(
        f"  [W{worker_id}|T{trial.number}] "
        f"C={params['C_PARAM']:.2f} D={params['DEPTH_LIMIT']} "
        f"PB={params['PB_WEIGHT']:.1f} EGL={params['EARLY_GAME_TARGET_LENGTH']}",
        flush=True,
    )

    metrics = run_trial(worker_id, params, trial.number)
    score   = metrics["score"]

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

    # CSV: one row per game
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
                        round(params["C_PARAM"],                  4),
                        params["DEPTH_LIMIT"],
                        round(params["PB_WEIGHT"],                4),
                        params["EARLY_GAME_TARGET_LENGTH"],
                        rec["result"],
                        round(rec["survival"],                    4),
                        rec["length"],
                        round(row_score,                          4),
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
    ax.set_title("Optimization History — MCTS3 vs Mixed Opponent Pool")
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
    ax.set_title("Score Distribution — MCTS3")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ score_distribution.png")

    # ── 3. Win rate over trials + per-opponent breakdown ──────────────────────
    if CSV_PATH.exists():
        try:
            rows = []
            with CSV_PATH.open() as f:
                for row in csv.DictReader(f):
                    rows.append(row)

            if rows:
                # Overall win rate per trial
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
                ax.set_title("Win Rate over Trials — MCTS3 (Mixed Pool)")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(PLOT_DIR / "win_rate_over_trials.png", dpi=150)
                plt.close(fig)
                print("  ✓ win_rate_over_trials.png")

                # Per-opponent aggregate win rate bar chart
                opp_stats: dict = {}
                for row in rows:
                    opp = row["opponent"]
                    opp_stats.setdefault(opp, {"wins": 0, "games": 0})
                    opp_stats[opp]["games"] += 1
                    if row["result"] == "win":
                        opp_stats[opp]["wins"] += 1

                opp_names_plot = list(opp_stats.keys())
                opp_wr    = [opp_stats[n]["wins"] / max(1, opp_stats[n]["games"]) for n in opp_names_plot]
                opp_games = [opp_stats[n]["games"] for n in opp_names_plot]

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(
                    opp_names_plot, opp_wr,
                    color=["steelblue", "tomato", "seagreen"][:len(opp_names_plot)],
                )
                for bar, g in zip(bars, opp_games):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"n={g}", ha="center", va="bottom", fontsize=9,
                    )
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Win rate")
                ax.set_title("Win Rate vs Each Opponent (all trials) — MCTS3")
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
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names, values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlabel("Importance (Fanova)")
        ax.set_title("Parameter Importances — MCTS3")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "param_importances.png", dpi=150)
        plt.close(fig)
        print("  ✓ param_importances.png")
    except Exception as e:
        print(f"  ⚠ importances skipped: {e}")

    # ── 5. Slice plots (score vs each param) ──────────────────────────────────
    plotted = 0
    for pname, pvals in params_all.items():
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(pvals, scores, c=trial_nums, cmap="viridis",
                            alpha=0.7, s=25)
            plt.colorbar(sc, ax=ax, label="Trial #")
            ax.set_xlabel(pname)
            ax.set_ylabel("Score")
            ax.set_title(f"Score vs {pname} — MCTS3")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"slice_{pname}.png", dpi=120)
            plt.close(fig)
            plotted += 1
        except Exception:
            pass
    print(f"  ✓ slice_*.png ({plotted} params)")

    # ── 6. Contour plots — all 4C2 = 6 pairs (small param space, do them all) ─
    try:
        param_names = list(params_all.keys())
        n_contour   = 0
        for i, p1 in enumerate(param_names):
            for p2 in param_names[i+1:]:
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
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(keys, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(corr[i, j]) > 0.5 else "black")
        ax.set_title("Parameter Correlation Matrix — MCTS3")
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
    print( "║              BEST TRIAL — MCTS3                              ║")
    print(f"║  Trial #{bt.number:<5}   Score: {bt.value:.6f}                        ║")
    print( "╠══════════════════════════════════════════════════════════════╣")
    for k, v in bt.params.items():
        line = f"║  {k:<30} = {v}"
        print(f"{line:<63}║")
    print( "╚══════════════════════════════════════════════════════════════╝")

    out = RESULTS_DIR / "best_params.txt"
    with out.open("w") as f:
        f.write(f"# Best Trial #{bt.number}  Score={bt.value:.6f}\n\n")
        f.write("# Paste these values into mcts_agent3.py, or export as env vars.\n\n")
        for k, v in bt.params.items():
            env_key = f"MCTS3_{k}"
            f.write(f"{env_key}={v}\n")
    print(f"\nBest params → {out}")

    # Also print a ready-to-paste Python snippet
    print("\n# ── copy-paste into mcts_agent3.py ──────────────────────────────")
    for k, v in bt.params.items():
        if isinstance(v, float):
            print(f"{k} = {v}")
        else:
            print(f"{k} = {v}")
    print("# ─────────────────────────────────────────────────────────────────")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=12),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        load_if_exists=True,
    )

    done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)

    if remaining == 0:
        print(f"Study already has {done} completed trials (≥ N_TRIALS={N_TRIALS}).")
        print("Delete mcts3_tuning_results/mcts3_optuna.db to start fresh.\n")
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


# =============================================================================
# PATCH NOTE — mcts_agent3.py env-var reading
# =============================================================================
# For this tuner to work, mcts_agent3.py must read its hyper-parameters from
# environment variables.  Replace the four hard-coded constants at the top of
# mcts_agent3.py with:
#
#   import os
#   C_PARAM                  = float(os.environ.get("MCTS3_C_PARAM",                  "1.452"))
#   DEPTH_LIMIT              = int(  os.environ.get("MCTS3_DEPTH_LIMIT",              "6"))
#   PB_WEIGHT                = float(os.environ.get("MCTS3_PB_WEIGHT",                "10.805"))
#   EARLY_GAME_TARGET_LENGTH = int(  os.environ.get("MCTS3_EARLY_GAME_TARGET_LENGTH", "15"))
#
# That's the only change needed in mcts_agent3.py itself.
# =============================================================================