"""
tournament_runner.py
====================
Runs tournament games for BattleSnakes Phase 4.

CURRENT SETUP: Experiment 7 only (all 4 agents simultaneously).
After this completes, comment out Exp 7 and uncomment the next
experiment you want to run. Results accumulate in experiment_map.csv.

AGENTS (keep these running in 4 separate terminals before running this):
    Port 8000 → heuristic_agent.py    (Snake1)
    Port 8001 → mcts.py               (Snake2)
    Port 8002 → mcts_variant.py       (Snake3)
    Port 8003 → vanilla_mcts.py       (Snake4)
"""

import csv
import json
import random
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOARD_W           = 11
BOARD_H           = 11
TIMEOUT_MS        = 1000
MAX_TURNS         = 300
POLL_INTERVAL     = 0.15
GAME_HARD_TIMEOUT = 400
LOG_PATH          = Path("game.json")
EXP_MAP_CSV       = Path("logs/experiment_map.csv")
GAMES_PER_EXP     = 50

# ---------------------------------------------------------------------------
# Experiments
# Uncomment ONE experiment at a time.
# Run this file, wait for it to finish, then uncomment the next one.
# All results accumulate safely in experiment_map.csv and game_summaries.csv
# ---------------------------------------------------------------------------

EXPERIMENTS = {

    # ── Run this first ───────────────────────────────────────────────────
    #"7_AllFour": [
    #   ("Snake1", 8000),
    #  ("Snake2", 8001),
    #    ("Snake3", 8002),
    #    ("Snake4", 8003),
    #], 

    #── Then uncomment one at a time below ───────────────────────────────

    # "1_Heuristic_vs_VanillaMCTS": [
    #     ("Snake1", 8000),
    #    ("Snake4", 8003),
    #     ("Snake1", 8000),
    #     ("Snake4", 8003),
    # ],

    # "2_Heuristic_vs_MCTS": [
    #     ("Snake1", 8000),
    #     ("Snake2", 8001),
    #     ("Snake1", 8000),
    #     ("Snake2", 8001),
    # ],

    # "3_Heuristic_vs_MCTSVar": [
    #     ("Snake1", 8000),
    #     ("Snake3", 8002),
    #     ("Snake1", 8000),
    #     ("Snake3", 8002),
    # ],

    # "4_VanillaMCTS_vs_MCTS": [
    #     ("Snake4", 8003),
    #     ("Snake2", 8001),
    #     ("Snake4", 8003),
    #     ("Snake2", 8001),
    # ],

    # "5_VanillaMCTS_vs_MCTSVar": [
    #     ("Snake4", 8003),
    #     ("Snake3", 8002),
    #     ("Snake4", 8003),
    #     ("Snake3", 8002),
    # ],

     "6_MCTS_vs_MCTSVar": [
         ("Snake2", 8001),
         ("Snake3", 8002),
         ("Snake2", 8001),
         ("Snake3", 8002),
     ],
}

# ---------------------------------------------------------------------------
# Experiment map CSV  — only thing tournament_runner writes
# ---------------------------------------------------------------------------

EXP_MAP_FIELDS = ["experiment", "game_index", "game_id", "seed"]


def ensure_exp_map():
    EXP_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not EXP_MAP_CSV.exists():
        with open(EXP_MAP_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EXP_MAP_FIELDS).writeheader()
        print(f"Created {EXP_MAP_CSV}")


def append_exp_map(row: dict):
    with open(EXP_MAP_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EXP_MAP_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def build_cmd(snakes: list, seed: int) -> list:
    cmd = [
        "battlesnake", "play",
        "-W", str(BOARD_W),
        "-H", str(BOARD_H),
        "-g", "standard",
        "-m", "hz_hazard_pits",
        "--foodSpawnChance", "25",
        "--minimumFood",     "2",
        "--seed",            str(seed),
        "--timeout",         str(TIMEOUT_MS),
        "--output",          str(LOG_PATH),
        # NO --browser
    ]
    for name, port in snakes:
        cmd += ["--name", name, "--url", f"http://127.0.0.1:{port}"]
    return cmd


def read_last_state(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "turn" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_one_game(experiment: str, game_index: int, snakes: list) -> bool:
    seed = random.randint(1, 99999)

    if LOG_PATH.exists():
        LOG_PATH.unlink()

    cmd   = build_cmd(snakes, seed)
    label = f"[{experiment}] game {game_index:>3}"
    print(f"  {label} | seed={seed:>5} | ", end="", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline    = time.time() + GAME_HARD_TIMEOUT
    last_turn   = -1
    game_id     = "unknown"
    final_state = None

    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                state = read_last_state(LOG_PATH)
                if state:
                    final_state = state
                    game_id = state.get("game", {}).get("id", "unknown")
                break

            state = read_last_state(LOG_PATH)
            if state:
                turn = int(state.get("turn", -1))
                if game_id == "unknown":
                    game_id = state.get("game", {}).get("id", "unknown")
                if turn != last_turn:
                    last_turn   = turn
                    final_state = state
                    print(
                        f"\r  {label} | seed={seed:>5} | turn={turn:<4}",
                        end="", flush=True,
                    )
                if turn >= MAX_TURNS:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            time.sleep(POLL_INTERVAL)
        else:
            print(f" — TIMED OUT, skipping")
            proc.kill()
            return False

    finally:
        if proc.poll() is None:
            proc.kill()

    if final_state is None or game_id == "unknown":
        print(f" — no state parsed, skipping")
        return False

    append_exp_map({
        "experiment": experiment,
        "game_index": game_index,
        "game_id":    game_id,
        "seed":       seed,
    })

    print(f" — done | turns={final_state.get('turn', '?')}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_exp_map()

    total  = len(EXPERIMENTS) * GAMES_PER_EXP
    played = 0
    failed = 0

    print("=" * 65)
    print(f"  BattleSnakes Tournament")
    print(f"  {len(EXPERIMENTS)} experiment(s) x {GAMES_PER_EXP} games = {total} planned")
    print(f"  Experiment map → {EXP_MAP_CSV}")
    print(f"  Game summaries → logs/game_summaries.csv  (written by agents)")
    print("=" * 65)

    for exp_name, snake_config in EXPERIMENTS.items():
        print(f"\n{'─'*65}")
        print(f"  Experiment : {exp_name}")
        print(f"  Snakes     : {[n for n, _ in snake_config]}")
        print(f"  Games      : {GAMES_PER_EXP}")
        print(f"{'─'*65}")

        for i in range(1, GAMES_PER_EXP + 1):
            success = run_one_game(exp_name, i, snake_config)
            if success:
                played += 1
            else:
                failed += 1
            time.sleep(2.0)

    print(f"\n{'='*65}")
    print(f"  Done — {played} played, {failed} failed/skipped")
    print(f"  Next step: python analyze_results.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
