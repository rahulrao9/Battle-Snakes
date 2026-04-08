"""
tournament_runner_parallel.py
==============================
Parallelised BattleSnakes tournament — 8 concurrent workers.

HOW IT WORKS
------------
Each worker slot (0-7) owns a dedicated port group:
    Worker 0 → ports 8000-8003
    Worker 1 → ports 8010-8013
    ...
    Worker 7 → ports 8070-8073

Slots inside each group:
    slot 0  →  heuristic_agent.py    (Snake1)
    slot 1  →  mcts_agent_final.py   (Snake2)
    slot 2  →  mcts_agent_variation.py (Snake3)
    slot 3  →  vanilla_mcts.py       (Snake4)

The runner launches and tears down each agent automatically — you do NOT
need to start them manually.

OUTPUT
------
    logs/experiment_map.csv   — game index / id / seed / experiment
    logs/game_summaries.csv   — per-snake result rows  (written by agents)

Run analysis afterwards:
    python analyze_results.py
"""

import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from joblib import Parallel, delayed
except ImportError:
    sys.exit("Missing: pip install joblib")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOARD_W           = 11
BOARD_H           = 11
TIMEOUT_MS        = 1000
MAX_TURNS         = 300
POLL_INTERVAL     = 0.15
GAME_HARD_TIMEOUT = 450        # seconds before a game is declared stuck
AGENT_BOOT_WAIT   = 3.0        # seconds to wait for Flask agents to start
GAMES_PER_EXP     = 100        # 700 games total across 7 experiments
N_WORKERS         = 8

LOGS_DIR          = Path("logs")
EXP_MAP_CSV       = LOGS_DIR / "experiment_map.csv"

# ---------------------------------------------------------------------------
# Agent definitions
# Adjust filenames to match your actual .py files.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR.parent
# slot → (readable_name, filename)
AGENT_SLOTS = {
    0: ("Snake1", str(ROOT_DIR / "heuristic_agent.py")),
    1: ("Snake2", str(ROOT_DIR / "mcts_agent_final.py")),
    2: ("Snake3", str(ROOT_DIR / "mcts_agent_variation.py")),
    3: ("Snake4", str(ROOT_DIR / "vanilla_mcts.py")),
}

def worker_port(worker_id: int, slot: int) -> int:
    """Return the port for a given worker / slot combination."""
    return 8000 + worker_id * 10 + slot


# ---------------------------------------------------------------------------
# Experiments
# Each value is a list of slot indices (length 4 for a standard 4-snake game).
# Repeat a slot to pit two copies of the same agent against each other.
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, list[int]] = {
    "1_Heuristic_vs_VanillaMCTS":  [0, 3, 0, 3],
    "2_Heuristic_vs_MCTS":         [0, 1, 0, 1],
    "3_Heuristic_vs_MCTSVar":      [0, 2, 0, 2],
    "4_VanillaMCTS_vs_MCTS":       [3, 1, 3, 1],
    "5_VanillaMCTS_vs_MCTSVar":    [3, 2, 3, 2],
    "6_MCTS_vs_MCTSVar":           [1, 2, 1, 2],
    "7_AllFour":                   [0, 1, 2, 3],
}

# ---------------------------------------------------------------------------
# Experiment-map CSV helpers
# ---------------------------------------------------------------------------

EXP_MAP_FIELDS = ["experiment", "game_index", "game_id", "seed", "worker_id"]

_csv_lock = None  # replaced with a real lock in main()


def ensure_exp_map():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not EXP_MAP_CSV.exists():
        with open(EXP_MAP_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EXP_MAP_FIELDS).writeheader()
        print(f"Created {EXP_MAP_CSV}")


def append_exp_map(row: dict):
    """Append one row — safe to call from multiple processes via file locking."""
    with open(EXP_MAP_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EXP_MAP_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# Agent process management
# ---------------------------------------------------------------------------

def start_agents(worker_id: int, needed_slots: set[int]) -> dict[int, subprocess.Popen]:
    """
    Launch Flask agent processes for the required slots.
    Returns {slot: Popen} mapping.
    """
    procs: dict[int, subprocess.Popen] = {}
    for slot in needed_slots:
        name, filename = AGENT_SLOTS[slot]
        port = worker_port(worker_id, slot)
        env = os.environ.copy()
        env["PORT"] = str(port)
        log_file = LOGS_DIR / f"agent_w{worker_id}_slot{slot}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                [sys.executable, filename],
                env=env,
                stdout=lf,
                stderr=lf,
            )
        procs[slot] = proc
        print(f"  [worker {worker_id}] Started {filename} on port {port} (pid={proc.pid})")
    return procs


def stop_agents(procs: dict[int, subprocess.Popen]):
    for slot, proc in procs.items():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def wait_for_agents(worker_id: int, needed_slots: set[int], timeout: float = 10.0):
    """Poll each agent's /ping endpoint until it responds or timeout expires."""
    try:
        import urllib.request
    except ImportError:
        time.sleep(AGENT_BOOT_WAIT)
        return

    deadline = time.time() + timeout
    remaining = set(needed_slots)
    while remaining and time.time() < deadline:
        for slot in list(remaining):
            port = worker_port(worker_id, slot)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                remaining.discard(slot)
            except Exception:
                pass
        if remaining:
            time.sleep(0.3)

    if remaining:
        # Fall back to a fixed sleep if ping didn't work
        time.sleep(AGENT_BOOT_WAIT)


# ---------------------------------------------------------------------------
# Game log helpers
# ---------------------------------------------------------------------------

def log_path_for(worker_id: int) -> Path:
    p = LOGS_DIR / f"worker_{worker_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p / "game.json"


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
# CLI command builder
# ---------------------------------------------------------------------------

def build_cmd(worker_id: int, slots: list[int], seed: int) -> list:
    log = str(log_path_for(worker_id))
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
        "--output",          log,
        # NO --browser  (headless)
    ]
    for i, slot in enumerate(slots):
        name, _ = AGENT_SLOTS[slot]
        # Give duplicate snakes a disambiguated name
        count = slots[:i].count(slot)
        display_name = name if count == 0 else f"{name}_b"
        port = worker_port(worker_id, slot)
        cmd += ["--name", display_name, "--url", f"http://127.0.0.1:{port}"]
    return cmd


# ---------------------------------------------------------------------------
# Single game runner  (called inside joblib worker)
# ---------------------------------------------------------------------------

def run_one_game(
    experiment: str,
    game_index: int,
    slots: list[int],
    worker_id: int,
    agent_procs: dict[int, subprocess.Popen],
) -> bool:
    """Run a single game and append its record to experiment_map.csv."""
    seed    = random.randint(1, 99_999)
    log_p   = log_path_for(worker_id)

    if log_p.exists():
        log_p.unlink()

    cmd   = build_cmd(worker_id, slots, seed)
    label = f"[W{worker_id}|{experiment}] #{game_index:>3}"

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
                state = read_last_state(log_p)
                if state:
                    final_state = state
                    game_id = state.get("game", {}).get("id", "unknown")
                break

            state = read_last_state(log_p)
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
            print(f"\n  {label} — TIMED OUT, skipping")
            proc.kill()
            return False

    finally:
        if proc.poll() is None:
            proc.kill()

    if final_state is None or game_id == "unknown":
        print(f"\n  {label} — no state parsed, skipping")
        return False

    # Write to experiment map (append — file already exists)
    append_exp_map({
        "experiment": experiment,
        "game_index": game_index,
        "game_id":    game_id,
        "seed":       seed,
        "worker_id":  worker_id,
    })
    print(f"\n  {label} — done | turns={final_state.get('turn', '?')}")
    return True


# ---------------------------------------------------------------------------
# Worker function  (one experiment × N games, run inside joblib)
# ---------------------------------------------------------------------------

def worker_run_experiment(
    experiment: str,
    slots: list[int],
    game_indices: list[int],
    worker_id: int,
) -> tuple[int, int]:
    """
    Start agents, run all assigned games, stop agents.
    Returns (played, failed).
    """
    played = 0
    failed = 0
    needed = set(slots)

    # Start only the agents we actually need
    agent_procs = start_agents(worker_id, needed)
    wait_for_agents(worker_id, needed)

    try:
        for idx in game_indices:
            ok = run_one_game(experiment, idx, slots, worker_id, agent_procs)
            if ok:
                played += 1
            else:
                failed += 1
            time.sleep(1.5)   # brief pause between games on the same worker
    finally:
        stop_agents(agent_procs)

    return played, failed


# ---------------------------------------------------------------------------
# Job builder: distribute games across workers
# ---------------------------------------------------------------------------

def build_jobs() -> list[tuple]:
    """
    Returns a list of (experiment, slots, [game_indices], worker_id) tuples.
    Games within each experiment are spread round-robin across workers.
    """
    jobs = []
    worker_id = 0
    for exp_name, slots in EXPERIMENTS.items():
        # Split GAMES_PER_EXP game indices across N_WORKERS batches
        all_indices = list(range(1, GAMES_PER_EXP + 1))
        batches: list[list[int]] = [[] for _ in range(N_WORKERS)]
        for i, idx in enumerate(all_indices):
            batches[i % N_WORKERS].append(idx)

        for w_offset, batch in enumerate(batches):
            if not batch:
                continue
            wid = (worker_id + w_offset) % N_WORKERS
            jobs.append((exp_name, slots, batch, wid))

        worker_id = (worker_id + N_WORKERS) % N_WORKERS  # rotate starting worker

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_exp_map()

    total_planned = len(EXPERIMENTS) * GAMES_PER_EXP
    jobs = build_jobs()

    print("=" * 70)
    print("  BattleSnakes Parallel Tournament")
    print(f"  Workers   : {N_WORKERS}")
    print(f"  Experiments: {len(EXPERIMENTS)}  ×  {GAMES_PER_EXP} games = {total_planned} planned")
    print(f"  Port range: 8000 – {8000 + N_WORKERS * 10 - 1}")
    print(f"  Exp map   : {EXP_MAP_CSV}")
    print(f"  Summaries : logs/game_summaries.csv  (written by agents)")
    print("=" * 70)
    print()

    for exp_name, slots in EXPERIMENTS.items():
        names = [AGENT_SLOTS[s][0] for s in slots]
        print(f"  {exp_name:<35}  snakes={names}")
    print()

    t0 = time.time()

    results = Parallel(n_jobs=N_WORKERS, backend="loky", verbose=5)(
        delayed(worker_run_experiment)(exp, slots, indices, wid)
        for exp, slots, indices, wid in jobs
    )

    elapsed = time.time() - t0
    total_played = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)

    print()
    print("=" * 70)
    print(f"  Finished in {elapsed/60:.1f} min")
    print(f"  Played : {total_played}")
    print(f"  Failed : {total_failed}")
    print(f"  Next   : python analyze_results.py")
    print("=" * 70)


if __name__ == "__main__":
    main()