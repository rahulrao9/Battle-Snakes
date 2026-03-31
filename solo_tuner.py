import subprocess
import time
import os
import csv
import itertools
import json
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
GAMES_PER_COMBO = 1  # Start with 10 for testing
BOARD_SIZE = 11

# Ports
BASELINE_PORT = "8000"
TUNING_PORT = "8001"

# The static baseline we are testing against
BASELINE_CMD = ["python", "heuristic_agent.py"]
LOG_PATH = Path("tuning_match.json")

# ==========================================
# CHOOSE WHICH AGENT TO TUNE HERE:
# ==========================================
TUNE_AGENT = "advance"  # Change to "vanilla" to tune the vanilla agent

if TUNE_AGENT == "advance":
    TUNING_CMD = ["python", "mcts_agent_advance.py"]
    CSV_FILENAME = "advance_tuning_results.csv"
    GRID = {
        "C_PARAM": ["0.5", "1.0", "1.414"],
        "DEPTH_LIMIT": ["8", "12", "15"],
        "PB_WEIGHT": ["3.0", "5.0", "10.0"] 
    }
else:
    TUNING_CMD = ["python", "vanilla_mcts_agent.py"]
    CSV_FILENAME = "vanilla_tuning_results.csv"
    GRID = {
        "C_PARAM": ["0.5", "1.414", "2.0"],
        "DEPTH_LIMIT": ["5", "10", "15"],
        "PB_WEIGHT": ["0.0"] 
    }

def load_last_state(path: Path):
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines: return None
    for line in reversed(lines):
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(obj, dict) and "turn" in obj:
            return obj
    return None

def run_tournament():
    keys, values = zip(*GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Tuning '{TUNE_AGENT.upper()}' Agent against Heuristic Baseline ---")
    print(f"Total combinations: {len(combinations)} | Games per combo: {GAMES_PER_COMBO}")
    print("-" * 60)

    # 1. Boot up the Baseline (Heuristic) Server once
    env_baseline = os.environ.copy()
    env_baseline["PORT"] = BASELINE_PORT
    print("Starting Heuristic Baseline Server on Port 8000...")
    baseline_server = subprocess.Popen(BASELINE_CMD, env=env_baseline)
    time.sleep(3) # Wait 3 seconds for server to fully initialize

    try:
        with open(CSV_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['C_PARAM', 'DEPTH_LIMIT', 'PB_WEIGHT', 'MCTS_Wins', 'Heuristic_Wins', 'Draws', 'Win_Rate'])

            for index, combo in enumerate(combinations):
                print(f"\n[Combo {index + 1}/{len(combinations)}]: C={combo['C_PARAM']}, Depth={combo['DEPTH_LIMIT']}, PB={combo['PB_WEIGHT']}")
                
                # Pass Hyperparameters via Environment Variables
                env_tune = os.environ.copy()
                env_tune["MCTS_C_PARAM"] = combo["C_PARAM"]
                env_tune["MCTS_DEPTH_LIMIT"] = combo["DEPTH_LIMIT"]
                env_tune["MCTS_PB_WEIGHT"] = combo["PB_WEIGHT"]
                env_tune["PORT"] = TUNING_PORT

                # 2. Boot up the MCTS Server
                tuning_server = subprocess.Popen(TUNING_CMD, env=env_tune)
                time.sleep(2) # Wait 2 seconds for MCTS server to initialize

                mcts_wins = 0
                heuristic_wins = 0
                draws = 0

                # 3. Play the matches
                for game_num in range(GAMES_PER_COMBO):
                    if LOG_PATH.exists(): LOG_PATH.unlink()

                    # THE FIX: Added --timeout 1000 and changed localhost to 127.0.0.1
                    cli_cmd = [
                        "battlesnake", "play", 
                        "-W", str(BOARD_SIZE), "-H", str(BOARD_SIZE),
                        "-g", "standard",
                        "--timeout", "1000",
                        "--name", "MCTS_Tuning", "--url", f"http://127.0.0.1:{TUNING_PORT}",
                        "--name", "Heuristic", "--url", f"http://127.0.0.1:{BASELINE_PORT}",
                        "--output", str(LOG_PATH)
                    ]
                    
                    subprocess.run(cli_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 4. Parse the exact winner using the JSON file
                    final_state = load_last_state(LOG_PATH)
                    if final_state:
                        turn_count = final_state.get("turn", 0)
                        
                        # DIAGNOSTIC CHECK: Did the game end immediately?
                        if turn_count <= 2:
                            print("    [WARNING] Game ended on Turn 1 or 2. Your agents might be crashing or timing out!")
                        
                        surviving_snakes = final_state["board"]["snakes"]
                        
                        if len(surviving_snakes) == 1:
                            winner_name = surviving_snakes[0]["name"]
                            if winner_name == "MCTS_Tuning":
                                mcts_wins += 1
                            elif winner_name == "Heuristic":
                                heuristic_wins += 1
                        else:
                            draws += 1
                    else:
                        print("    [Error] Could not read game.json. Assuming draw.")
                        draws += 1

                    print(f"  -> Game {game_num + 1}/{GAMES_PER_COMBO} | Score: MCTS {mcts_wins} - {heuristic_wins} Heuristic (Draws: {draws})")

                win_rate = (mcts_wins / GAMES_PER_COMBO) * 100
                print(f"Result for Combo {index + 1}: MCTS Win Rate: {win_rate:.1f}%")
                
                writer.writerow([
                    combo["C_PARAM"], combo["DEPTH_LIMIT"], combo["PB_WEIGHT"], 
                    mcts_wins, heuristic_wins, draws, f"{win_rate:.1f}%"
                ])
                file.flush()

                # Kill MCTS server before next loop
                tuning_server.terminate()
                tuning_server.wait()

    finally:
        print("\nShutting down Baseline Server...")
        baseline_server.terminate()
        baseline_server.wait()
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        print(f"Done. Check {CSV_FILENAME} for the best hyperparameters!")

if __name__ == "__main__":
    run_tournament()