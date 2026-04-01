import subprocess
import time
import os
import json
import csv
import optuna
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
GAMES_PER_TRIAL = 25    
BOARD_SIZE = 11
N_TRIALS = 30          

# Static Baseline Agents
AGENTS = {
    "heuristic": {"cmd": ["python", "heuristic_agent.py"], "port": "8000", "name": "Heuristic"},
    "vanilla":   {"cmd": ["python", "vanilla_mcts_agent.py"], "port": "8001", "name": "Vanilla"},
    "advance_1": {"cmd": ["python", "mcts_agent_advance.py"], "port": "8002", "name": "Advance_v1"}
}

# The Dynamic Agent we are tuning
TUNING_CMD = ["python", "mcts_agent_advance_v3.py"]
TUNING_PORT = "8003"
TARGET_SNAKE_NAME = "Advance_V3"

BATTLESNAKE_BIN = "/tmp/battlesnake-rules/battlesnake" 

LOG_PATH = Path("4player_optuna_match.json")
CSV_FILENAME = "4player_optuna_results_v3.csv"


# ==========================================
# THE FIX: Track exact death turns to find the true winner
# ==========================================
def calculate_match_results(path: Path, target_name: str) -> tuple[float, str]:
    if not path.exists():
        print(f"    [ERROR] {path} was not created! The game crashed before starting.")
        return 0.0, "Draw"
        
    max_turn = 0
    last_seen_alive = {} # Dictionary to track the last turn each snake was alive
    
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        
    for line in lines:
        if not line.strip(): continue
        try:
            state = json.loads(line)
            turn = state.get("turn", 0)
            max_turn = max(max_turn, turn)
            
            snakes = state.get("board", {}).get("snakes", [])
            for s in snakes:
                # Update the highest turn this snake has reached so far
                last_seen_alive[s["name"]] = turn
                    
        except json.JSONDecodeError:
            continue
            
    # If the game didn't run properly
    if max_turn == 0 or not last_seen_alive:
        return 0.0, "Draw"
        
    # Find out what the highest survived turn was
    highest_turn = max(last_seen_alive.values())
    
    # Find which snake(s) made it to that highest turn
    winners = [name for name, turn in last_seen_alive.items() if turn == highest_turn]
    
    if len(winners) == 1:
        winner_name = winners[0]
    else:
        winner_name = "Draw" # Multiple snakes died on the exact same final turn
        
    # Calculate the target snake's survival score
    target_last_alive_turn = last_seen_alive.get(target_name, 0)
    survival_score = target_last_alive_turn / max_turn if max_turn > 0 else 0.0
    
    return survival_score, winner_name


def objective(trial):
    c_param = trial.suggest_float("C_PARAM", 0.5, 3.0)
    depth_limit = trial.suggest_int("DEPTH_LIMIT", 6, 16)
    pb_weight = trial.suggest_float("PB_WEIGHT", 2.0, 15.0)
    target_length = trial.suggest_int("TARGET_LENGTH", 8, 20)

    print(f"\n--- Trial {trial.number} | C={c_param:.2f}, Depth={depth_limit}, PB={pb_weight:.1f}, TargetLen={target_length} ---")

    env_tune = os.environ.copy()
    env_tune["MCTS_C_PARAM"] = str(c_param)
    env_tune["MCTS_DEPTH_LIMIT"] = str(depth_limit)
    env_tune["MCTS_PB_WEIGHT"] = str(pb_weight)
    env_tune["MCTS_TARGET_LENGTH"] = str(target_length)
    env_tune["PORT"] = TUNING_PORT

    # Boot the tuning server
    tuning_server = subprocess.Popen(TUNING_CMD, env=env_tune)
    time.sleep(1.5)

    total_survival_score = 0.0
    
    win_counts = {
        TARGET_SNAKE_NAME: 0,
        AGENTS["heuristic"]["name"]: 0,
        AGENTS["vanilla"]["name"]: 0,
        AGENTS["advance_1"]["name"]: 0,
        "Draw": 0
    }

    try:
        for game_num in range(GAMES_PER_TRIAL):
            if LOG_PATH.exists(): LOG_PATH.unlink()

            cli_cmd = [
                BATTLESNAKE_BIN, "play", 
                "-W", str(BOARD_SIZE), "-H", str(BOARD_SIZE),
                "-g", "standard", "--timeout", "1000",
                "--name", AGENTS["heuristic"]["name"], "--url", f"http://127.0.0.1:{AGENTS['heuristic']['port']}",
                "--name", AGENTS["vanilla"]["name"], "--url", f"http://127.0.0.1:{AGENTS['vanilla']['port']}",
                "--name", AGENTS["advance_1"]["name"], "--url", f"http://127.0.0.1:{AGENTS['advance_1']['port']}",
                "--name", TARGET_SNAKE_NAME, "--url", f"http://127.0.0.1:{TUNING_PORT}",
                "--output", str(LOG_PATH)
            ]
            
            subprocess.run(cli_cmd, stdout=subprocess.DEVNULL) 
            
            # Use the new multi-return function
            score, winner = calculate_match_results(LOG_PATH, TARGET_SNAKE_NAME)
            total_survival_score += score
            
            # Log the win for the specific agent
            if winner in win_counts:
                win_counts[winner] += 1
            else:
                win_counts["Draw"] += 1

            if game_num == 12 and total_survival_score < 0.3:
                print("    [Pruned] Agent is dying way too early. Aborting trial.")
                break

        avg_score = total_survival_score / GAMES_PER_TRIAL
        
        # Print breakdown of who won to the terminal
        print(f"Result: Target({win_counts[TARGET_SNAKE_NAME]}) | Heur({win_counts[AGENTS['heuristic']['name']]}) | Vanilla({win_counts[AGENTS['vanilla']['name']]}) | V3({win_counts[AGENTS['advance_1']['name']]}) | Draws({win_counts['Draw']})")
        print(f"Avg Target Survival Score: {avg_score:.3f}")

        # Update CSV writing to include all the new agent columns
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                trial.number, round(c_param, 3), depth_limit, 
                round(pb_weight, 3), target_length, 
                win_counts[TARGET_SNAKE_NAME],
                win_counts[AGENTS['heuristic']['name']],
                win_counts[AGENTS['vanilla']['name']],
                win_counts[AGENTS['advance_1']['name']],
                win_counts["Draw"],
                round(avg_score, 3)
            ])

        return avg_score

    finally:
        tuning_server.terminate()
        tuning_server.wait()

def main():
    print("Initializing 4-Player Optuna Study for Advance V1...")
    
    # Update CSV Header definitions
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Trial_Number', 'C_PARAM', 'DEPTH_LIMIT', 'PB_WEIGHT', 'TARGET_LENGTH', 
            'Target_Wins', 'Heuristic_Wins', 'Vanilla_Wins', 'Advance3_Wins', 'Draws', 'Avg_Survival_Score'
        ])

    static_processes = []
    for agent_key, info in AGENTS.items():
        print(f"Booting static server: {info['name']} on port {info['port']}...")
        env_static = os.environ.copy()
        env_static["PORT"] = info["port"]
        
        # Booting the baselines
        proc = subprocess.Popen(info["cmd"], env=env_static)
        static_processes.append(proc)
        
    time.sleep(3) 

    try:
        study = optuna.create_study(direction="maximize", study_name="v1_4player_tuning")
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n" + "="*50)
        print("🏆 4-PLAYER V1 OPTIMIZATION COMPLETE 🏆")
        print("="*50)
        print(f"Best Trial Score: {study.best_trial.value:.3f} / 1.000")
        for key, value in study.best_trial.params.items():
            if isinstance(value, float): print(f"  {key}: {value:.3f}")
            else: print(f"  {key}: {value}")

        df = study.trials_dataframe()
        df.to_csv("4player_optuna_master_dump.csv", index=False)

    finally:
        print("\nShutting down all static servers...")
        for proc in static_processes:
            proc.terminate()
            proc.wait()
        if LOG_PATH.exists(): LOG_PATH.unlink()

if __name__ == "__main__":
    main()