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
GAMES_PER_TRIAL = 1  
BOARD_SIZE = 11
N_TRIALS = 40         

# Ports & Commands
BASELINE_PORT = "8000"
TUNING_PORT = "8001"
BASELINE_CMD = ["python", "heuristic_agent.py"]
TUNING_CMD = ["python", "vanilla_mcts_agent.py"]  

LOG_PATH = Path("vanilla_optuna_match.json")
# THE NEW CSV LOG FILE
CSV_FILENAME = "vanilla_optuna_results.csv"

def load_last_state(path: Path):
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines: return None
    for line in reversed(lines):
        if not line.strip(): continue
        try: obj = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(obj, dict) and "turn" in obj: return obj
    return None

def objective(trial):
    c_param = trial.suggest_float("C_PARAM", 0.1, 3.0)
    depth_limit = trial.suggest_int("DEPTH_LIMIT", 3, 20)

    print(f"\n--- Trial {trial.number} | C={c_param:.3f}, Depth={depth_limit} ---")

    env_tune = os.environ.copy()
    env_tune["MCTS_C_PARAM"] = str(c_param)
    env_tune["MCTS_DEPTH_LIMIT"] = str(depth_limit)
    env_tune["PORT"] = TUNING_PORT

    tuning_server = subprocess.Popen(TUNING_CMD, env=env_tune, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    mcts_wins = 0
    heuristic_wins = 0
    draws = 0

    try:
        for game_num in range(GAMES_PER_TRIAL):
            if LOG_PATH.exists(): LOG_PATH.unlink()

            cli_cmd = [
                "battlesnake", "play", 
                "-W", str(BOARD_SIZE), "-H", str(BOARD_SIZE),
                "-g", "standard", "--timeout", "1000",
                "--name", "Vanilla_MCTS", "--url", f"http://127.0.0.1:{TUNING_PORT}",
                "--name", "Heuristic", "--url", f"http://127.0.0.1:{BASELINE_PORT}",
                "--output", str(LOG_PATH)
            ]
            
            subprocess.run(cli_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            final_state = load_last_state(LOG_PATH)
            if final_state:
                surviving_snakes = final_state["board"]["snakes"]
                if len(surviving_snakes) == 1:
                    if surviving_snakes[0]["name"] == "Vanilla_MCTS":
                        mcts_wins += 1
                    else:
                        heuristic_wins += 1
                else: draws += 1
            else: draws += 1

            if game_num == 2 and mcts_wins == 0 and draws == 0:
                print("    [Pruned] Agent lost 3 straight. Aborting to save time.")
                break

        score = (mcts_wins + (0.5 * draws)) / GAMES_PER_TRIAL
        print(f"Result: {mcts_wins} Wins | {heuristic_wins} Losses | {draws} Draws | Score: {score:.3f}")

        # ==========================================
        # LIVE DATA SAVING (CRASH-PROOF)
        # ==========================================
        # Append the results to the CSV immediately after the trial finishes
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                trial.number, 
                round(c_param, 3), 
                depth_limit, 
                mcts_wins, 
                heuristic_wins, 
                draws, 
                round(score, 3)
            ])

        return score

    finally:
        tuning_server.terminate()
        tuning_server.wait()

def main():
    print("Initializing Optuna Study for VANILLA MCTS...")
    
    # Initialize the CSV file and write the headers before starting the loop
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial_Number', 'C_PARAM', 'DEPTH_LIMIT', 'MCTS_Wins', 'Heuristic_Losses', 'Draws', 'Optuna_Score'])

    env_baseline = os.environ.copy()
    env_baseline["PORT"] = BASELINE_PORT
    baseline_server = subprocess.Popen(BASELINE_CMD, env=env_baseline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        study = optuna.create_study(direction="maximize", study_name="vanilla_mcts_tuning")
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n" + "="*50)
        print("🏆 VANILLA OPTIMIZATION COMPLETE 🏆")
        print("="*50)
        print(f"Best Trial Score: {study.best_trial.value:.3f}")
        for key, value in study.best_trial.params.items():
            if isinstance(value, float): print(f"  {key}: {value:.3f}")
            else: print(f"  {key}: {value}")

        # ==========================================
        # THE MASTER DUMP
        # ==========================================
        # Save Optuna's internal DataFrame (includes trial timestamps and status)
        df = study.trials_dataframe()
        df.to_csv("vanilla_optuna_master_dump.csv", index=False)
        print(f"\nSaved live results to: {CSV_FILENAME}")
        print(f"Saved complete Optuna metadata to: vanilla_optuna_master_dump.csv")

    finally:
        baseline_server.terminate()
        baseline_server.wait()
        if LOG_PATH.exists(): LOG_PATH.unlink()

if __name__ == "__main__":
    main()