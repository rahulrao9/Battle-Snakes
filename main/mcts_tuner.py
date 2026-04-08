import subprocess
import time
import os
import sys
import json
import csv
import optuna
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
GAMES_PER_TRIAL = 20   
BOARD_SIZE = 11
N_TRIALS = 50          

# Create the results directory
RESULTS_DIR = Path("v3_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Static Baseline Agents (Swapped V3 for V1)
AGENTS = {
    "heuristic": {"cmd": ["python", "heuristic_agent.py"], "port": "8000", "name": "Heuristic"},
    "vanilla":   {"cmd": ["python", "vanilla_mcts_agent.py"], "port": "8001", "name": "Vanilla"},
    "advance_1": {"cmd": ["python", "mcts_agent.py"], "port": "8002", "name": "Advance_V1"}
}

# The Dynamic Agent we are tuning (Now V3)
TUNING_CMD = ["python", "mcts_agent_advance_v3.py"]
TARGET_SNAKE_NAME = "Advance_V3"

BATTLESNAKE_BIN = "./battlesnake" 

LOG_PATH = RESULTS_DIR / "4player_optuna_match.json"
CSV_FILENAME = RESULTS_DIR / "4player_optuna_results_v3.csv"

# ==========================================
# PRE-FLIGHT CHECK
# ==========================================
def run_pre_flight_check():
    print("--- PRE-FLIGHT CHECK ---")
    all_clear = True
    
    for key, info in AGENTS.items():
        file_name = info["cmd"][1]
        if not os.path.exists(file_name):
            print(f"❌ ERROR: Cannot find '{file_name}'. Check your spelling or folder!")
            all_clear = False
        else:
            print(f"✅ Found {file_name}")
            
    tune_file = TUNING_CMD[1]
    if not os.path.exists(tune_file):
        print(f"❌ ERROR: Cannot find '{tune_file}' for the Tuning Agent.")
        all_clear = False
    else:
        print(f"✅ Found {tune_file}")
        
    if not all_clear:
        print("\n❌ Pre-flight failed! The script will now exit. Please fix the filenames in the code.")
        sys.exit(1)
    print("------------------------\n")

# ==========================================
# JSON PARSER
# ==========================================
def calculate_match_results(path: Path, target_name: str) -> tuple[float, str, int]:
    if not path.exists():
        return 0.0, "Draw", 3
        
    max_turn = 0
    last_seen_alive = {} 
    max_lengths = {} 
    
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
                name = s["name"]
                last_seen_alive[name] = turn
                max_lengths[name] = max(max_lengths.get(name, 3), s.get("length", 3))
                    
        except json.JSONDecodeError:
            continue
            
    if max_turn <= 1 or not last_seen_alive:
        return 0.0, "Draw", 3
        
    highest_turn = max(last_seen_alive.values())
    winners = [name for name, turn in last_seen_alive.items() if turn == highest_turn]
    
    if len(winners) > 1:
        winners.sort(key=lambda w: max_lengths.get(w, 0), reverse=True)
        if max_lengths.get(winners[0], 0) > max_lengths.get(winners[1], 0):
            winner_name = winners[0]
        else:
            winner_name = "Draw"
    else:
        winner_name = winners[0]
        
    target_last_alive_turn = last_seen_alive.get(target_name, 0)
    target_final_length = max_lengths.get(target_name, 3)
    
    survival_score = target_last_alive_turn / 300.0 
    
    return survival_score, winner_name, target_final_length


def objective(trial):
    c_param = trial.suggest_float("C_PARAM", 0.5, 3.0)
    depth_limit = trial.suggest_int("DEPTH_LIMIT", 6, 16)
    pb_weight = trial.suggest_float("PB_WEIGHT", 2.0, 15.0)
    target_length = trial.suggest_int("TARGET_LENGTH", 8, 20)

    dynamic_tuning_port = str(8103 + trial.number)

    print(f"\n--- Trial {trial.number} | Port={dynamic_tuning_port} | C={c_param:.2f}, Depth={depth_limit}, PB={pb_weight:.1f}, TargetLen={target_length} ---")

    env_tune = os.environ.copy()
    env_tune["MCTS_C_PARAM"] = str(c_param)
    env_tune["MCTS_DEPTH_LIMIT"] = str(depth_limit)
    env_tune["MCTS_PB_WEIGHT"] = str(pb_weight)
    env_tune["MCTS_TARGET_LENGTH"] = str(target_length)
    env_tune["PORT"] = dynamic_tuning_port

    tuning_server = subprocess.Popen(TUNING_CMD, env=env_tune)
    time.sleep(2.0) 

    total_survival_score = 0.0
    total_length_score = 0.0
    
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
                "-g", "standard", "-m", "hz_hazard_pits", "--timeout", "1000",
                "--name", AGENTS["heuristic"]["name"], "--url", f"http://127.0.0.1:{AGENTS['heuristic']['port']}",
                "--name", AGENTS["vanilla"]["name"], "--url", f"http://127.0.0.1:{AGENTS['vanilla']['port']}",
                "--name", AGENTS["advance_1"]["name"], "--url", f"http://127.0.0.1:{AGENTS['advance_1']['port']}",
                "--name", TARGET_SNAKE_NAME, "--url", f"http://127.0.0.1:{dynamic_tuning_port}", 
                "--output", str(LOG_PATH)
            ]
            
            result = subprocess.run(cli_cmd, capture_output=True, text=True) 
            
            score, winner, final_length = calculate_match_results(LOG_PATH, TARGET_SNAKE_NAME)
            
            if score == 0.0 and winner == "Draw" and final_length == 3:
                print(f"    ❌ ENGINE CRASH! The Battlesnake engine aborted the match.")
                print(f"    Engine Error Log:\n{result.stderr}")
            
            total_survival_score += score
            
            growth = max(0, final_length - 3)
            length_score = min(1.0, growth / 12.0) 
            total_length_score += length_score
            
            if winner in win_counts:
                win_counts[winner] += 1
            else:
                win_counts["Draw"] += 1

            if game_num == 10 and (total_survival_score / (game_num+1)) < 0.1:
                print("    [Pruned] Agent is dying way too early. Aborting trial.")
                break

        avg_survival = total_survival_score / GAMES_PER_TRIAL
        avg_length = total_length_score / GAMES_PER_TRIAL
        target_win_rate = win_counts[TARGET_SNAKE_NAME] / GAMES_PER_TRIAL
        
        base_score = (avg_survival * 0.8) + (avg_length * 0.2)
        final_optuna_score = base_score + (target_win_rate * 0.2)
        
        print(f"Result: Target({win_counts[TARGET_SNAKE_NAME]}) | Heur({win_counts[AGENTS['heuristic']['name']]}) | Vanilla({win_counts[AGENTS['vanilla']['name']]}) | V1({win_counts[AGENTS['advance_1']['name']]}) | Draws({win_counts['Draw']})")
        print(f"Avg Survival (80%): {avg_survival:.3f} | Avg Length (20%): {avg_length:.3f} | FINAL SCORE: {final_optuna_score:.3f}")

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
                round(avg_survival, 3),
                round(avg_length, 3),
                round(final_optuna_score, 3)
            ])

        return final_optuna_score

    finally:
        tuning_server.terminate()
        tuning_server.wait()

def main():
    run_pre_flight_check()
    print("Initializing 4-Player Optuna Study for Advance V3...")
    
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Trial_Number', 'C_PARAM', 'DEPTH_LIMIT', 'PB_WEIGHT', 'TARGET_LENGTH', 
            'Target_Wins', 'Heuristic_Wins', 'Vanilla_Wins', 'Advance1_Wins', 'Draws', 
            'Avg_Survival_Score', 'Avg_Length_Score', 'Optuna_Composite_Score'
        ])

    static_processes = []
    for agent_key, info in AGENTS.items():
        print(f"Booting static server: {info['name']} on port {info['port']}...")
        env_static = os.environ.copy()
        env_static["PORT"] = info["port"]
        
        proc = subprocess.Popen(info["cmd"], env=env_static)
        static_processes.append(proc)
        
    time.sleep(4.0) 

    try:
        study = optuna.create_study(direction="maximize", study_name="v3_4player_tuning")
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n" + "="*50)
        print("🏆 FINAL V3 OPTIMIZATION COMPLETE 🏆")
        print("="*50)
        print(f"Best Trial Score: {study.best_trial.value:.3f}")
        for key, value in study.best_trial.params.items():
            if isinstance(value, float): print(f"  {key}: {value:.3f}")
            else: print(f"  {key}: {value}")
            
        # 1. Save the Master Dump
        df = study.trials_dataframe()
        df.to_csv(RESULTS_DIR / "4player_optuna_master_dump.csv", index=False)
        print(f"\n📁 Data saved to {RESULTS_DIR}/")

        # 2. Extract and Save Plots
        print("\nGenerating Optuna Plots for Scientific Report...")
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice

            # Plot 1: Optimization History
            fig_hist = plot_optimization_history(study)
            plt.savefig(RESULTS_DIR / "optimization_history.png", bbox_inches='tight')
            plt.close()

            # Plot 2: Parameter Importances (If enough trials have run successfully)
            try:
                fig_import = plot_param_importances(study)
                plt.savefig(RESULTS_DIR / "parameter_importance.png", bbox_inches='tight')
                plt.close()
            except Exception:
                pass # Fails if all trials scored identically

            # Plot 3: Parameter Slice Plot
            fig_slice = plot_slice(study)
            plt.savefig(RESULTS_DIR / "slice_distribution.png", bbox_inches='tight')
            plt.close()

            print("📊 Plots successfully generated as PNG files in the v3_results folder!")
            
        except ImportError:
            print("⚠️ Matplotlib is not installed. To generate graphs, run: pip install matplotlib")

    finally:
        print("\nShutting down all static servers...")
        for proc in static_processes:
            proc.terminate()
            proc.wait()
        if LOG_PATH.exists(): LOG_PATH.unlink()

if __name__ == "__main__":
    main()