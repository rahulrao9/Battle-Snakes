
<div align="center">
<img src="https://media.battlesnake.com/social/StarterSnakeGitHubRepos_Python.png" alt="Battlesnake Logo" width="100%"/>

# 🐍 Advanced Battlesnake AI

A collection of high-performance intelligent agents built for the Battlesnake arena, featuring Monte Carlo Tree Search (MCTS), Reinforcement Learning (RL), and Advanced Heuristics.

</div>

---

## 🚀 Quick Start: Running on Localhost

Want to watch the AI battle it out locally? We've made it incredibly simple using the included runner scripts. These scripts will automatically start your Python server and use the Battlesnake CLI to initiate a local game.

### 1. Install Dependencies

First, ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
````

---

### 2. Start the Game!

Depending on your operating system, use one of the following scripts:

#### 🪟 Windows

```bash
.\run.bat
```

#### 🍎 Mac / 🐧 Linux

```bash
chmod +x run.sh
./run.sh
```

> These scripts use the pre-compiled Battlesnake CLI binary included in the root folder to start a game against your local `server.py`.

---

## 🧠 Our AI Agents

This repository includes multiple AI implementations inside the `main/` directory:

* **`mcts_agent-final.py`**
  Highly tuned Monte Carlo Tree Search agent simulating thousands of future states.

* **`mcts_agent-variation.py`**
  Experimental MCTS version for A/B testing.

* **`rl_agent.py`**
  Reinforcement Learning-based agent trained for survival and trapping.

* **`heuristic_agent.py`**
  Fast rule-based agent using weighted evaluations.

* **`vanilla_mcts.py`**
  Baseline MCTS implementation for benchmarking.

Additional tools:

* **`tournammentrunner.py`** → Automated tournaments
* **`mcts_tuner.py`** → Hyperparameter tuning using Optuna

---

## 📂 Project Structure

```
📦 Battlesnake-AI
 ┣ 📂 main
 ┃ ┣ 📜 graphs.py
 ┃ ┣ 📜 heuristic_agent.py
 ┃ ┣ 📜 logger.py
 ┃ ┣ 📜 mcts_agent-final.py
 ┃ ┣ 📜 mcts_agent-variation.py
 ┃ ┣ 📜 mcts_tuner.py
 ┃ ┣ 📜 rl_agent.py
 ┃ ┣ 📜 run_game.py
 ┃ ┣ 📜 server.py
 ┃ ┣ 📜 tournammentrunner.py
 ┃ ┗ 📜 vanilla_mcts.py
 ┣ 📂 optuna-mcts-final
 ┣ 📂 RL_Agent
 ┣ 📂 tourny
 ┣ 📜 Assignment_Description_MGAI_2.pdf
 ┣ 📜 run.sh
 ┣ 📜 run.bat
 ┣ 📜 run.txt
 ┣ 📜 battlesnake
 ┣ 📜 requirements.txt
 ┣ 📜 .gitignore
 ┗ 📜 README.md
```

---

## 🛠️ Battlesnake CLI Usage

You can manually run games using the included CLI:

```bash
./battlesnake play -W 11 -H 11 --name "My_AI_Snake" --url http://localhost:8000 -g solo -v
```

---

### 🔄 Updating CLI

Download latest binaries or install via Go:

```bash
go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
```

---

## ❓ FAQ & Feedback

### Can I run games locally?

Yes — use `run.sh` or `run.bat` for automated local matches.

### Feedback or suggestions?

* Open an issue in this repository
* Or use the official Battlesnake feedback channels

---

<div align="center">
<i>Happy coding and may your snake never starve! 🐍🍎</i>
</div>
