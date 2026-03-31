#!/bin/bash

# Function to clean up background processes when the script exits
cleanup() {
    echo -e "\nShutting down Snake servers..."
    kill $SNAKE1 $SNAKE2 $SNAKE3 $SNAKE4 2>/dev/null
    echo "All processes terminated!"
    exit
}

trap cleanup SIGINT EXIT

echo "Starting Snake servers in the background..."

# Start Snake 1
PORT=8000 python3 heuristic_agent.py &
SNAKE1=$!
echo "Started Heuristic Agent on Port 8000 (PID: $SNAKE1)"

# Start Snake 2
PORT=8001 python3 mcts_agent3.py &
SNAKE2=$!
echo "Started MCTS Agent 3 on Port 8001 (PID: $SNAKE2)"

# Start Snake 3
PORT=8002 python3 rl_agent.py &
SNAKE3=$!
echo "Started RL Agent on Port 8002 (PID: $SNAKE3)"

# Start Snake 4
PORT=8003 python3 vanilla_mcts.py &
SNAKE4=$!
echo "Started Vanilla MCTS on Port 8003 (PID: $SNAKE4)"

# Wait for 3 seconds to ensure all snake servers have started
echo "Waiting 3 seconds for servers to boot up before starting the game..."
sleep 3

# Start the main game in the foreground
echo "Starting run_game.py..."
python3 run_game.py