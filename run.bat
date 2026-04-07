@echo off
echo Starting Snake servers...

:: Start Snake 1 on port 8000
echo Starting Heuristic Agent on Port 8000...
start "Snake 1 - Heuristic" cmd /k "set PORT=8000 && python heuristic_agent.py"

:: Start Snake 2 on port 8001
echo Starting MCTS Agent 3 on Port 8001...
start "Snake 2 - MCTS 3" cmd /k "set PORT=8001 && python mcts_agent3.py"

:: Start Snake 3 on port 8002
echo Starting RL Agent on Port 8002...
start "Snake 3 - nagin" cmd /k "set PORT=8002 && python rahul_mcts_tunable.py"

:: Start Snake 4 on port 8003
echo Starting Vanilla MCTS on Port 8003...
start "Snake 4 - Vanilla MCTS" cmd /k "set PORT=8003 && python vanilla_mcts.py"

:: Wait for 3 seconds to ensure all snake servers have started
echo Waiting 3 seconds for servers to boot up before starting the game...
timeout /t 6 /nobreak >nul

:: Start the main game
echo Starting run_game.py...
start "Battlesnake Game" cmd /k "python run_game.py"

echo All processes launched!
pause