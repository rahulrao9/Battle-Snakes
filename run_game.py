import json
import subprocess
import time
from pathlib import Path

MAX_TURNS = 300
LOG_PATH = Path("game.json")

CMD = [
    "battlesnake", "play",
    "-W", "11", "-H", "11",
    "-g", "standard",
    "-m", "hz_hazard_pits",
    "--name", "Snake1", "--url", "http://127.0.0.1:8000",
    "--name", "Snake2", "--url", "http://127.0.0.1:8001",
    "--name", "Snake3", "--url", "http://127.0.0.1:8002",
    "--name", "Snake4", "--url", "http://127.0.0.1:8003",
    "--foodSpawnChance", "25",
    "--minimumFood", "2",
    "--seed", "123",
    "--timeout", "1000",
    "--browser",
    "--output", str(LOG_PATH),
]

def load_last_state(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return None

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict) and "turn" in obj:
            return obj

    return None

def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    proc = subprocess.Popen(CMD)

    last_turn = -1
    last_state = None

    try:
        while proc.poll() is None:
            state = load_last_state(LOG_PATH)
            if state is not None:
                last_state = state
                turn = int(state.get("turn", -1))

                if turn == -1:
                    time.sleep(0.1)
                    continue

                if turn != last_turn:
                    last_turn = turn
                    print(f"turn={turn}")

                if turn >= MAX_TURNS:
                    print(f"Reached cap at turn {turn}. Stopping game.")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            time.sleep(0.1)

    finally:
        if proc.poll() is None:
            proc.kill()

    if last_state is None:
        print("Game end")
        return

    for snake in last_state["board"]["snakes"]:
          print(snake["name"], snake["length"])


if __name__ == "__main__":
    main()