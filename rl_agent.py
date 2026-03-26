"""
Battlesnake AlphaZero Agent — Main Entry Point
===============================================
Drop-in replacement for the original MCTS move.py.

Changes from original:
  ✦ Loads a trained BattlesnakeNet at startup (fast — only once)
  ✦ Uses AlphaZeroMCTS instead of random-rollout MCTSAgent
  ✦ Falls back gracefully to pure PUCT with random init if no checkpoint exists
  ✦ Forward model (GameState, Snake, MOVES) is unchanged

Run:
    python move.py                        # uses checkpoint if present
    python self_play_trainer.py --quick   # smoke test training
    python self_play_trainer.py           # full training run (then restart server)
"""

import os
import torch
from typing import Dict

from RL_Agent.neural_net    import BattlesnakeNet
from RL_Agent.az_mcts       import AlphaZeroMCTS
from RL_Agent.forward_model import GameState       # your existing forward model file
from server        import run_server      # your existing server file

# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get("SNAKE_CHECKPOINT", "battlesnake_net.pt")
BOARD_SIZE      = int(os.environ.get("SNAKE_BOARD_SIZE", "11"))
TIME_LIMIT_MS   = int(os.environ.get("SNAKE_TIME_MS", "650"))
NUM_FILTERS     = 64
NUM_RES_BLOCKS  = 8
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load network once at startup ──────────────────────────────────────────────

print(f"[Init] Loading network on {DEVICE}...")
_net = BattlesnakeNet(
    in_channels   = 10,
    num_filters   = NUM_FILTERS,
    num_res_blocks = NUM_RES_BLOCKS,
)

if os.path.exists(CHECKPOINT_PATH):
    try:
        payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        _net.load_state_dict(payload["model_state"])
        iteration = payload.get("iteration", "?")
        print(f"[Init] Loaded checkpoint '{CHECKPOINT_PATH}' (iteration {iteration})")
    except Exception as e:
        print(f"[Init] WARNING — Failed to load checkpoint: {e}")
        print("[Init] Using random-weight network (performance will be poor).")
else:
    print(f"[Init] No checkpoint found at '{CHECKPOINT_PATH}'.")
    print("[Init] Run 'python self_play_trainer.py' to train the network.")
    print("[Init] Using random-weight network for now.")

_net.to(DEVICE)
_net.eval()
print(f"[Init] Network ready ({sum(p.numel() for p in _net.parameters()):,} params)")


# ── Battlesnake API handlers ──────────────────────────────────────────────────

def info() -> Dict:
    return {
        "apiversion": "1",
        "author":     "MGAIA_AlphaZero",
        "color":      "#24e374",
        "head":       "default",
        "tail":       "default",
    }


def start(game_state: Dict):
    """Called at the start of each game. No state to initialise here."""
    pass


def end(game_state: Dict):
    """Called at the end of each game."""
    pass


def move(game_state: Dict) -> Dict:
    """
    Main decision handler. Called every turn.
    Budget: ~500ms of the 650ms limit goes to MCTS; 150ms kept as safety margin.
    """
    my_id      = game_state["you"]["id"]
    root_state = GameState.from_json(game_state)

    agent = AlphaZeroMCTS(
        my_id         = my_id,
        net           = _net,
        time_limit_ms = TIME_LIMIT_MS,
        device        = DEVICE,
    )
    best_move = agent.search(root_state, training=False)

    print(f"[Turn {game_state['turn']}] → {best_move}")
    return {"move": best_move}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})
