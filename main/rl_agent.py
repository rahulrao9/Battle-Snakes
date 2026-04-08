"""
Battlesnake AlphaZero Agent — Main Entry Point (v2, frame-stacked)
===================================================================
Changes from v1:
  ✦ Network architecture read from checkpoint config (no hardcoded sizes)
  ✦ Per-game state history maintained so MCTS receives 3-frame context
  ✦ start() initialises the history deque; end() cleans it up
"""

import os
from collections import deque
from typing import Dict

import torch

from RL_Agent.neural_net    import BattlesnakeNet
from RL_Agent.az_mcts       import AlphaZeroMCTS
from RL_Agent.forward_model import GameState
from server                 import run_server

# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get("SNAKE_CHECKPOINT", "./RL_Agent/battlesnake_net-v2.pt")
TIME_LIMIT_MS   = int(os.environ.get("SNAKE_TIME_MS", "650"))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load network once at startup ──────────────────────────────────────────────

print(f"[Init] Loading network on {DEVICE}...")

_net = None
if os.path.exists(CHECKPOINT_PATH):
    try:
        payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

        # Read architecture from the checkpoint so this never needs manual updates
        saved_cfg      = payload.get("config", {})
        in_channels    = saved_cfg.get("in_channels",    30)   # 30 = 3 frames × 10
        num_filters    = saved_cfg.get("num_filters",    128)
        num_res_blocks = saved_cfg.get("num_res_blocks", 12)

        _net = BattlesnakeNet(in_channels, num_filters, num_res_blocks)
        _net.load_state_dict(payload["model_state"])
        iteration = payload.get("iteration", "?")
        print(f"[Init] Loaded '{CHECKPOINT_PATH}' "
              f"(iter {iteration}, {in_channels}ch, "
              f"{num_filters}f×{num_res_blocks} blocks)")
    except Exception as e:
        print(f"[Init] WARNING — checkpoint load failed: {e}")
        _net = None

if _net is None:
    print("[Init] Falling back to default architecture with random weights.")
    _net = BattlesnakeNet(in_channels=30, num_filters=128, num_res_blocks=12)

_net.to(DEVICE).eval()
print(f"[Init] Network ready ({sum(p.numel() for p in _net.parameters()):,} params)")

# ── Per-game state history ────────────────────────────────────────────────────
# Each live game gets its own deque keyed by game_id.
# The deque holds up to 3 board states, most-recent first.

_game_histories: Dict[str, deque] = {}


# ── Battlesnake API handlers ──────────────────────────────────────────────────

def info() -> Dict:
    return {
        "apiversion": "1",
        "author":     "MGAIA_AlphaZero",
        "color":      "#ff0000",
        "head":       "default",
        "tail":       "default",
    }


def start(game_state: Dict):
    """Called once at game start — initialise a fresh history deque."""
    game_id = game_state["game"]["id"]
    _game_histories[game_id] = deque(maxlen=3)


def end(game_state: Dict):
    """Called once at game end — clean up to avoid memory leaks."""
    game_id = game_state["game"]["id"]
    _game_histories.pop(game_id, None)


def move(game_state: Dict) -> Dict:
    """
    Main decision handler called every turn.

    Flow:
      1. Parse the board state from the JSON payload.
      2. Prepend it to this game's history deque (most-recent first).
      3. Pass the previous frames as root_history to MCTS so the network
         receives a full [30, 11, 11] tensor at every node evaluation.
      4. Return the chosen move.
    """
    game_id    = game_state["game"]["id"]
    my_id      = game_state["you"]["id"]
    root_state = GameState.from_json(game_state)

    # Ensure history exists even if start() was somehow missed
    if game_id not in _game_histories:
        _game_histories[game_id] = deque(maxlen=3)

    history = _game_histories[game_id]

    # Prepend current state — index 0 is always the newest frame
    history.appendleft(root_state)

    # root_history = frames *before* the current turn (S_{t-1}, S_{t-2}).
    # MCTS will set S_t as the root node's state internally, so we skip [0].
    root_hist = list(history)[1:]

    agent = AlphaZeroMCTS(
        my_id         = my_id,
        net           = _net,
        time_limit_ms = TIME_LIMIT_MS,
        device        = DEVICE,
        root_history  = root_hist,
    )
    best_move = agent.search(root_state, training=False)

    print(f"[Turn {game_state['turn']}] → {best_move}")
    return {"move": best_move}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})