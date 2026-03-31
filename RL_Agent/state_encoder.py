import numpy as np

NUM_CHANNELS = 10
HISTORY_LEN  = 3
IN_CHANNELS  = NUM_CHANNELS * HISTORY_LEN   # 30 — what the network sees
MOVE_ORDER   = ["up", "down", "left", "right"]
MOVE_TO_IDX  = {m: i for i, m in enumerate(MOVE_ORDER)}


def _encode_single_state(state, my_id: str) -> np.ndarray:
    """Encode one board state into a [10, H, W] tensor."""
    W, H = state.board_width, state.board_height
    board = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)
    me = state.snakes.get(my_id)
    if not me or not me.is_alive:
        return board
    body_list = list(me.body)
    hx, hy = me.head
    board[0, hy, hx] = 1.0
    for i, (bx, by) in enumerate(body_list[1:], 1):
        board[1, by, bx] = max(board[1, by, bx], 1.0 - i / max(len(body_list), 1))
    tx, ty = body_list[-1]
    board[2, ty, tx] = 1.0
    enemies = [s for sid, s in state.snakes.items() if sid != my_id and s.is_alive]
    for e in enemies:
        ex, ey = e.head
        board[3, ey, ex] = 1.0
        for i, (bx, by) in enumerate(list(e.body)[1:], 1):
            board[4, by, bx] = max(board[4, by, bx], 1.0 - i / max(len(e.body), 1))
        if e.length >= me.length:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < W and 0 <= ny < H:
                    board[5, ny, nx] = 1.0
    for fx, fy in state.food:
        board[6, fy, fx] = 1.0
    for hzx, hzy in state.hazards:
        board[7, hzy, hzx] = 1.0
    board[8] = me.health / 100.0
    max_enemy_len = max((e.length for e in enemies), default=0)
    board[9] = np.clip((me.length - max_enemy_len) / 10.0, -1.0, 1.0)
    return board


def encode_state(state_history, my_id: str) -> np.ndarray:
    """
    Encode a history of up to HISTORY_LEN states into a [30, H, W] tensor.

    state_history: list/deque ordered most-recent-first, e.g. [S_t, S_{t-1}, S_{t-2}].
                   May also be a single bare state object (backwards-compatible).
    Returns: float32 array of shape (IN_CHANNELS, H, W) = (30, 11, 11).

    Older frames that are unavailable (early in the game) are left as zeros,
    which the network learns to treat as "no information".
    """
    # Backwards-compatibility: if called with a bare state object, wrap it.
    if not isinstance(state_history, (list, tuple)) and not hasattr(state_history, '__iter__'):
        state_history = [state_history]
    # deque / list both work; materialise to a plain list
    history = list(state_history)

    # Find board dimensions from the first non-None entry
    valid = next((s for s in history if s is not None), None)
    H = valid.board_height if valid else 11
    W = valid.board_width  if valid else 11

    out = np.zeros((IN_CHANNELS, H, W), dtype=np.float32)
    for i in range(HISTORY_LEN):
        if i < len(history) and history[i] is not None:
            out[i * NUM_CHANNELS : (i + 1) * NUM_CHANNELS] = (
                _encode_single_state(history[i], my_id)
            )
        # else: leave as zeros (no-history padding)
    return out


def decode_policy_mask(state, my_id: str) -> np.ndarray:
    legal = state.get_action_space(my_id)
    mask  = np.zeros(4, dtype=np.float32)
    for m in legal:
        mask[MOVE_TO_IDX[m]] = 1.0
    return mask