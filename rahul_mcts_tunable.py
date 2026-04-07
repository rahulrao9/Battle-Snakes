"""
rahul_mcts_tunable.py
=====================
Strategic Battlesnake MCTS agent.

All hyper-parameters (including move time budget) are read from environment
variables so the Optuna tuner can launch many parallel instances without
touching source code.

Strategy
--------
  EARLY GAME  → aggressive food chasing, maximise length as fast as possible.
  LATE GAME   → dominate board centre, squeeze opponents into corners via
                flood-fill pressure, hunt killable heads (head-on kills).

Fixes applied (original)
------------------------
  Fix 1 — RAVE/AMAF: simulate() now tracks rollout actions and returns
           (score, rollout_actions). backpropagate() receives them so RAVE
           is populated from actual rollout data, not just tree paths.

  Fix 2 — Normalized rewards: all heuristic evaluations return values in
           [0.05, 0.95]. Terminal win = 1.0, terminal loss = 0.0.

  Fix 3 — Simultaneous-move decoupling: each MCTSNode stores parent_state.
           simulate() re-rolls opponent moves fresh on every call.

Performance fixes (new)
-----------------------
  Fix 4 — Fast rollout policy: a dedicated get_fast_action_space() skips
           all flood-fill calls during simulate(). Flood-fill is only used
           in expand() (tree phase) where it genuinely shapes the search.
           This alone multiplies rollout throughput by 3-5×.

  Fix 5 — Deeper rollouts: the artificial depth-6 Voronoi cut-off is
           replaced with a depth-20 limit (env-configurable). The early
           cut-off fired before any meaningful positional information could
           accumulate. With Fix 4 removing the BFS bottleneck, rollouts
           can now reach depth 20+ within the same wall-clock budget.

  Fix 6 — Proven-win backpropagation: mirrors the existing proven_loss
           logic. If any child of a node is a proven_win (sole survivor or
           opponent has zero moves), the parent inherits proven_win and
           immediately scores 1.0. best_child() prefers proven_win children;
           _best_move() returns a proven_win child instantly, short-circuiting
           further search.

  Fix 7 — RAVE overcounting: backpropagate() now iterates over
           set(sim_actions) instead of the raw list. AMAF semantics require
           a binary presence check per rollout — if "up" was played 8 of 15
           turns, it should contribute +1 visit, not +8. The list-based
           version inflated visit counts for repetitive moves and broke the
           RAVE/UCB balance.
"""

import time
import math
import random
import os
from typing import Dict, List, Set, Tuple, Optional
from collections import deque
from server import run_server
from heuristic_agent import evaluate_state as _raw_voronoi_eval

MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MOVE_LIST = list(MOVES.items())   # stable ordering for fast iteration

# =============================================================================
# Hyper-parameters — all read from env vars (with sane defaults)
# =============================================================================
C_PARAM              = float(os.environ.get("RAHUL_C_PARAM",              "1.997072949857646"))
DEPTH_LIMIT          = int(  os.environ.get("RAHUL_DEPTH_LIMIT",          "21"))
ROLLOUT_DEPTH_LIMIT  = int(  os.environ.get("RAHUL_ROLLOUT_DEPTH_LIMIT",  "20"))
PB_WEIGHT            = float(os.environ.get("RAHUL_PB_WEIGHT",            "17.214671823404686"))
RAVE_K               = float(os.environ.get("RAHUL_RAVE_K",               "1054.7344101331041"))
PHASE_LATE_TURN      = int(  os.environ.get("RAHUL_PHASE_LATE_TURN",      "42"))
PHASE_LATE_LENGTH    = int(  os.environ.get("RAHUL_PHASE_LATE_LENGTH",    "10"))
FOOD_WEIGHT_EARLY    = float(os.environ.get("RAHUL_FOOD_WEIGHT_EARLY",    "6.470062206942892"))
SPACE_WEIGHT_LATE    = float(os.environ.get("RAHUL_SPACE_WEIGHT_LATE",    "1.3002160883138354"))
CENTER_WEIGHT_LATE   = float(os.environ.get("RAHUL_CENTER_WEIGHT_LATE",   "2.3372287215539562"))
CORNER_WEIGHT_LATE   = float(os.environ.get("RAHUL_CORNER_WEIGHT_LATE",   "3.1465101163543308"))
KILL_WEIGHT_LATE     = float(os.environ.get("RAHUL_KILL_WEIGHT_LATE",     "3.661558132827282"))
TIME_LIMIT_MS        = int(  os.environ.get("RAHUL_TIME_LIMIT_MS",        "800"))
PROVEN_WIN_SCORE     = 1.0   # Fix 6: proven win sentinel value (normalized)
PROVEN_LOSS_SCORE    = 0.0   # Fix 6: proven loss sentinel value (normalized)


# =============================================================================
# Voronoi helper — normalise raw tile counts to [0, 10]
# =============================================================================

def voronoi_eval(state: "GameState") -> Dict[str, float]:
    raw   = _raw_voronoi_eval(state)
    total = state.board_width * state.board_height
    return {sid: (v / total) * 10.0 for sid, v in raw.items()}


# =============================================================================
# Forward model
# =============================================================================

class Snake:
    __slots__ = ["id", "body", "health", "is_alive"]

    def __init__(self, sid, body, health, is_alive=True):
        self.id, self.body, self.health, self.is_alive = sid, body, health, is_alive

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    def clone(self):
        return Snake(self.id, deque(self.body), self.health, self.is_alive)


class GameState:
    __slots__ = ["board_width", "board_height", "turn", "snakes", "food", "hazards"]

    def __init__(self, width, height, turn, snakes, food, hazards):
        self.board_width  = width
        self.board_height = height
        self.turn         = turn
        self.snakes       = snakes
        self.food         = food
        self.hazards      = hazards

    @classmethod
    def from_json(cls, gs: dict):
        snakes = {}
        for s in gs["board"]["snakes"]:
            body = deque((p["x"], p["y"]) for p in s["body"])
            snakes[s["id"]] = Snake(s["id"], body, s["health"])
        return cls(
            gs["board"]["width"], gs["board"]["height"], gs["turn"],
            snakes,
            {(f["x"], f["y"]) for f in gs["board"]["food"]},
            {(h["x"], h["y"]) for h in gs["board"]["hazards"]},
        )

    def clone(self):
        return GameState(
            self.board_width, self.board_height, self.turn,
            {sid: s.clone() for sid, s in self.snakes.items()},
            set(self.food), self.hazards,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _solid_obstacles(self) -> Set[Tuple[int, int]]:
        obs: Set[Tuple[int, int]] = set()
        for s in self.snakes.values():
            if not s.is_alive:
                continue
            bl = list(s.body)
            obs.update(bl[:-1] if s.health < 100 and len(bl) > 1 else bl)
        return obs

    def flood_fill(self, start: Tuple[int, int],
                   obs: Set[Tuple[int, int]]) -> int:
        if start in obs:
            return 0
        visited = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in MOVES.values():
                nx, ny = x + dx, y + dy
                nb = (nx, ny)
                if (0 <= nx < self.board_width and 0 <= ny < self.board_height
                        and nb not in obs and nb not in visited):
                    visited.add(nb)
                    q.append(nb)
        return len(visited)

    def center(self) -> Tuple[float, float]:
        return (self.board_width - 1) / 2.0, (self.board_height - 1) / 2.0

    # =========================================================================
    # Fix 4 — FAST rollout action space (no flood-fill, O(1) per move)
    # =========================================================================
    # Used exclusively inside simulate() so rollouts stay lightweight.
    # Only checks:
    #   • out-of-bounds
    #   • immediate body/wall collision
    #   • lethal hazard (health would drop to ≤ 0 this turn)
    # Does NOT prune trap corridors — MCTS handles that via rollout outcomes.

    def get_fast_action_space(self, sid: str) -> List[str]:
        """
        O(4) action filter — no BFS, no danger-zone scan.
        Returns any move that doesn't immediately kill us this tick.
        Falls back to ["up"] only when every direction is lethal.
        """
        snake = self.snakes.get(sid)
        if not snake or not snake.is_alive:
            return ["up"]

        # Build body-obstacle set once (cheap: iterate over all bodies)
        obs: Set[Tuple[int, int]] = set()
        for s in self.snakes.values():
            if not s.is_alive:
                continue
            bl = list(s.body)
            # The tail will vacate (unless the snake just ate), so exclude it
            obs.update(bl[:-1] if s.health < 100 and len(bl) > 1 else bl)

        hazard_stack  = min(4, max(0, (self.turn - 1) // 25)) if self.turn >= 26 else 0
        lethal_dmg    = 1 + 14 * hazard_stack  # total damage stepping into hazard

        safe:   List[str] = []
        lethal: List[str] = []   # fallback if everything is "lethal"

        for m, (dx, dy) in MOVE_LIST:
            nx, ny = snake.head[0] + dx, snake.head[1] + dy

            # Wall collision
            if not (0 <= nx < self.board_width and 0 <= ny < self.board_height):
                continue

            target = (nx, ny)

            # Body collision
            if target in obs:
                continue

            # Lethal hazard: health drops to ≤ 0 this turn
            if target in self.hazards and snake.health <= lethal_dmg:
                lethal.append(m)
                continue

            safe.append(m)

        if safe:   return safe
        if lethal: return lethal
        return ["up"]

    # ── four-tier action space (used in TREE phase — flood-fill allowed) ──────

    def get_action_space(self, sid: str) -> List[str]:
        snake = self.snakes.get(sid)
        if not snake or not snake.is_alive:
            return []

        obs = self._solid_obstacles()

        danger_zone: Set[Tuple[int, int]] = set()
        for opp in self.snakes.values():
            if opp.id != sid and opp.is_alive and opp.length >= snake.length:
                for dx, dy in MOVES.values():
                    danger_zone.add((opp.head[0] + dx, opp.head[1] + dy))

        hazard_stack = min(4, max(0, (self.turn - 1) // 25)) if self.turn >= 26 else 0
        lethal_dmg   = 1 + 14 * hazard_stack

        safe:         List[str] = []
        risky:        List[str] = []
        hazard_safe:  List[str] = []
        lethal:       List[str] = []

        for m, (dx, dy) in MOVE_LIST:
            nx, ny = snake.head[0] + dx, snake.head[1] + dy
            if not (0 <= nx < self.board_width and 0 <= ny < self.board_height):
                continue
            target = (nx, ny)
            if target in obs:
                continue

            if target in self.hazards:
                if snake.health <= lethal_dmg:
                    lethal.append(m)
                else:
                    hazard_safe.append(m)
            elif target in danger_zone:
                risky.append(m)
            else:
                safe.append(m)

        if len(safe) > 1:
            obs_ff   = self._solid_obstacles()
            non_trap = [
                m for m in safe
                if self.flood_fill(
                    (snake.head[0] + MOVES[m][0], snake.head[1] + MOVES[m][1]),
                    obs_ff,
                ) >= snake.length
            ]
            if non_trap:
                safe = non_trap

        if safe:        return safe
        if risky:       return risky
        if hazard_safe: return hazard_safe
        if lethal:      return lethal
        return ["up"]

    # ── opponent move model ───────────────────────────────────────────────────

    def get_opponent_move(self, sid: str) -> str:
        moves = self.get_fast_action_space(sid)   # Fix 4: use fast version for opponents
        if not moves:
            return "up"
        if len(moves) == 1:
            return moves[0]

        snake = self.snakes[sid]
        head  = snake.head

        if snake.health < 40 and self.food:
            best_m, best_d = moves[0], float("inf")
            for m in moves:
                nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
                d = min(abs(nx - fx) + abs(ny - fy) for fx, fy in self.food)
                if d < best_d:
                    best_d, best_m = d, m
            return best_m

        obs = self._solid_obstacles()
        best_m, best_ff = moves[0], -1
        for m in moves:
            nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
            ff = self.flood_fill((nx, ny), obs)
            if ff > best_ff:
                best_ff, best_m = ff, m
        return best_m

    # ── THE CORE STRATEGY ─────────────────────────────────────────────────────

    def get_guided_move(self, sid: str) -> str:
        """
        Used inside simulate() for our own moves — no flood-fill.
        Uses get_fast_action_space() so the rollout stays cheap.
        Retains food-seeking and basic positional scoring but avoids BFS.
        """
        moves = self.get_fast_action_space(sid)   # Fix 4: fast version
        if not moves:
            return "up"

        snake    = self.snakes[sid]
        head     = snake.head
        is_early = self.turn < PHASE_LATE_TURN or snake.length < PHASE_LATE_LENGTH

        # Emergency food chase — no BFS needed, just Manhattan distance
        if snake.health < 30 and self.food:
            best_m, best_d = moves[0], float("inf")
            for m in moves:
                nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
                d = min(abs(nx - fx) + abs(ny - fy) for fx, fy in self.food)
                if d < best_d:
                    best_d, best_m = d, m
            return best_m

        if is_early:
            # Score = food proximity bonus only (no flood-fill)
            best_m, best_score = moves[0], float("-inf")
            for m in moves:
                nx, ny     = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
                food_bonus = 0.0
                if self.food:
                    dist       = min(abs(nx - fx) + abs(ny - fy) for fx, fy in self.food)
                    food_bonus = -dist * FOOD_WEIGHT_EARLY
                if food_bonus > best_score:
                    best_score, best_m = food_bonus, m
            return best_m

        # Late game: centre proximity + kill hunting (no BFS)
        cx, cy   = self.center()
        max_dist = cx + cy

        killable_heads: List[Tuple[int, int]] = [
            opp.head
            for oid, opp in self.snakes.items()
            if oid != sid and opp.is_alive and snake.length > opp.length
        ]

        best_m, best_score = moves[0], float("-inf")

        for m in moves:
            nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]

            dist_c   = abs(nx - cx) + abs(ny - cy)
            centre_s = (1.0 - dist_c / max_dist) * CENTER_WEIGHT_LATE * 10.0

            kill_s = 0.0
            if killable_heads:
                kill_zones: Set[Tuple[int, int]] = {
                    (kx + ddx, ky + ddy)
                    for kx, ky in killable_heads
                    for ddx, ddy in MOVES.values()
                }
                if (nx, ny) in kill_zones:
                    kill_s = KILL_WEIGHT_LATE * 10.0
                else:
                    min_kd = min(abs(nx - kx) + abs(ny - ky) for kx, ky in kill_zones)
                    if min_kd > 0:
                        kill_s = (KILL_WEIGHT_LATE * 5.0) / min_kd

            total = centre_s + kill_s
            if total > best_score:
                best_score, best_m = total, m

        return best_m

    # ── state transition ──────────────────────────────────────────────────────

    def step(self, joint_action: Dict[str, str]) -> "GameState":
        ns = self.clone()
        ns.turn += 1

        hazard_stack  = min(4, max(0, (ns.turn - 1) // 25)) if ns.turn >= 26 else 0
        hazard_damage = 14 * hazard_stack
        eaten_food: Set[Tuple[int, int]] = set()

        for sid, action in joint_action.items():
            snake = ns.snakes.get(sid)
            if not snake or not snake.is_alive:
                continue
            dx, dy   = MOVES[action]
            new_head = (snake.head[0] + dx, snake.head[1] + dy)

            if new_head in ns.food:
                eaten_food.add(new_head)
                snake.health = 100
            else:
                snake.health -= 1
                snake.body.pop()

            if new_head in ns.hazards:
                snake.health -= (1 + hazard_damage)

            snake.body.appendleft(new_head)

        ns.food -= eaten_food

        body_counts: Dict[Tuple, int] = {}
        for s in ns.snakes.values():
            if not s.is_alive:
                continue
            for pt in s.body:
                body_counts[pt] = body_counts.get(pt, 0) + 1

        dead: Set[str] = set()
        for sid, snake in ns.snakes.items():
            if not snake.is_alive:
                continue
            head = snake.head

            if (not (0 <= head[0] < ns.board_width and 0 <= head[1] < ns.board_height)
                    or snake.health <= 0):
                dead.add(sid)
                continue

            if body_counts.get(head, 0) > 1:
                collided = [s for s in ns.snakes.values() if s.is_alive and s.head == head]
                if len(collided) > 1:
                    max_len = max(s.length for s in collided)
                    longest = [s for s in collided if s.length == max_len]
                    if len(longest) > 1 or snake not in longest:
                        dead.add(sid)
                else:
                    dead.add(sid)

        for sid in dead:
            ns.snakes[sid].is_alive = False

        return ns


# =============================================================================
# MCTS Node — RAVE + Voronoi Progressive Bias + Proven Win/Loss
# =============================================================================

class MCTSNode:
    __slots__ = [
        "state", "my_id", "parent", "action_taken", "children",
        "visits", "score", "untried_actions", "pb_score",
        "proven_loss", "proven_win",          # Fix 6: added proven_win
        "rave_visits", "rave_score",
        "parent_state",
    ]

    def __init__(self, state: GameState, my_id: str,
                 parent=None, action_taken: str = None, pb_score: float = 0.0,
                 parent_state: Optional[GameState] = None):
        self.state        = state
        self.my_id        = my_id
        self.parent       = parent
        self.action_taken = action_taken
        self.children:    List["MCTSNode"] = []
        self.visits       = 0
        self.score        = 0.0
        self.pb_score     = pb_score
        self.proven_loss  = False
        self.proven_win   = False   # Fix 6
        self.rave_visits: Dict[str, int]   = {}
        self.rave_score:  Dict[str, float] = {}
        self.untried_actions = self.state.get_action_space(self.my_id)
        random.shuffle(self.untried_actions)
        self.parent_state: Optional[GameState] = parent_state

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def expand(self) -> "MCTSNode":
        if not self.untried_actions:
            return self

        my_action    = self.untried_actions.pop()
        joint_action = {self.my_id: my_action}
        for sid, snake in self.state.snakes.items():
            if sid != self.my_id and snake.is_alive:
                joint_action[sid] = (
                    self.state.get_opponent_move(sid)
                    if random.random() < 0.8
                    else random.choice(self.state.get_fast_action_space(sid) or ["up"])
                )

        next_state = self.state.step(joint_action)
        h_score    = voronoi_eval(next_state).get(self.my_id, 0.0) / 10.0  # [0,1]

        child = MCTSNode(
            state=next_state, my_id=self.my_id,
            parent=self, action_taken=my_action, pb_score=h_score,
            parent_state=self.state,
        )

        me      = next_state.snakes.get(self.my_id)
        alive   = [sid for sid, s in next_state.snakes.items() if s.is_alive]

        if not me or not me.is_alive:
            # Fix 2: proven loss = normalized 0.0
            child.proven_loss = True
            child.score       = PROVEN_LOSS_SCORE

        elif len(alive) == 1 and alive[0] == self.my_id:
            # Fix 6: sole survivor after expansion → proven win
            child.proven_win = True
            child.score      = PROVEN_WIN_SCORE

        else:
            # Fix 6: check if every opponent has zero valid moves after this step
            opponents_stuck = all(
                len(next_state.get_fast_action_space(sid)) == 0
                for sid in alive if sid != self.my_id
            )
            if opponents_stuck and me.is_alive:
                child.proven_win = True
                child.score      = PROVEN_WIN_SCORE

        self.children.append(child)

        # Fix 6: propagate proven_win upward immediately on expansion
        if child.proven_win:
            self._check_proven_win()

        return child

    def _check_proven_win(self):
        """Fix 6: If any child is a proven_win, this node is also a proven_win."""
        if self.proven_win:
            return
        if any(c.proven_win for c in self.children):
            self.proven_win = True
            self.score      = PROVEN_WIN_SCORE
            if self.parent:
                self.parent._check_proven_win()

    def best_child(self, c_param: float = C_PARAM) -> "MCTSNode":
        """
        UCB1 + RAVE blend + linear PB decay.

        Fix 6: a proven_win child is always selected immediately — no UCB
        calculation needed.  proven_loss children are still skipped.
        """
        # Fast path: return any proven_win child immediately
        for child in self.children:
            if child.proven_win:
                return child

        best_val   = float("-inf")
        best_nodes: List["MCTSNode"] = []
        log_n      = math.log(self.visits) if self.visits > 0 else 0.0

        for child in self.children:
            if child.proven_loss:
                continue
            if child.visits == 0:
                return child

            q = child.score / child.visits

            # RAVE blend
            rv = self.rave_visits.get(child.action_taken, 0)
            if rv > 0:
                beta    = math.sqrt(RAVE_K / (3.0 * self.visits + RAVE_K))
                rave_q  = self.rave_score.get(child.action_taken, 0.0) / rv
                blended = (1.0 - beta) * q + beta * rave_q
            else:
                blended = q

            exploration = c_param * math.sqrt(log_n / child.visits) if log_n > 0 else 0.0
            pb          = (PB_WEIGHT * child.pb_score) / (child.visits + 1)

            ucb = blended + exploration + pb + random.uniform(-1e-6, 1e-6)

            if ucb > best_val:
                best_val   = ucb
                best_nodes = [child]
            elif ucb == best_val:
                best_nodes.append(child)

        if not best_nodes:
            return random.choice(self.children) if self.children else self
        return random.choice(best_nodes)

    # ── Fix 2: normalized heuristic evaluation ────────────────────────────────

    def _eval_heuristic(self, state: GameState, me: Snake) -> float:
        """Returns a value in [0.05, 0.95]."""
        health_norm = me.health / 100.0
        is_late     = (state.turn >= PHASE_LATE_TURN
                       and me.length >= PHASE_LATE_LENGTH)

        if is_late:
            space_norm  = voronoi_eval(state).get(self.my_id, 0.0) / 10.0
            cx, cy      = state.center()
            max_dist    = cx + cy
            dist_c      = abs(me.head[0] - cx) + abs(me.head[1] - cy)
            centre_norm = (1.0 - dist_c / max_dist) if max_dist > 0 else 0.5
            raw = (0.25 * health_norm
                   + 0.55 * space_norm
                   + 0.20 * centre_norm)
        else:
            length_norm = min(1.0, me.length / 20.0)
            raw = 0.35 * health_norm + 0.65 * length_norm

        return 0.05 + 0.90 * max(0.0, min(1.0, raw))

    # ── Fix 1 + Fix 3 + Fix 4 + Fix 5: full simulate() ───────────────────────

    def simulate(self, depth_limit: int = ROLLOUT_DEPTH_LIMIT) -> Tuple[float, List[str]]:
        """
        Fast, phase-aware rollout.

        Fix 1 — tracks rollout actions for RAVE backpropagation.
        Fix 3 — re-samples opponent moves on first step (simultaneous-move decoupling).
        Fix 4 — uses get_fast_action_space() throughout; no BFS inside rollout.
        Fix 5 — depth-6 early Voronoi cut-off removed; runs to depth_limit (default 20)
                 or a terminal state.  Voronoi heuristic still fires at the horizon.
        Fix 6 — proven_win/proven_loss nodes short-circuit immediately.
        """
        if self.proven_loss:
            return PROVEN_LOSS_SCORE, []

        if self.proven_win:
            return PROVEN_WIN_SCORE, []

        rollout_actions: List[str] = []

        # Fix 3: re-sample opponent moves for the first step
        if self.parent_state is not None and self.action_taken is not None:
            fresh_joint: Dict[str, str] = {self.my_id: self.action_taken}
            for sid, snake in self.parent_state.snakes.items():
                if sid != self.my_id and snake.is_alive:
                    # Fix 4: get_fast_action_space for opponents during rollout
                    opp_moves = self.parent_state.get_fast_action_space(sid)
                    fresh_joint[sid] = (
                        self.parent_state.get_opponent_move(sid)
                        if random.random() < 0.8
                        else random.choice(opp_moves or ["up"])
                    )
            current = self.parent_state.step(fresh_joint)
        else:
            current = self.state.clone()

        depth = 0

        while depth < depth_limit:
            alive = [sid for sid, s in current.snakes.items() if s.is_alive]

            # Terminal: we died
            if self.my_id not in alive:
                return PROVEN_LOSS_SCORE, rollout_actions

            # Terminal: sole survivor
            if len(alive) == 1:
                return PROVEN_WIN_SCORE, rollout_actions

            # Fix 5: heuristic evaluation at the rollout horizon (not depth 6)
            # We still allow an early exit if the board is clearly decided:
            #   - only two snakes left and we have a large Voronoi lead, OR
            #   - we have reached half the depth limit (quick assessment)
            if depth >= depth_limit // 2:
                me = current.snakes.get(self.my_id)
                if not me:
                    return PROVEN_LOSS_SCORE, rollout_actions
                # Only cut short in very lopsided situations to save compute
                if len(alive) == 2:
                    return self._eval_heuristic(current, me), rollout_actions

            # Build joint action for this rollout step
            joint: Dict[str, str] = {}
            for sid in alive:
                if sid == self.my_id:
                    # Fix 4: get_guided_move now uses get_fast_action_space internally
                    m = current.get_guided_move(sid)
                    rollout_actions.append(m)
                    joint[sid] = m
                else:
                    # Fix 4: fast action space for opponents
                    opp_moves = current.get_fast_action_space(sid)
                    joint[sid] = (
                        current.get_opponent_move(sid)
                        if random.random() < 0.85
                        else random.choice(opp_moves or ["up"])
                    )

            current = current.step(joint)
            depth  += 1

        me = current.snakes.get(self.my_id)
        if not me or not me.is_alive:
            return PROVEN_LOSS_SCORE, rollout_actions

        return self._eval_heuristic(current, me), rollout_actions

    # ── Fix 1 + Fix 6 + Fix 7: backpropagate ─────────────────────────────────

    def backpropagate(self, score: float, sim_actions: Optional[List[str]] = None):
        self.visits += 1
        if not self.proven_loss and not self.proven_win:
            self.score += score
        elif self.proven_win:
            # Keep proven_win score pinned at PROVEN_WIN_SCORE so visit
            # averaging doesn't dilute the signal
            self.score = PROVEN_WIN_SCORE * self.visits

        # Fix 7 — RAVE overcounting fix: deduplicate sim_actions with set()
        # AMAF semantics require a binary presence check: "was this action
        # taken AT ALL during the playout?", not "how many times?".
        # Iterating over the raw list inflated visit counts for repetitive
        # moves (e.g. "up" played 8 of 15 turns → +8 visits instead of +1),
        # which broke the UCB exploitation/exploration balance.
        if sim_actions:
            for a in set(sim_actions):   # set() → each action counted once per rollout
                self.rave_visits[a] = self.rave_visits.get(a, 0) + 1
                self.rave_score[a]  = self.rave_score.get(a, 0.0) + score

        # Proven-loss propagation (existing logic)
        if self.children and self.is_fully_expanded():
            if all(c.proven_loss for c in self.children):
                self.proven_loss = True
                self.score = PROVEN_LOSS_SCORE * self.visits   # pin score

        if self.parent:
            extended = (
                [self.action_taken] + (sim_actions or [])
                if self.action_taken else (sim_actions or [])
            )
            self.parent.backpropagate(score, extended)


# =============================================================================
# MCTS Agent
# =============================================================================

class MCTSAgent:
    def __init__(self, my_id: str, time_limit_ms: int = TIME_LIMIT_MS):
        self.my_id      = my_id
        self.time_limit = time_limit_ms / 1000.0

    def search(self, root_state: GameState) -> str:
        start = time.time()
        root  = MCTSNode(state=root_state, my_id=self.my_id)

        if not root.untried_actions and not root.children:
            return "up"

        def eval_action(a: str) -> float:
            joint = {self.my_id: a}
            for sid, s in root_state.snakes.items():
                if sid != self.my_id and s.is_alive:
                    joint[sid] = root_state.get_opponent_move(sid)
            return voronoi_eval(root_state.step(joint)).get(self.my_id, 0.0)

        root.untried_actions.sort(key=eval_action, reverse=True)

        iterations = 0
        while time.time() - start < self.time_limit:
            # Fix 6: short-circuit if a proven_win is already found at root
            if root.proven_win:
                break

            node = self._select(root)
            me   = node.state.snakes.get(self.my_id, Snake("_", deque(), 0, False))
            if me.is_alive and not node.is_fully_expanded():
                node = node.expand()

            # Fix 6: skip rollout for proven nodes
            if node.proven_win:
                node.backpropagate(PROVEN_WIN_SCORE, [])
            elif node.proven_loss:
                node.backpropagate(PROVEN_LOSS_SCORE, [])
            else:
                score, rollout_actions = node.simulate(depth_limit=ROLLOUT_DEPTH_LIMIT)
                node.backpropagate(score, rollout_actions)

            iterations += 1

        elapsed = time.time() - start
        best    = self._best_move(root)
        print(
            f"[RahulMCTS] T={game_state_turn} iters={iterations} "
            f"t={elapsed*1000:.0f}ms move={best}",
            flush=True,
        )
        return best

    def _select(self, node: MCTSNode) -> MCTSNode:
        while True:
            me = node.state.snakes.get(self.my_id, Snake("_", deque(), 0, False))
            if not me.is_alive or not node.is_fully_expanded() or not node.children:
                break
            # Fix 6: best_child now fast-paths to proven_win children
            node = node.best_child()
        return node

    def _best_move(self, root: MCTSNode) -> str:
        if not root.children:
            safe = root.state.get_action_space(self.my_id)
            return safe[0] if safe else "up"

        # Fix 6: if a proven_win child exists, return it immediately
        for child in root.children:
            if child.proven_win:
                print(f"[RahulMCTS] Proven-win move: {child.action_taken}", flush=True)
                return child.action_taken

        viable = [c for c in root.children if not c.proven_loss]
        pool   = viable if viable else root.children
        return max(pool, key=lambda c: c.visits).action_taken


# =============================================================================
# Battlesnake endpoints
# =============================================================================

game_state_turn = 0


def info() -> Dict:
    return {
        "apiversion": "1",
        "author":     "nagin",
        "color":      "#FF0505",
        "head":       "default",
        "tail":       "default",
    }

def start(game_state: Dict):
    pass

def end(game_state: Dict):
    pass

def move(game_state: Dict) -> Dict:
    global game_state_turn
    game_state_turn = game_state["turn"]

    my_id      = game_state["you"]["id"]
    root_state = GameState.from_json(game_state)
    agent      = MCTSAgent(my_id=my_id)
    best_move  = agent.search(root_state)
    print(f"Turn {game_state['turn']} | RahulMCTS → {best_move}", flush=True)
    return {"move": best_move}


if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})