"""
az_mcts.py — AlphaZero MCTS: sequentialised tree + frame-stacked evaluation.

Changes in this version:
  - _eval(node, sid) now calls get_history() to build a 3-frame state stack,
    giving the network velocity/direction information about every snake.
  - AlphaZeroMCTS accepts root_history so the very first MCTS call has access
    to the real game's previous frames, not just the current state.
  - AZNode.get_history() walks up the tree collecting distinct board states,
    skipping mid-round intermediate nodes (same state, different actor).
"""

import math, time, numpy as np, torch
from typing import Dict, List, Optional
from state_encoder import encode_state, decode_policy_mask, MOVE_ORDER, MOVE_TO_IDX

C_PUCT            = 2.5
DIRICHLET_ALPHA   = 0.25
DIRICHLET_EPSILON = 0.25
FPU_VALUE         = -0.2
TEMP_THRESHOLD    = 20


class AZNode:
    __slots__ = [
        "state", "my_id", "actor_id",
        "pending_actions",
        "parent", "action_taken",
        "children", "N", "W", "Q", "P",
        "legal_moves", "is_terminal", "proven_loss",
    ]

    def __init__(self, state, my_id, actor_id,
                 pending_actions=None,
                 parent=None, action_taken=None, prior=0.0):
        self.state           = state
        self.my_id           = my_id
        self.actor_id        = actor_id
        self.pending_actions = pending_actions or {}
        self.parent          = parent
        self.action_taken    = action_taken
        self.children: Dict[str, "AZNode"] = {}
        self.N  = 0; self.W = 0.0; self.Q = 0.0; self.P = prior

        self.legal_moves = state.get_action_space(actor_id)

        me    = state.snakes.get(my_id)
        alive = sum(1 for s in state.snakes.values() if s.is_alive)
        self.proven_loss = not me or not me.is_alive
        self.is_terminal = self.proven_loss or alive <= 1

    @property
    def is_expanded(self): return len(self.children) > 0

    def puct_score(self, pN):
        if self.N == 0:
            return FPU_VALUE + C_PUCT * self.P * math.sqrt(max(pN, 1))
        return self.Q + C_PUCT * self.P * math.sqrt(pN) / (1 + self.N)

    def select_child(self):
        return max(self.children.values(), key=lambda c: c.puct_score(self.N))

    def get_history(self, root_history: List) -> List:
        """
        Walk up the tree to collect the last 3 *distinct* board states
        (most-recent first).

        Mid-round nodes share the same board state as their parent (only
        actor_id and pending_actions differ), so we skip them to avoid
        duplicates — we only record a state when pending_actions is empty,
        meaning we are at a real post-step node.

        If fewer than 3 real states exist in the tree (early game), we pad
        from root_history (the actual game history before MCTS started).
        """
        hist = []
        curr = self
        while curr is not None and len(hist) < 3:
            # A node with empty pending_actions is a real new-state node.
            # The root always qualifies (its pending_actions starts empty).
            if not curr.pending_actions:
                if not hist or hist[-1] is not curr.state:
                    hist.append(curr.state)
            curr = curr.parent

        # Pad from the real game history that predates this MCTS call
        if len(hist) < 3 and root_history:
            needed = 3 - len(hist)
            hist.extend(root_history[:needed])

        # Fill remaining slots with None (zero-padded by encode_state)
        while len(hist) < 3:
            hist.append(None)

        return hist  # [S_t, S_{t-1}, S_{t-2}], most-recent first

    def _alive_turn_order(self):
        return sorted(
            [sid for sid, s in self.state.snakes.items() if s.is_alive],
            key=lambda sid: (0 if sid == self.my_id else 1, sid)
        )

    def expand(self, policy_probs, add_noise=False):
        if not self.legal_moves:
            return

        noise = (np.random.dirichlet([DIRICHLET_ALPHA] * len(self.legal_moves))
                 if add_noise else None)

        alive_ids = self._alive_turn_order()

        for i, move in enumerate(self.legal_moves):
            prior = float(policy_probs[MOVE_TO_IDX[move]])
            if add_noise and noise is not None:
                prior = (1 - DIRICHLET_EPSILON) * prior + DIRICHLET_EPSILON * noise[i]

            new_pending = dict(self.pending_actions)
            new_pending[self.actor_id] = move

            next_actor = None
            for sid in alive_ids:
                if sid not in new_pending:
                    next_actor = sid
                    break

            if next_actor is not None:
                child = AZNode(
                    state           = self.state,
                    my_id           = self.my_id,
                    actor_id        = next_actor,
                    pending_actions = new_pending,
                    parent          = self,
                    action_taken    = move,
                    prior           = prior,
                )
            else:
                next_state  = self.state.step(new_pending)
                new_alive   = [sid for sid, s in next_state.snakes.items() if s.is_alive]
                first_actor = (self.my_id if self.my_id in new_alive
                               else (new_alive[0] if new_alive else self.my_id))
                child = AZNode(
                    state           = next_state,
                    my_id           = self.my_id,
                    actor_id        = first_actor,
                    pending_actions = {},
                    parent          = self,
                    action_taken    = move,
                    prior           = prior,
                )

            self.children[move] = child

    def backup(self, v):
        self.N += 1; self.W += v; self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(v)


class AlphaZeroMCTS:
    def __init__(self, my_id, net, time_limit_ms=600, device="cpu",
                 root_history=None):
        self.my_id        = my_id
        self.net          = net
        self.time_limit   = time_limit_ms / 1000.0
        self.device       = torch.device(device)
        self.root_history = root_history or []   # real game frames before MCTS
        self.net.to(self.device).eval()

    def _eval(self, node):
        """
        Evaluate node from node.actor_id's perspective using a 3-frame stack.
        The history is reconstructed by walking up the tree + real game history.
        """
        if node.proven_loss:
            return np.ones(4) * 0.25, -1.0

        sid  = node.actor_id
        me   = node.state.snakes.get(sid)
        if not me or not me.is_alive:
            return np.ones(4) * 0.25, -1.0

        hist = node.get_history(self.root_history)   # [S_t, S_{t-1}, S_{t-2}]
        t    = torch.from_numpy(encode_state(hist, sid)).to(self.device)
        m    = decode_policy_mask(node.state, sid)
        return self.net.predict(t, mask=torch.from_numpy(m).to(self.device))

    def search(self, root_state, training=False):
        best, _ = self._run(root_state, training)
        return best

    def search_with_policy(self, root_state, training=True):
        return self._run(root_state, training)

    def _run(self, root_state, training):
        start = time.time()

        root = AZNode(root_state, self.my_id, actor_id=self.my_id)
        if not root.legal_moves:
            return "up", np.array([0.25, 0.25, 0.25, 0.25])

        probs, _ = self._eval(root)
        root.expand(probs, add_noise=training)

        while time.time() - start < self.time_limit:
            node = root
            while node.is_expanded and not node.is_terminal:
                node = node.select_child()

            if node.is_terminal or node.proven_loss:
                v = -1.0 if node.proven_loss else 0.0
            else:
                probs, v = self._eval(node)
                if node.actor_id != self.my_id:
                    v = -v
                node.expand(probs, add_noise=False)

            node.backup(v)

        vp  = np.zeros(4, dtype=np.float32)
        tot = sum(c.N for c in root.children.values())
        for m, c in root.children.items():
            vp[MOVE_TO_IDX[m]] = c.N / max(tot, 1)

        if training:
            if root_state.turn < TEMP_THRESHOLD and vp.sum() > 0:
                best = np.random.choice(MOVE_ORDER, p=vp)
            else:
                best = MOVE_ORDER[int(np.argmax(vp))]
        else:
            pool = ({m: c for m, c in root.children.items() if not c.proven_loss}
                    or root.children)
            best = max(pool, key=lambda m: pool[m].N)

        return best, vp