import math, random, time, numpy as np, torch
from typing import Dict, Optional, Tuple
from state_encoder import encode_state, decode_policy_mask, MOVE_ORDER, MOVE_TO_IDX

C_PUCT            = 2.5
DIRICHLET_ALPHA   = 0.25
DIRICHLET_EPSILON = 0.25
FPU_VALUE         = -0.2


class AZNode:
    __slots__ = ["state","my_id","parent","action_taken","children",
                 "N","W","Q","P","legal_moves","is_terminal","proven_loss"]
    def __init__(self, state, my_id, parent=None, action_taken=None, prior=0.0):
        self.state = state; self.my_id = my_id
        self.parent = parent; self.action_taken = action_taken
        self.children: Dict[str, "AZNode"] = {}
        self.N = 0; self.W = 0.0; self.Q = 0.0; self.P = prior
        self.legal_moves = state.get_action_space(my_id)
        me = state.snakes.get(my_id)
        alive = sum(1 for s in state.snakes.values() if s.is_alive)
        self.proven_loss = not me or not me.is_alive
        self.is_terminal  = self.proven_loss or alive <= 1
    @property
    def is_expanded(self): return len(self.children) > 0
    def puct_score(self, pN):
        if self.N == 0:
            return FPU_VALUE + C_PUCT * self.P * math.sqrt(max(pN, 1))
        return self.Q + C_PUCT * self.P * math.sqrt(pN) / (1 + self.N)
    def select_child(self):
        return max(self.children.values(), key=lambda c: c.puct_score(self.N))
    def expand(self, policy_probs, add_noise=False):
        if not self.legal_moves: return
        noise = np.random.dirichlet([DIRICHLET_ALPHA]*len(self.legal_moves)) if add_noise else None
        for i, move in enumerate(self.legal_moves):
            prior = float(policy_probs[MOVE_TO_IDX[move]])
            if add_noise:
                prior = (1-DIRICHLET_EPSILON)*prior + DIRICHLET_EPSILON*noise[i]
            joint = {self.my_id: move}
            for sid, snake in self.state.snakes.items():
                if sid != self.my_id and snake.is_alive:
                    joint[sid] = self.state.get_guided_move(sid)
            ns = self.state.step(joint)
            self.children[move] = AZNode(ns, self.my_id, self, move, prior)
    def backup(self, v):
        self.N += 1; self.W += v; self.Q = self.W / self.N
        if self.parent: self.parent.backup(v)


class AlphaZeroMCTS:
    def __init__(self, my_id, net, time_limit_ms=600, device="cpu"):
        self.my_id = my_id; self.net = net
        self.time_limit = time_limit_ms / 1000.0
        self.device = torch.device(device)
        self.net.to(self.device).eval()
    def _eval(self, node):
        if node.proven_loss: return np.ones(4)*0.25, -1.0
        t = torch.from_numpy(encode_state(node.state, self.my_id)).to(self.device)
        m = decode_policy_mask(node.state, self.my_id)
        return self.net.predict(t, mask=torch.from_numpy(m).to(self.device))
    def search(self, root_state, training=False):
        best, _ = self._run(root_state, training)
        return best
    def search_with_policy(self, root_state, training=True):
        return self._run(root_state, training)
    def _run(self, root_state, training):
        start = time.time()
        root = AZNode(root_state, self.my_id)
        if not root.legal_moves:
            return "up", np.array([0.25,0.25,0.25,0.25])
        probs, _ = self._eval(root)
        root.expand(probs, add_noise=training)
        iters = 0
        while time.time() - start < self.time_limit:
            node = root
            while node.is_expanded and not node.is_terminal:
                node = node.select_child()
            if node.is_terminal or node.proven_loss:
                v = -1.0 if node.proven_loss else 0.0
            else:
                probs, v = self._eval(node)
                node.expand(probs, False)
            node.backup(v)
            iters += 1
        vp = np.zeros(4, dtype=np.float32)
        tot = sum(c.N for c in root.children.values())
        for m, c in root.children.items():
            vp[MOVE_TO_IDX[m]] = c.N / max(tot, 1)
        if training and vp.sum() > 0:
            best = np.random.choice(MOVE_ORDER, p=vp)
        else:
            pool = {m: c for m,c in root.children.items() if not c.proven_loss} or root.children
            best = max(pool, key=lambda m: pool[m].N)
        return best, vp