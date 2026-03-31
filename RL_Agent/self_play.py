import os, random, time, pickle, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from forward_model import GameState, Snake, MOVES
from state_encoder  import encode_state
from neural_net     import BattlesnakeNet
from az_mcts        import AlphaZeroMCTS


class ReplayBuffer(Dataset):
    def __init__(self, maxn=300_000):
        self.buf = deque(maxlen=maxn)

    def push(self, s, p, v):
        self.buf.append((s.astype(np.float32), p.astype(np.float32), np.float32(v)))

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        s, p, v = self.buf[i]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(list(self.buf), f)

    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        for x in data:
            self.buf.append(x)
        print(f"[Buffer] Loaded {len(data)} samples")


def make_start(bsize=11, nsnakes=4):
    interior  = [(x, y) for x in range(2, bsize - 2) for y in range(2, bsize - 2)]
    positions = random.sample(interior, nsnakes)
    occupied  = set(positions)
    snakes    = {}
    for i, (x, y) in enumerate(positions):
        snakes[f"s{i}"] = Snake(f"s{i}", deque([(x, y)] * 3), 100)
    pool     = [(x, y) for x in range(bsize) for y in range(bsize) if (x, y) not in occupied]
    food     = set(random.sample(pool, min(5, len(pool))))
    n_pits   = max(0, int(bsize * bsize * 0.08))
    haz_pool = [(x, y) for x, y in pool if (x, y) not in food]
    hazards  = set(random.sample(haz_pool, min(n_pits, len(haz_pool))))
    return GameState(bsize, bsize, 0, snakes, food, hazards)


def _get_raw_model(net):
    raw = net
    while hasattr(raw, "module") or hasattr(raw, "_orig_mod"):
        if hasattr(raw, "module"):    raw = raw.module
        if hasattr(raw, "_orig_mod"): raw = raw._orig_mod
    return raw


def _shaped_value(sid, alive_end, turn, length, bsize):
    max_turns = bsize * bsize * 4
    if sid in alive_end:
        return 1.0
    if not alive_end:
        return 0.0
    survival_bonus = min(turn / max_turns, 1.0) * 0.5
    length_bonus   = min(length / 20.0,   1.0) * 0.3
    return float(np.clip(-1.0 + survival_bonus + length_bonus, -1.0, 1.0))


def run_game(net, bsize=11, nsnakes=4, ms=150, device="cpu"):
    """
    Run one self-play game with 3-frame history passed to every MCTS call.

    game_history is a deque of the last 3 real board states, most-recent first.
    Each MCTS agent receives [S_{t-1}, S_{t-2}] as its root_history so it can
    reconstruct [S_t, S_{t-1}, S_{t-2}] once S_t is appended inside the tree.
    """
    state        = make_start(bsize, nsnakes)
    hist         = []          # replay buffer accumulator
    maxT         = bsize * bsize * 4
    game_history = deque(maxlen=3)   # sliding window, most-recent first

    for t in range(maxT):
        # Prepend current state so index 0 is always the newest frame
        game_history.appendleft(state)

        alive = [sid for sid, s in state.snakes.items() if s.is_alive]
        if len(alive) <= 1:
            break

        joint = {}
        for sid in alive:
            # root_history = frames *before* the current state (S_{t-1}, S_{t-2})
            # MCTS will see [current_node.state, …] via get_history(), which
            # adds root_history only after exhausting tree-internal frames.
            root_hist = list(game_history)[1:]   # skip index 0 (= current state)

            ag = AlphaZeroMCTS(sid, net, ms, device, root_history=root_hist)
            mv, vp = ag.search_with_policy(state, training=True)
            joint[sid] = mv

            # Encode using the full 3-frame window for the replay buffer
            hist.append((
                encode_state(list(game_history), sid),
                vp,
                sid,
                t,
                state.snakes[sid].length,
            ))

        state = state.step(joint)

    alive_end = {sid for sid, s in state.snakes.items() if s.is_alive}
    out = []
    for tensor, policy, sid, turn, length in hist:
        v = _shaped_value(sid, alive_end, turn, length, bsize)
        out.append((tensor, policy, v))
    return out


def train_batch(net, opt, buf, batch=512, device="cpu"):
    if len(buf) < batch:
        return 0.0, 0.0
    net.train()
    loader = DataLoader(buf, batch_size=batch, shuffle=True, drop_last=True)
    s, p, v = next(iter(loader))
    s, p, v = s.to(device), p.to(device), v.to(device)

    lg, pv = net(s)
    pl   = -(p * torch.log_softmax(lg, -1)).sum(-1).mean()
    vl   = ((pv.squeeze() - v) ** 2).mean()
    loss = pl + vl

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(_get_raw_model(net).parameters(), 1.0)
    opt.step()
    return float(pl.detach()), float(vl.detach())


def train(cfg, pretrained_net=None):
    import warnings
    warnings.filterwarnings("ignore")

    net    = pretrained_net
    device = cfg.device
    opt    = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    buf     = ReplayBuffer(300_000)
    start   = 0
    raw_net = _get_raw_model(net)

    net_config = {
        "in_channels":    raw_net.in_channels,
        "num_filters":    raw_net.num_filters,
        "num_res_blocks": raw_net.num_res_blocks,
    }

    if os.path.exists(cfg.ckpt) and pretrained_net is None:
        ck = torch.load(cfg.ckpt, map_location=device, weights_only=False)
        raw_net.load_state_dict(ck["model_state"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = ck.get("iteration", 0)
        print(f"Resumed from iter {start}")

    buf.load(cfg.buf)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.iters, eta_min=1e-5,
        last_epoch=start - 1 if start > 0 else -1
    )

    writer = SummaryWriter(log_dir="runs/self_play_experiment")

    for i in range(start, cfg.iters):
        t0 = time.time()

        net.eval()
        new_samples = 0
        for g in range(cfg.games):
            try:
                samples = run_game(raw_net, cfg.board, cfg.nsnakes, cfg.ms, device)
                for sample in samples:
                    buf.push(*sample)
                    new_samples += 1
            except Exception as e:
                print(f"  [Game {g} error] {e}")
            if (g + 1) % 10 == 0:
                print(f"  self-play {g+1}/{cfg.games} | "
                      f"+{new_samples} samples | "
                      f"~{(time.time()-t0)/((g+1)/10):.0f}s per 10 games",
                      flush=True)

        sp_time = time.time() - t0

        net.train()
        total_pl = total_vl = 0.0
        if len(buf) >= cfg.batch:
            for _ in range(cfg.steps):
                pl, vl = train_batch(net, opt, buf, cfg.batch, device)
                total_pl += pl
                total_vl += vl
            total_pl /= cfg.steps
            total_vl /= cfg.steps

        sched.step()
        elapsed = time.time() - t0
        print(
            f"[{i+1:4d}/{cfg.iters}] buf={len(buf):6d} +{new_samples:4d} | "
            f"pl={total_pl:.4f} vl={total_vl:.4f} | "
            f"lr={sched.get_last_lr()[0]:.2e} | "
            f"sp={sp_time:.0f}s total={elapsed:.0f}s",
            flush=True,
        )

        writer.add_scalar("Loss/Policy",                  total_pl,               i)
        writer.add_scalar("Loss/Value",                   total_vl,               i)
        writer.add_scalar("Hyperparameters/LearningRate", sched.get_last_lr()[0], i)
        writer.add_scalar("System/BufferSize",            len(buf),               i)
        writer.add_scalar("System/SamplesPerIter",        new_samples,            i)

        torch.save(
            {
                "model_state": raw_net.state_dict(),
                "opt":         opt.state_dict(),
                "iteration":   i + 1,
                "config":      net_config,
            },
            cfg.ckpt,
        )

        if (i + 1) % 50 == 0:
            buf.save(cfg.buf)

    buf.save(cfg.buf)
    writer.close()
    return net