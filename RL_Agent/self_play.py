
import os, random, time, pickle, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque

from forward_model import GameState, Snake, MOVES
from state_encoder  import encode_state
from neural_net     import BattlesnakeNet
from az_mcts        import AlphaZeroMCTS

class ReplayBuffer(Dataset):
    def __init__(self, maxn=300_000):
        self.buf = deque(maxlen=maxn)
    def push(self, s, p, v):
        self.buf.append((s.astype(np.float32), p.astype(np.float32), np.float32(v)))
    def __len__(self): return len(self.buf)
    def __getitem__(self, i):
        s,p,v = self.buf[i]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v)
    def save(self, path):
        with open(path,"wb") as f: pickle.dump(list(self.buf), f)
    def load(self, path):
        if not os.path.exists(path): return
        with open(path,"rb") as f: data=pickle.load(f)
        for x in data: self.buf.append(x)
        print(f"[Buffer] Loaded {len(data)} samples")

def make_start(bsize=11, nsnakes=4):
    interior=[(x,y) for x in range(2,bsize-2) for y in range(2,bsize-2)]
    positions=random.sample(interior, nsnakes)
    occupied=set(positions)
    snakes={}
    for i,(x,y) in enumerate(positions):
        snakes[f"s{i}"]=Snake(f"s{i}", deque([(x,y)]*3), 100)
    pool=[(x,y) for x in range(bsize) for y in range(bsize) if (x,y) not in occupied]
    food=set(random.sample(pool, min(5, len(pool))))
    n_pits=max(0, int(bsize*bsize*0.08))
    haz_pool=[(x,y) for x,y in pool if (x,y) not in food]
    hazards=set(random.sample(haz_pool, min(n_pits, len(haz_pool))))
    return GameState(bsize, bsize, 0, snakes, food, hazards)

def _get_raw_model(net):
    """Recursively unwraps DataParallel and Compiled wrappers."""
    raw = net
    while hasattr(raw, "module") or hasattr(raw, "_orig_mod"):
        if hasattr(raw, "module"):
            raw = raw.module
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
    return raw

def _run_game_on_device(net, bsize, nsnakes, ms, device):
    state = make_start(bsize, nsnakes)
    hist  = []
    maxT  = bsize * bsize * 4
    
    # CRITICAL: MCTS needs the custom .predict method which is only on the raw model
    inference_net = _get_raw_model(net)
    
    for _ in range(maxT):
        alive = [sid for sid,s in state.snakes.items() if s.is_alive]
        if len(alive) <= 1: break
        joint = {}
        for sid in alive:
            ag = AlphaZeroMCTS(sid, inference_net, ms, device)
            mv, vp = ag.search_with_policy(state, training=True)
            joint[sid] = mv
            hist.append((encode_state(state, sid), vp, sid))
        state = state.step(joint)
    
    alive_end = {sid for sid,s in state.snakes.items() if s.is_alive}
    out = []
    for tensor, policy, sid in hist:
        v = 1.0 if sid in alive_end else (0.0 if not alive_end else -1.0)
        out.append((tensor, policy, v))
    return out

def run_game(net, bsize=11, nsnakes=4, ms=150, device="cpu"):
    try:
        return _run_game_on_device(net, bsize, nsnakes, ms, device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print(f"  [CUDA crash] Retrying on CPU...")
            raw_net = _get_raw_model(net)
            # Create a fresh CPU-based instance of the network class
            net_cpu = type(raw_net)(raw_net.in_channels, raw_net.num_filters, raw_net.num_res_blocks)
            net_cpu.load_state_dict(raw_net.state_dict())
            return _run_game_on_device(net_cpu, bsize, nsnakes, ms, "cpu")
        raise

def train_batch(net, opt, buf, batch=512, device="cpu"):
    if len(buf) < batch: return 0.0, 0.0
    net.train()
    loader = DataLoader(buf, batch_size=batch, shuffle=True, drop_last=True)
    s, p, v = next(iter(loader))
    s, p, v = s.to(device), p.to(device), v.to(device)
    
    lg, pv = net(s)
    pl = -(p * torch.log_softmax(lg, -1)).sum(-1).mean()
    vl = ((pv.squeeze() - v) ** 2).mean()
    loss = pl + vl
    
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return float(pl.detach()), float(vl.detach())

def train(cfg, pretrained_net=None):
    net = pretrained_net if pretrained_net is not None else BattlesnakeNet(10, cfg.filters, cfg.res_blocks).to(cfg.device)
    opt = optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters, eta_min=1e-5)
    buf = ReplayBuffer(300_000)
    start = 0
    device = cfg.device 

    raw_net = _get_raw_model(net)

    if os.path.exists(cfg.ckpt) and pretrained_net is None:
        ck = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
        raw_net.load_state_dict(ck["model_state"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start = ck.get("iteration", 0)
        print(f"Resumed from iter {start}")
    
    buf.load(cfg.buf)

    for i in range(start, cfg.iters):
        t0 = time.time()
        new_samples = 0
        for g in range(cfg.games):
            try:
                for sample in run_game(net, cfg.board, cfg.nsnakes, cfg.ms, device):
                    buf.push(*sample)
                    new_samples += 1
            except Exception as e:
                print(f"  Game {g+1} error: {str(e)[:100]}")

        total_pl = total_vl = 0.0
        if len(buf) >= cfg.batch:
            for _ in range(cfg.steps):
                pl, vl = train_batch(net, opt, buf, cfg.batch, device)
                total_pl += pl; total_vl += vl
            total_pl /= cfg.steps; total_vl /= cfg.steps

        sched.step()
        elapsed = time.time() - t0
        print(f"[{i+1:4d}/{cfg.iters}] buf={len(buf):6d} +{new_samples:4d} | "
              f"pl={total_pl:.4f} vl={total_vl:.4f} | lr={sched.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Always save the UNWRAPPED model weights
        torch.save({
            "model_state": raw_net.state_dict(), 
            "opt": opt.state_dict(),
            "iteration": i+1,
            "config": {"in_channels": 10, "num_filters": cfg.filters, "num_res_blocks": cfg.res_blocks}
        }, cfg.ckpt)
        
        if (i+1) % 50 == 0: buf.save(cfg.buf)

    buf.save(cfg.buf)
    return net 