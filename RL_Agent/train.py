"""
train_local.py — Battlesnake AlphaZero Local Training
======================================================
Usage:
    python train_local.py                        # fresh run
    python train_local.py --ckpt my_net.pt       # custom checkpoint path
    python train_local.py --iters 200 --ms 80    # override config
    python train_local.py --eval-only            # skip training, just evaluate
"""

import argparse
import os
import types

import numpy as np
import torch
import torch.nn as nn

# ── Parse args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       default="battlesnake_net.pt")
parser.add_argument("--buf",        default="replay_buffer.pkl")
parser.add_argument("--iters",      type=int,   default=300)
parser.add_argument("--games",      type=int,   default=30)
parser.add_argument("--steps",      type=int,   default=80)
parser.add_argument("--batch",      type=int,   default=512)
parser.add_argument("--ms",         type=int,   default=40)
parser.add_argument("--filters",    type=int,   default=64)
parser.add_argument("--res-blocks", type=int,   default=8)
parser.add_argument("--pretrain-games", type=int, default=300)
parser.add_argument("--pretrain-epochs", type=int, default=5)
parser.add_argument("--eval-only",  action="store_true")
parser.add_argument("--eval-games", type=int,   default=10)
parser.add_argument("--eval-ms",    type=int,   default=200)
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    print("No GPU — running on CPU (slower)")
print(f"Device: {DEVICE}\n")

# ── Config ────────────────────────────────────────────────────────────────────

cfg = types.SimpleNamespace(
    ckpt       = args.ckpt,
    buf        = args.buf,
    iters      = args.iters,
    games      = args.games,
    steps      = args.steps,
    batch      = args.batch,
    ms         = args.ms,
    board      = 11,
    nsnakes    = 4,
    filters    = args.filters,
    res_blocks = args.res_blocks,
    device     = DEVICE,
)

iter_time_est = cfg.games * 4 * 100 * cfg.ms / 1000
print(f"Config: {cfg.iters} iters | {cfg.games} games | {cfg.ms}ms MCTS | batch {cfg.batch}")
print(f"Estimated ~{iter_time_est:.0f}s/iter self-play | "
      f"~{cfg.steps} gradient steps/iter")
print(f"Network: {cfg.filters}f x {cfg.res_blocks} res blocks\n")

# ── Imports (local files must be in same directory) ───────────────────────────

from neural_net    import BattlesnakeNet
from az_mcts       import AlphaZeroMCTS
from forward_model import GameState, Snake, MOVES
from state_encoder import encode_state, decode_policy_mask, MOVE_TO_IDX
from self_play     import ReplayBuffer, make_start, run_game, train_batch, train

# ── Build network ─────────────────────────────────────────────────────────────

net = BattlesnakeNet(10, cfg.filters, cfg.res_blocks)
print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
net = net.to(DEVICE)

# DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    net = nn.DataParallel(net)

# torch.compile for speed (Python 3.11+, PyTorch 2.0+)
try:
    net = torch.compile(net, mode="reduce-overhead")
    print("torch.compile: enabled")
except Exception as e:
    print(f"torch.compile skipped: {e}")

print()

# ── Heuristic flood-fill for supervised pretraining ───────────────────────────

def heuristic_move(state, sid):
    from collections import deque as dq
    snake = state.snakes.get(sid)
    if not snake or not snake.is_alive:
        return "up"
    W, H = state.board_width, state.board_height
    head, my_len, my_hp = snake.head, snake.length, snake.health

    obs = {}
    for s in state.snakes.values():
        if not s.is_alive:
            continue
        is_fed = s.health == 100
        for i, pt in enumerate(reversed(list(s.body))):
            obs[pt] = max(obs.get(pt, 0), i + (1 if is_fed else 0))

    danger, kill = set(), set()
    for s in state.snakes.values():
        if s.id == sid or not s.is_alive:
            continue
        for dx, dy in MOVES.values():
            nx, ny = s.head[0] + dx, s.head[1] + dy
            (danger if s.length >= my_len else kill).add((nx, ny))

    def flood(start):
        visited, q = {start}, dq([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in MOVES.values():
                nb = (cx + dx, cy + dy)
                if nb in visited:
                    continue
                if not (0 <= nb[0] < W and 0 <= nb[1] < H):
                    continue
                if nb in obs and obs[nb] >= 1:
                    continue
                visited.add(nb)
                q.append(nb)
        return len(visited)

    scores = {}
    for m, (dx, dy) in MOVES.items():
        nx, ny = head[0] + dx, head[1] + dy
        nb = (nx, ny)
        if not (0 <= nx < W and 0 <= ny < H) or (nb in obs and obs[nb] >= 1):
            scores[m] = -1e9
            continue
        sc = flood(nb) * 2.0
        if nb in danger:      sc -= 500.0
        elif nb in kill:      sc += 150.0
        if nb in state.food:  sc += 200.0 if my_hp < 60 else 20.0
        scores[m] = sc
    return max(scores, key=scores.get)


def generate_supervised_data(n_games=300):
    data = []
    for g in range(n_games):
        state = make_start(cfg.board, cfg.nsnakes)
        for _ in range(cfg.board * cfg.board * 4):
            alive = [sid for sid, s in state.snakes.items() if s.is_alive]
            if len(alive) <= 1:
                break
            joint = {}
            for sid in alive:
                mv    = heuristic_move(state, sid)
                legal = state.get_action_space(sid)
                if mv not in legal:
                    mv = legal[0] if legal else "up"
                tensor = encode_state(state, sid)
                mask   = decode_policy_mask(state, sid)
                target = np.zeros(4, dtype=np.float32)
                target[MOVE_TO_IDX[mv]] = 1.0
                data.append((tensor, target, mask))
                joint[sid] = mv
            state = state.step(joint)
        if (g + 1) % 50 == 0:
            print(f"  game {g+1}/{n_games}  ({len(data):,} samples)")
    return data


def pretrain(net, n_games, epochs, batch, lr=1e-3):
    print(f"Supervised pretraining: {n_games} heuristic games, {epochs} epochs...")
    data = generate_supervised_data(n_games)
    print(f"  {len(data):,} (state, move) pairs collected.")

    raw = net.module if hasattr(net, "module") else net
    raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw

    opt   = torch.optim.Adam(raw.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(data) - batch, batch):
            bd      = data[i:i + batch]
            states  = torch.from_numpy(np.stack([d[0] for d in bd])).to(DEVICE)
            targets = torch.from_numpy(np.stack([d[1] for d in bd])).to(DEVICE)
            masks   = torch.from_numpy(np.stack([d[2] for d in bd])).to(DEVICE)
            net.train()
            logits, _ = net(states)
            logits = logits + (1 - masks) * -1e9
            loss = -(targets * torch.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach())
            n_batches  += 1
        sched.step()
        print(f"  epoch {epoch+1}/{epochs} | "
              f"loss={total_loss/max(n_batches,1):.4f} | "
              f"lr={sched.get_last_lr()[0]:.2e}")

    # Save pretrained checkpoint (iteration=0)
    raw = net.module if hasattr(net, "module") else net
    raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    torch.save({
        "model_state": raw.state_dict(),
        "iteration":   0,
        "config": {"in_channels": 10,
                   "num_filters": cfg.filters,
                   "num_res_blocks": cfg.res_blocks},
    }, cfg.ckpt)
    print(f"Pretrained checkpoint saved → {cfg.ckpt}\n")
    return net

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(n_games=10, ms=200):
    print(f"\nEvaluating over {n_games} games (CPU, {ms}ms/move)...")
    eval_net = BattlesnakeNet(10, cfg.filters, cfg.res_blocks)
    if os.path.exists(cfg.ckpt):
        ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
        eval_net.load_state_dict(ck["model_state"])
        print(f"Loaded checkpoint (iter {ck.get('iteration','?')})")
    else:
        print("No checkpoint — using random weights")
    eval_net.eval()

    wins = draws = losses = 0
    for g in range(n_games):
        state = make_start(11, 4)
        for _ in range(11 * 11 * 4):
            alive = [sid for sid, s in state.snakes.items() if s.is_alive]
            if len(alive) <= 1:
                break
            joint = {}
            for sid in alive:
                try:
                    ag = AlphaZeroMCTS(sid, eval_net, ms, "cpu")
                    joint[sid] = ag.search(state, training=False)
                except Exception:
                    joint[sid] = state.get_guided_move(sid)
            state = state.step(joint)
        alive_end = {sid for sid, s in state.snakes.items() if s.is_alive}
        if "s0" in alive_end and len(alive_end) == 1:
            wins   += 1; result = "WIN"
        elif not alive_end or "s0" in alive_end:
            draws  += 1; result = "draw"
        else:
            losses += 1; result = "loss"
        print(f"  Game {g+1}: {result}")

    print(f"\nResults ({n_games} games, s0 = our agent):")
    print(f"  Wins:   {wins}/{n_games}  ({100*wins/n_games:.0f}%)")
    print(f"  Draws:  {draws}/{n_games}  ({100*draws/n_games:.0f}%)")
    print(f"  Losses: {losses}/{n_games}  ({100*losses/n_games:.0f}%)")
    print(f"  Random baseline: 25%")
    if wins / n_games > 0.35:
        print("  STATUS: above random baseline ✓")
    elif wins / n_games > 0.20:
        print("  STATUS: near random — needs more training")
    else:
        print("  STATUS: below random — check training logs")

# ── Main ──────────────────────────────────────────────────────────────────────

if args.eval_only:
    evaluate(args.eval_games, args.eval_ms)
else:
    # Pretraining — skip if checkpoint already exists
    if os.path.exists(cfg.ckpt):
        print(f"Checkpoint found at '{cfg.ckpt}' — loading weights, skipping pretraining.")
        ck = torch.load(cfg.ckpt, map_location=DEVICE, weights_only=False)
        raw = net.module if hasattr(net, "module") else net
        raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        raw.load_state_dict(ck["model_state"])
        print(f"  Resumed from iteration {ck.get('iteration', '?')}/{cfg.iters}\n")
    else:
        net = pretrain(net,
                       n_games = args.pretrain_games,
                       epochs  = args.pretrain_epochs,
                       batch   = cfg.batch)

    # Self-play training loop
    train(cfg, pretrained_net=net)
    print("Training complete!")

    # Evaluate after training
    evaluate(args.eval_games, args.eval_ms)