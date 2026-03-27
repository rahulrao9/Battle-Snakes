"""
train.py — Battlesnake AlphaZero Local Training
======================================================
Usage:
    python train.py                          # fresh run
    python train.py --ckpt my_net.pt         # custom checkpoint path
    python train.py --iters 200 --ms 80      # override config
    python train.py --eval-only              # skip training, just evaluate
"""

import argparse
import os
import types
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",            default="battlesnake_net.pt")
parser.add_argument("--buf",             default="replay_buffer.pkl")
parser.add_argument("--iters",           type=int,   default=150)
parser.add_argument("--games",           type=int,   default=60)
parser.add_argument("--steps",           type=int,   default=300)
parser.add_argument("--batch",           type=int,   default=512)
parser.add_argument("--ms",              type=int,   default=80)
parser.add_argument("--filters",         type=int,   default=128)
parser.add_argument("--res-blocks",      type=int,   default=12)
parser.add_argument("--pretrain-games",  type=int,   default=300)
parser.add_argument("--pretrain-epochs", type=int,   default=10)
parser.add_argument("--eval-only",       action="store_true")
parser.add_argument("--eval-games",      type=int,   default=10)
parser.add_argument("--eval-ms",         type=int,   default=200)
parser.add_argument("--lr",              type=float, default=5e-4)
parser.add_argument("--resume-lr",       type=float, default=1e-4)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    lr         = args.lr,
    resume_lr  = args.resume_lr,
)

# ── Heuristic flood-fill for supervised pretraining ───────────────────────────

def heuristic_move(state, sid):
    from collections import deque as dq
    snake = state.snakes.get(sid)
    if not snake or not snake.is_alive:
        return "up"
    from forward_model import MOVES
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
        if nb in danger:     sc -= 500.0
        elif nb in kill:     sc += 150.0
        if nb in state.food: sc += 200.0 if my_hp < 60 else 20.0
        scores[m] = sc
    return max(scores, key=scores.get)


def generate_supervised_data(n_games, cfg):
    from state_encoder import decode_policy_mask, MOVE_TO_IDX
    from self_play import make_start
    data = []
    for g in range(n_games):
        state = make_start(cfg.board, cfg.nsnakes)
        for _ in range(cfg.board * cfg.board * 4):
            alive = [sid for sid, s in state.snakes.items() if s.is_alive]
            if len(alive) <= 1:
                break
            joint = {}
            for sid in alive:
                from state_encoder import encode_state
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


def pretrain(net, cfg):
    n_games = args.pretrain_games
    epochs  = args.pretrain_epochs
    batch   = cfg.batch

    print(f"Supervised pretraining: {n_games} heuristic games, {epochs} epochs...")
    data = generate_supervised_data(n_games, cfg)
    print(f"  {len(data):,} (state, move) pairs collected.")

    from self_play import _get_raw_model
    from state_encoder import MOVE_TO_IDX
    raw = _get_raw_model(net)

    opt   = torch.optim.Adam(raw.parameters(), lr=1e-3)
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
        print(
            f"  epoch {epoch+1}/{epochs} | "
            f"loss={total_loss/max(n_batches,1):.4f} | "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )

    raw2 = _get_raw_model(net)
    torch.save(
        {
            "model_state": raw2.state_dict(),
            "iteration":   0,
            "config": {
                "in_channels":    10,
                "num_filters":    cfg.filters,
                "num_res_blocks": cfg.res_blocks,
            },
        },
        cfg.ckpt,
    )
    print(f"Pretrained checkpoint saved → {cfg.ckpt}\n")
    return net


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(cfg, n_games=10, ms=200):
    from neural_net import BattlesnakeNet
    from az_mcts    import AlphaZeroMCTS
    from self_play  import make_start

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
            wins  += 1; result = "WIN"
        elif not alive_end or "s0" in alive_end:
            draws += 1; result = "draw"
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


# ── Entry point ───────────────────────────────────────────────────────────────
# FIX: network is built ONLY inside this guard.
# Building it at module level caused it to be constructed in every spawned
# worker process that imports this file, wasting memory and triggering CUDA
# re-initialisation errors.

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    from neural_net import BattlesnakeNet
    from self_play  import _get_raw_model, train

    if DEVICE == "cuda":
        print(
            f"GPU : {torch.cuda.get_device_name(0)}  "
            f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)"
        )
    else:
        print("No GPU — running on CPU (slower)")
    print(f"Device: {DEVICE}\n")

    iter_time_est = cfg.games * 4 * 100 * cfg.ms / 1000
    print(f"Config: {cfg.iters} iters | {cfg.games} games | {cfg.ms}ms MCTS | batch {cfg.batch}")
    print(
        f"Estimated ~{iter_time_est:.0f}s/iter self-play | "
        f"~{cfg.steps} gradient steps/iter"
    )
    print(f"Network: {cfg.filters}f x {cfg.res_blocks} res blocks\n")

    net = BattlesnakeNet(10, cfg.filters, cfg.res_blocks)
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    net = net.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net = nn.DataParallel(net)

    try:
        net = torch.compile(net, mode="reduce-overhead")
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    print()

    if args.eval_only:
        evaluate(cfg, args.eval_games, args.eval_ms)
    else:
        if os.path.exists(cfg.ckpt):
            print(f"Checkpoint found at '{cfg.ckpt}' — loading weights...")
            ck = torch.load(cfg.ckpt, map_location=DEVICE, weights_only=False)

            raw = _get_raw_model(net)
            raw.load_state_dict(ck["model_state"])

            start_iter = ck.get("iteration", 0)
            if start_iter > 0:
                cfg.lr = cfg.resume_lr
                print(f"  Resuming from iter {start_iter} → lowering LR to {cfg.lr:.2e}")
            else:
                print("  Loaded pretrained model (iter 0)")
            print()
        else:
            net = pretrain(net, cfg)

        train(cfg, pretrained_net=net)
        print("Training complete!")

        evaluate(cfg, args.eval_games, args.eval_ms)