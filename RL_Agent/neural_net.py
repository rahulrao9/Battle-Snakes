import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, f, d=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(f)
        self.c2 = nn.Conv2d(f, f, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(f)
        self.drop = nn.Dropout2d(p=d) if d > 0 else nn.Identity()
    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.drop(x)
        x = self.b2(self.c2(x))
        return F.relu(x + r)


class BattlesnakeNet(nn.Module):
    MOVE_ORDER = ["up", "down", "left", "right"]
    def __init__(self, in_channels=10, num_filters=64, num_res_blocks=6, dropout=0.1):
        super().__init__()
        self.in_channels    = in_channels
        self.num_filters    = num_filters
        self.num_res_blocks = num_res_blocks
        self.stem   = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters), nn.ReLU(inplace=True))
        self.tower  = nn.Sequential(*[ResidualBlock(num_filters, dropout) for _ in range(num_res_blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(num_filters, 32, 1, bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.p_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.p_fc   = nn.Linear(32*4*4, 4)
        self.v_conv = nn.Sequential(nn.Conv2d(num_filters, 16, 1, bias=False),
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.v_gap  = nn.AdaptiveAvgPool2d(1)
        self.v_fc   = nn.Sequential(nn.Flatten(), nn.Linear(16, 64),
                                    nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.p_pool(self.p_conv(x)).flatten(1)
        return self.p_fc(p), self.v_fc(self.v_gap(self.v_conv(x)))
    @torch.no_grad()
    def predict(self, t, mask=None):
        self.eval()
        if t.dim() == 3: t = t.unsqueeze(0)
        logits, v = self.forward(t)
        if mask is not None:
            if isinstance(mask, np.ndarray): mask = torch.from_numpy(mask).to(t.device)
            logits = logits + (1 - mask.unsqueeze(0)) * -1e9
        probs = torch.softmax(logits, -1).squeeze(0).cpu().numpy()
        return probs, float(v.squeeze().cpu())
    def save(self, path, extra=None):
        p = {"model_state": self.state_dict(),
             "config": {"in_channels": self.in_channels,
                        "num_filters": self.num_filters,
                        "num_res_blocks": self.num_res_blocks}}
        if extra: p.update(extra)
        torch.save(p, path)
    def load(self, path, device="cpu"):
        if not os.path.exists(path): return self
        p = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(p["model_state"])
        return self
    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        p = torch.load(path, map_location=device, weights_only=True)
        net = cls(**p.get("config", {}))
        net.load_state_dict(p["model_state"])
        return net.to(device).eval()