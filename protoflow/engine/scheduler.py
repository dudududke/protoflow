from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupPolyLR:
    def __init__(self, optimizer: Optimizer, base_lr: float, max_iters: int, power: float = 0.9, warmup_iters: int = 1000, warmup_lr: float = 1e-4):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.max_iters = max(1, int(max_iters))
        self.power = float(power)
        self.warmup_iters = int(warmup_iters)
        self.warmup_lr = float(warmup_lr)
        self.iter = 0
        self.step(0)

    def get_lr(self, iteration: int) -> float:
        iteration = min(max(0, iteration), self.max_iters)
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            alpha = iteration / max(1, self.warmup_iters)
            return self.warmup_lr * (1.0 - alpha) + self.base_lr * alpha
        progress = (iteration - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
        return self.base_lr * ((1.0 - progress) ** self.power)

    def step(self, iteration: int | None = None) -> float:
        if iteration is None:
            self.iter += 1
        else:
            self.iter = int(iteration)
        lr = self.get_lr(self.iter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
