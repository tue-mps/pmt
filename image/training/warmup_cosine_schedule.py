# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineSchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-9,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                adjusted = max(0, step - self.warmup_steps)
                max_steps = max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * adjusted / max_steps)
                )
            lrs.append(lr)
        return lrs
