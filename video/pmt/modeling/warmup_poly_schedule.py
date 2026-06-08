# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# This file are adapted from:
# - the EoMT repository https://github.com/tue-mps/eomt/blob/master/training/two_stage_warmup_poly_schedule.py
# Used under the MIT License.
# ---------------------------------------------------------------


from torch.optim.lr_scheduler import LRScheduler


class WarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        poly_power: float,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                adjusted = max(0, step - self.warmup_steps)
                max_steps = max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            lrs.append(lr)
        return lrs