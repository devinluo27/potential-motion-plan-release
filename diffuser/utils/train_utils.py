import math
import os, copy
from typing import Tuple
import numpy as np
import torch
from typing import Union
from torch.optim.lr_scheduler import _LRScheduler


def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """CosineLR with Warmup.

    Code borrowed from:
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification. Default: 1.
        max_lr (float | Tuple): First cycle's max learning rate. Default: 0.1.
        min_lr (float | Tuple): Min learning rate. Default: 0.001.
        warmup_steps (int): Linear warmup step size. Default: 0.
        gamma (float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.,
        max_lr: Union[float, Tuple] = 0.1,
        min_lr: Union[float, Tuple] = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        num_opt = len(optimizer.param_groups)
        if isinstance(max_lr, float):
            max_lr = [max_lr] * num_opt
        else:
            assert isinstance(max_lr, tuple)
            assert len(max_lr) == num_opt
        if isinstance(min_lr, float):
            min_lr = [min_lr] * num_opt
        else:
            assert isinstance(min_lr, tuple)
            assert len(min_lr) == num_opt

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = self.min_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i]

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (max_lr - base_lr) * self.step_in_cycle / self.warmup_steps +
                base_lr for max_lr, base_lr in zip(self.max_lr, self.base_lrs)
            ]
        else:
            return [
                base_lr + (max_lr - base_lr) *
                (1 + math.cos(math.pi *
                              (self.step_in_cycle - self.warmup_steps) /
                              (self.cur_cycle_steps - self.warmup_steps))) / 2
                for max_lr, base_lr in zip(self.max_lr, self.base_lrs)
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) *
                    self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log((epoch / self.first_cycle_steps *
                                  (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps *
                                                     (self.cycle_mult**n - 1) /
                                                     (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * \
                        (self.cycle_mult**n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = [
            lr * (self.gamma**self.cycle) for lr in self.base_max_lr
        ]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_lr(optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']
