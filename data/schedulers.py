from torch.optim import Optimizer
import math
import torch.optim as optim

from data.const import TASK_TYPE_DICT, SCHEDULER_MODE


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int = 5, num_training_steps: int = 250,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(args, optimizer):
    target_metric: str = TASK_TYPE_DICT[args.dataset.lower()]
    if hasattr(args, 'scheduler_type') and args.scheduler_type == "cos_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=args.scheduler_warmup
                                                    if hasattr(args, "scheduler_warmup") else 5,
                                                    num_training_steps=args.scheduler_patience if hasattr(args, "scheduler_patience") else 50)

    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=SCHEDULER_MODE[target_metric],
                                                         factor=0.5,
                                                         patience=args.scheduler_patience
                                                         if hasattr(args, "scheduler_patience") else 50,
                                                         min_lr=1.e-5)
    return scheduler
