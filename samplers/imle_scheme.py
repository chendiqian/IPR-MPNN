import torch
import torch.nn as nn
from deterministic_scheme import select_from_candidates

from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle


LARGE_NUMBER = 1.e10

class IMLESampler(nn.Module):
    def __init__(self, sample_k, device, train_ensemble, val_ensemble, noise_scale, beta):
        super(IMLESampler, self).__init__()
        self.k = sample_k

        @imle(target_distribution=TargetDistribution(alpha=1.0, beta=beta),
              noise_distribution=GumbelDistribution(0., noise_scale, device),
              nb_samples=train_ensemble,
              input_noise_temperature=1.,
              target_noise_temperature=1.,)
        def imle_train_scheme(logits: torch.Tensor):
            return self.sample(logits)

        self.train_forward = imle_train_scheme

        @imle(target_distribution=None,
              noise_distribution=GumbelDistribution(0., noise_scale, device) if val_ensemble > 1 else None,
              nb_samples=val_ensemble,
              input_noise_temperature=1.,
              target_noise_temperature=1.,)
        def imle_val_scheme(logits: torch.Tensor):
            return self.sample(logits)

        self.val_forward = imle_val_scheme


    @torch.no_grad()
    def sample(self, logits: torch.Tensor):
        local_logits = logits.detach()
        mask = select_from_candidates(local_logits, self.k)
        return mask, None

    def forward(self, logits: torch.Tensor, train = True):
        if train:
            return self.train_forward(logits)
        else:
            return self.val_forward(logits)

    @torch.no_grad()
    def validation(self, scores):
        return self.forward(scores, False)
