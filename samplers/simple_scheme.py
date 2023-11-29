import math

import torch
import torch.nn as nn

from samplers.deterministic_scheme import select_from_candidates
from samplers.simple_pkg.simple import Layer

LARGE_NUMBER = 1.e10


class SIMPLESampler(nn.Module):
    def __init__(self,
                 k,
                 device,
                 val_ensemble=1,
                 train_ensemble=1,
                 assign_value=False):
        super(SIMPLESampler, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()
        assert val_ensemble > 0 and train_ensemble > 0
        self.val_ensemble = val_ensemble
        self.train_ensemble = train_ensemble
        self.assign_value = assign_value

    def forward(self, scores, train=True, sample_from_score=True):
        times_sampled = self.train_ensemble if train else self.val_ensemble

        nnodes, choices, ensemble = scores.shape
        local_k = min(self.k, choices)
        flat_scores = scores.permute((0, 2, 1)).reshape(nnodes * ensemble, choices)

        N = 2 ** math.ceil(math.log2(choices))
        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, self.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        flat_scores = torch.cat(
            [flat_scores,
             torch.full((flat_scores.shape[0], N - flat_scores.shape[1]),
                        fill_value=-LARGE_NUMBER,
                        dtype=flat_scores.dtype,
                        device=flat_scores.device)],
            dim=1)

        marginals = layer.log_pr(flat_scores).exp().permute(1, 0)
        marginals = marginals[:, :choices]  # unpadding
        marginals = marginals.reshape(nnodes, ensemble, choices).permute((0, 2, 1))

        if sample_from_score:
            # we potentially need to sample multiple times
            samples = layer.sample(flat_scores, local_k, times_sampled)
            samples = samples[..., :choices]
            samples = samples.reshape(times_sampled, nnodes, ensemble, choices).permute((0, 1, 3, 2))
            samples = (samples - marginals[None]).detach() + marginals[None]
            if self.assign_value:
                samples = samples * (marginals[None] + 1.e-7)
        else:
            samples = None

        return samples, marginals

    @torch.no_grad()
    def validation(self, scores):
        if self.val_ensemble == 1:
            _, marginals = self.forward(scores, False, sample_from_score=False)

            # do deterministic top-k
            mask = select_from_candidates(scores, self.k)

            if self.assign_value:
                mask = mask * (marginals + 1.e-7)

            return mask[None], marginals
        else:
            return self.forward(scores, False)
