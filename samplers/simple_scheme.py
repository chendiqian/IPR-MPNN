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
                 n_samples=1,
                 assign_value=False):
        super(SIMPLESampler, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()
        assert n_samples > 0
        self.n_samples = n_samples
        self.assign_value = assign_value

    def forward(self, scores, sample_from_score=True):
        nnodes, choices, ensemble = scores.shape
        local_k = min(self.k, choices)
        flat_scores = scores.permute((0, 2, 1)).reshape(nnodes * ensemble, choices)

        N = 2 ** math.ceil(math.log2(choices))
        if local_k >= N:
            # we don't need to sample
            samples = scores.new_ones(self.n_samples, nnodes, choices, ensemble)
            marginals = scores.new_ones(nnodes, choices, ensemble)
            return samples, marginals

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
            samples = layer.sample(flat_scores, local_k, self.n_samples)
            samples = samples[..., :choices]
            samples = samples.reshape(self.n_samples, nnodes, ensemble, choices).permute((0, 1, 3, 2))
            samples = (samples - marginals[None]).detach() + marginals[None]
            if self.assign_value:
                samples = samples * (marginals[None] + 1.e-7)
        else:
            samples = None

        return samples, marginals

    @torch.no_grad()
    def validation(self, scores):
        if self.n_samples == 1:
            marginals = None
            # do deterministic top-k
            mask = select_from_candidates(scores, self.k)

            if self.assign_value:
                _, marginals = self.forward(scores, sample_from_score=False)
                mask = mask * (marginals + 1.e-7)

            return mask[None], marginals
        else:
            return self.forward(scores)
