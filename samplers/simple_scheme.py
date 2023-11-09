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
                 train_ensemble=1):
        super(SIMPLESampler, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()
        assert val_ensemble > 0 and train_ensemble > 0
        self.val_ensemble = val_ensemble
        self.train_ensemble = train_ensemble

    def forward(self, scores, train = True):
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

        # we potentially need to sample multiple times
        marginals = layer.log_pr(flat_scores).exp().permute(1, 0)
        # (times_sampled) x (B x E) x (N x N)
        samples = layer.sample(flat_scores, local_k, times_sampled)
        samples = (samples - marginals[None]).detach() + marginals[None]

        # unpadding
        samples = samples[..., :choices]
        marginals = marginals[:, :choices]

        new_mask = samples.reshape(times_sampled, nnodes, ensemble, choices).permute((0, 1, 3, 2))
        new_marginals = marginals.reshape(nnodes, ensemble, choices).permute((0, 2, 1))

        return new_mask, new_marginals

    @torch.no_grad()
    def validation(self, scores):
        if self.val_ensemble == 1:
            _, marginals = self.forward(scores, False)

            # do deterministic top-k
            mask = select_from_candidates(scores, self.k)
            return mask[None], marginals
        else:
            return self.forward(scores, False)
