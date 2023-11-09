import torch
import numpy as np
from samplers.deterministic_scheme import select_from_candidates


EPSILON = np.finfo(np.float32).tiny
LARGE_NUMBER = 1.e10

class GumbelSampler(torch.nn.Module):
    def __init__(self, k, train_ensemble, val_ensemble, tau=0.1, hard=True):
        super(GumbelSampler, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
        self.train_ensemble = train_ensemble
        self.val_ensemble = val_ensemble

    def forward(self, scores, train = True):
        repeat_sample = self.train_ensemble if train else self.val_ensemble

        nnodes, choices, ensemble = scores.shape
        local_k = min(self.k, choices)
        flat_scores = scores.permute((0, 2, 1)).reshape(nnodes * ensemble, choices)

        # sample several times with
        flat_scores = flat_scores.repeat(repeat_sample, 1)

        m = torch.distributions.gumbel.Gumbel(flat_scores.new_zeros(flat_scores.shape),
                                              flat_scores.new_ones(flat_scores.shape))
        g = m.sample()
        flat_scores = flat_scores + g

        # continuous top k
        khot = flat_scores.new_zeros(flat_scores.shape)
        onehot_approx = flat_scores.new_zeros(flat_scores.shape)
        for i in range(local_k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=flat_scores.device))
            flat_scores = flat_scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(flat_scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = khot.new_zeros(khot.shape)
            val, ind = torch.topk(khot, local_k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        new_mask = res.reshape(repeat_sample, nnodes, ensemble, choices).permute((0, 1, 3, 2))
        return new_mask, None

    @torch.no_grad()
    def validation(self, scores):
        if self.val_ensemble == 1:
            mask = select_from_candidates(scores, self.k)
            return mask[None], None
        else:
            return self.forward(scores, False)
