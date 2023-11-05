import torch
from deterministic_scheme import select_from_candidates

LARGE_NUMBER = 1.e10

class IMLEScheme:
    def __init__(self, sample_k, train_ensemble, val_ensemble):
        self.k = sample_k
        self.train_ensemble = train_ensemble
        self.val_ensemble = val_ensemble

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):
        local_logits = logits.detach()
        mask = select_from_candidates(local_logits, self.k)
        return mask, None
