import torch

LARGE_NUMBER = 1.e10


def select_from_candidates(scores: torch.Tensor, k: int):
    nnodes, choices, ensemble = scores.shape
    if k >= choices:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask
