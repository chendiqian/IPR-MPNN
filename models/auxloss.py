import torch
import numpy as np


def sort_lexico(mask: torch.Tensor):
    # mask: n_centroids x nnodes
    np_mask = mask.detach().cpu().numpy().astype(np.int32)
    strings = [''.join(str(s) for s in m) for m in np_mask]
    args = np.argsort(strings)
    return mask[args].reshape(-1)


def get_auxloss(auxloss_dict, pool, graph_pool_idx, scores, data):
    auxloss = 0.
    nnodes, n_centroids, n_ensemble = scores.shape
    if hasattr(auxloss_dict, 'scorer_label_supervised') and auxloss_dict.scorer_label_supervised > 0.:
        # this must be node prediction task
        assert graph_pool_idx == 'output_mask'

        scores = pool(scores, getattr(data, graph_pool_idx))
        scores = scores.permute(2, 0, 1).reshape(n_ensemble * nnodes, n_centroids)
        labels = data.y.repeat(n_ensemble)
        assert scores.shape[1] >= data.y.max() + 1
        auxloss = auxloss + torch.nn.CrossEntropyLoss()(scores, labels) * auxloss_dict.scorer_label_supervised
    if hasattr(auxloss_dict, 'variance') and auxloss_dict.variance != 0.:
        if n_ensemble > 1:
            thresh = torch.topk(scores, 1, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
            node_mask = (scores >= thresh).to(torch.float)
            node_mask = node_mask - scores.detach() + scores

            node_mask = node_mask.permute(2, 1, 0)

            cumsum_nnodes = data._slice_dict['x']
            nnodes_list = cumsum_nnodes[1:] - cumsum_nnodes[:-1]

            masks = torch.split(node_mask, nnodes_list.tolist(), dim=2)

            flat_lexico_masks = []
            for mask in masks:
                # mask: repeats x n_centroids x nnode -> repeats x (n_centroids x nnode)
                flat_lexico_mask = torch.stack([sort_lexico(mask[i]) for i in range(n_ensemble)], dim=0)
                flat_lexico_masks.append(torch.log_softmax(flat_lexico_mask, dim=1))

            flat_lexico_masks = torch.hstack(flat_lexico_masks)

            idx = np.triu_indices(n_ensemble, k=1)
            src = flat_lexico_masks[idx[0]]
            dst = flat_lexico_masks[idx[1]]

            loss = (dst.exp() * (dst - src)).sum(-1).mean() + (src.exp() * (src - dst)).sum(-1).mean()
            loss = loss / data.num_graphs

            # care the sign
            auxloss = auxloss - loss * auxloss_dict.variance
    return auxloss
