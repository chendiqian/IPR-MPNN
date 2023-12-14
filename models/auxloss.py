import torch
import numpy as np
from torch_scatter import scatter_sum, scatter_mean


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
    if hasattr(auxloss_dict, 'partition') and auxloss_dict.partition > 0.:
        assert hasattr(data, 'partition')
        scores = scores.permute(2, 0, 1).reshape(n_ensemble * nnodes, n_centroids)
        labels = data.partition.repeat(n_ensemble)
        assert scores.shape[1] >= data.y.max() + 1
        auxloss = auxloss + torch.nn.CrossEntropyLoss()(scores, labels) * auxloss_dict.partition
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
    if hasattr(auxloss_dict, 'hard_empty') and auxloss_dict.hard_empty > 0.:
        # the grad will be all negative, thus the scores will all increase
        # tensor([[-0.3333, -0.0556, -0.3333],
        #         [-0.3333, -0.0556, -0.3333],
        #         [-0.3333, -0.0556, -0.3333],
        #         [-0.3333, -0.0556, -0.3333],
        #         [-0.3333, -0.0556, -0.3333]], device='cuda:0')

        thresh = torch.topk(scores, 1, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
        node_mask = (scores >= thresh).to(torch.float)
        node_mask = node_mask - scores.detach() + scores

        counts = scatter_sum(node_mask, data.batch, dim=0)
        loss = - torch.log(counts + 1.).sum(1).mean()  # otherwise the grad too large
        auxloss = auxloss + auxloss_dict.hard_empty * loss
    if hasattr(auxloss_dict, 'soft_empty'):
        # more flexible
        # tensor([[-0.0013, 0.0029, -0.0017],
        #         [-0.0013, 0.0029, -0.0016],
        #         [-0.0013, 0.0030, -0.0017],
        #         [-0.0012, 0.0029, -0.0017],
        #         [-0.0013, 0.0029, -0.0016]], device='cuda:0')

        node_mask = torch.softmax(scores / 10., dim=1)

        counts = scatter_mean(node_mask, data.batch, dim=0)
        loss = - torch.log(counts).sum(1).mean()
        auxloss = auxloss + auxloss_dict.soft_empty * loss
    if hasattr(auxloss_dict, 'kl') and auxloss_dict.kl > 0:
        # give some grad like
        # tensor([[1.3610, -0.6805, -0.6805],
        #         [1.3610, -0.6805, -0.6805],
        #         [1.3610, -0.6805, -0.6805],
        #         [1.3610, -0.6805, -0.6805],
        #         [1.3610, -0.6805, -0.6805]], device='cuda:0')

        thresh = torch.topk(scores, 1, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
        node_mask = (scores >= thresh).to(torch.float)
        node_mask = node_mask - scores.detach() + scores

        counts = scatter_sum(node_mask, data.batch, dim=0)
        loss = - torch.log_softmax(counts, dim=1).sum(1).mean()
        auxloss = auxloss + auxloss_dict.kl * loss
    if hasattr(auxloss_dict, 'scale') and auxloss_dict.scale > 0.:
        loss = (scores ** 2).mean()
        auxloss = auxloss + auxloss_dict.scale * loss
    return auxloss
