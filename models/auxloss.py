import torch


def get_auxloss(auxloss_dict, pool, graph_pool_idx, scores, data):
    auxloss = 0.
    if hasattr(auxloss_dict, 'scorer_label_supervised') and auxloss_dict.scorer_label_supervised > 0.:
        # this must be node prediction task
        assert graph_pool_idx == 'output_mask'

        scores = pool(scores, getattr(data, graph_pool_idx))
        nnodes, n_centroids, n_ensemble = scores.shape
        scores = scores.permute(2, 0, 1).reshape(n_ensemble * nnodes, n_centroids)
        labels = data.y.repeat(n_ensemble)
        assert scores.shape[1] >= data.y.max() + 1
        auxloss = auxloss + torch.nn.CrossEntropyLoss()(scores, labels) * auxloss_dict.scorer_label_supervised

    return auxloss
