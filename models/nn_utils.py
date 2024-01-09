from typing import List

import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


def residual(y_old: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
    if y_old.shape == y_new.shape:
        return (y_old + y_new) / 2 ** 0.5
    else:
        return y_new


def get_graph_pooling(graph_pooling):
    graph_pool_idx = 'batch'
    if graph_pooling == "sum":
        pool = global_add_pool
    elif graph_pooling == 'max':
        pool = global_max_pool
    elif graph_pooling == "mean":
        pool = global_mean_pool
    elif graph_pooling is None:  # node pred
        pool = lambda x, *args: x
    elif graph_pooling == 'root':
        pool = lambda x, root_mask: x[root_mask]
        graph_pool_idx = 'output_mask'
    else:
        raise NotImplementedError
    return pool, graph_pool_idx


def inter_ensemble_pooling(embeddings: torch.Tensor, inter_pool: str):
    n_ensemble, n_entries, n_features = embeddings.shape
    if inter_pool == 'mean':
        embeddings = torch.mean(embeddings, dim=0)
    elif inter_pool == 'max':
        embeddings = torch.max(embeddings, dim=0).values
    elif inter_pool == 'cat':
        embeddings = embeddings.permute(1, 0, 2).reshape(n_entries, n_ensemble * n_features)
    else:
        raise NotImplementedError
    return embeddings


def jumping_knowledge(embeddings: List[torch.Tensor], jk: str):
    if jk is None:
        embedding = embeddings[-1]
    elif jk == 'cat':
        embedding = torch.cat(embeddings, dim=1)
    elif jk == 'mean':
        try:
            embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        except RuntimeError:  # in case shape error
            embedding = embeddings[-1]
    elif jk == 'max':
        try:
            embedding = torch.stack(embeddings, dim=0).max(dim=0).values
        except RuntimeError:  # in case shape error
            embedding = embeddings[-1]
    else:
        raise NotImplementedError
    return embedding
