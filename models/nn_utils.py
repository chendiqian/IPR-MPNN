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
