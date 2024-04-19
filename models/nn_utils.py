from typing import List, Optional, Union, Tuple

import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import degree


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
    elif graph_pooling == 'edge':
        # pool = lambda x, edge_label_index: (x[edge_label_index[0]] * x[edge_label_index[1]]).sum(1, keepdims=True)
        pool = lambda x, *args: x
        graph_pool_idx = 'edge_label_index'
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
    elif jk == 'identity':
        return embeddings
    else:
        raise NotImplementedError
    return embedding


def add_self_loop_multi_target(edge_index: torch.Tensor,
                               num_nodes: int,
                               dim: int,
                               edge_attr: Optional[torch.Tensor] = None,
                               edge_weight: Optional[torch.Tensor] = None):
    self_loops = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    self_loops = self_loops[None].repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    if edge_attr is not None:
        _shape = torch.tensor(edge_attr.shape).cpu().tolist()
        _shape[dim] = num_nodes
        padding = edge_attr.new_zeros(_shape)
        edge_attr = torch.cat([edge_attr, padding], dim=dim)

    if edge_weight is not None:
        _shape = torch.tensor(edge_weight.shape).cpu().tolist()
        _shape[dim] = num_nodes
        padding = edge_weight.new_ones(_shape)
        edge_weight = torch.cat([edge_weight, padding], dim=dim)

    return edge_index, edge_attr, edge_weight


def compute_gcn_norm(edge_index: torch.Tensor, num_nodes: Union[int, Tuple]):
    if isinstance(num_nodes, int):
        num_nodes = (num_nodes, num_nodes)

    row, col = edge_index
    deg_src = degree(row, num_nodes[0], dtype=torch.float)
    deg_src_inv_sqrt = deg_src.pow(-0.5)
    deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
    deg_dst = degree(col, num_nodes[1], dtype=torch.float)
    deg_dst_inv_sqrt = deg_dst.pow(-0.5)
    deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
    norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]
    return norm
