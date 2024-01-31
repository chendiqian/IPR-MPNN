from typing import Callable

import numpy as np
import torch
from torch_geometric.data import HeteroData


class HybridModel(torch.nn.Module):
    def __init__(self,
                 base2centroid_model: torch.nn.Module,
                 hetero_gnn: torch.nn.Module,

                 jk_func: Callable,
                 graph_pool_idx: str,
                 pred_head: torch.nn.Module):
        super(HybridModel, self).__init__()

        self.base2centroid_model = base2centroid_model
        self.hetero_gnn = hetero_gnn

        self.jk_func = jk_func
        self.graph_pool_idx = graph_pool_idx
        self.pred_head = pred_head

    def forward(self, data, for_plots_only=False):
        assert hasattr(data, 'partition') and data.partition is not None
        n_centroids = data.partition.max().item() + 1

        device = data.x.device

        cumsum_nnodes = data._slice_dict['x']
        nnodes_list = (cumsum_nnodes[1:] - cumsum_nnodes[:-1]).to(device)
        n_graphs = data.num_graphs
        nnodes = data.num_nodes
        repeats = 1

        node_mask = torch.zeros(nnodes, n_centroids, dtype=torch.float, device=device)
        node_mask[torch.arange(nnodes, device=device), data.partition] = 1.

        if for_plots_only:
            return None, node_mask[None, :, :, None], None, None

        node_mask = node_mask.t()[None]

        # map clustered nodes into each centroid
        # add a dimension for multiply broadcasting
        # centx: repeats, n_centroids, n_graphs, features
        centroid_x = self.base2centroid_model(data, node_mask[..., None])
        centroid_x = centroid_x.permute(0, 2, 1, 3).reshape(-1, centroid_x.shape[-1])

        # construct a heterogeneous hierarchical graph
        # low to high hierarchy
        src = torch.arange(data.num_nodes * repeats, device=device).repeat_interleave(n_centroids)
        dst = torch.arange(repeats * n_graphs, device=device).repeat_interleave(
            nnodes_list.repeat(repeats)) * n_centroids
        dst = dst[None] + torch.arange(n_centroids, device=device, dtype=torch.long)[:, None]
        dst = dst.t().reshape(-1)

        # edges intra super nodes, assume graphs are undirected
        # dumb edge index
        idx = np.hstack([np.vstack(np.triu_indices(n_centroids, k=1)),
                         np.vstack(np.tril_indices(n_centroids, k=-1))])
        idx = torch.from_numpy(idx).to(device)

        new_data = HeteroData(
            base={'x': data.x,  # will be later filled
                  'batch': data.batch.repeat(repeats) +
                           torch.arange(repeats, device=device).repeat_interleave(nnodes) * n_graphs},
            centroid={'x': centroid_x,
                      'batch': torch.arange(repeats * n_graphs, device=device).repeat_interleave(n_centroids)},

            base__to__base={
                'edge_index': data.edge_index.repeat(1, repeats) +
                              torch.arange(repeats, device=device).repeat_interleave(data.num_edges) * data.num_nodes,
                'edge_attr': (data.edge_attr.repeat(repeats) if data.edge_attr.dim() == 1 else
                              data.edge_attr.repeat(repeats, 1)) if data.edge_attr is not None else None,
                'edge_weight': None
            },
            base__to__centroid={
                'edge_index': torch.vstack([src, dst]),
                'edge_attr': None,
                'edge_weight': node_mask.permute(0, 2, 1).reshape(-1)
            },
            centroid__to__base={
                'edge_index': torch.vstack([dst, src]),
                'edge_attr': None,
                'edge_weight': node_mask.permute(0, 2, 1).reshape(-1)
            },
            centroid__to__centroid={
                'edge_index': idx.repeat(1, n_graphs * repeats) +
                              (torch.arange(n_graphs * repeats, device=device) * n_centroids).repeat_interleave(
                                  idx.shape[1]),
                'edge_attr': None,
                'edge_weight': None
            }
        )

        list_base_embeddings, list_centroid_embeddings = self.hetero_gnn(
            data,
            new_data,
            hasattr(data, 'edge_attr') and data.edge_attr is not None)

        base_embedding = self.jk_func(list_base_embeddings)
        centroid_embedding = self.jk_func(list_centroid_embeddings)

        if isinstance(self.pred_head, torch.nn.ModuleList):
            assert isinstance(base_embedding, list) and isinstance(centroid_embedding, list)
            graph_embedding = []
            for head, n_emb, c_emb in zip(self.pred_head, base_embedding, centroid_embedding):
                graph_embedding.append(
                    head(n_emb.reshape(repeats, nnodes, n_emb.shape[-1]),
                         c_emb.reshape(repeats, n_graphs * n_centroids, c_emb.shape[-1]),
                         getattr(data, self.graph_pool_idx),
                         new_data['centroid']['batch'][:n_graphs * n_centroids])
                )
        else:
            graph_embedding = self.pred_head(
                base_embedding.reshape(repeats, nnodes, base_embedding.shape[-1]),
                centroid_embedding.reshape(repeats, n_graphs * n_centroids, centroid_embedding.shape[-1]),
                getattr(data, self.graph_pool_idx),
                new_data['centroid']['batch'][:n_graphs * n_centroids]
            )

        return graph_embedding, None, None, 0.
