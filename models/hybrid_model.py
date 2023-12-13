from functools import partial
from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import HeteroData
# from torch_geometric.utils import to_undirected
# from torch_scatter import scatter_sum

from data.utils import Config
from models.auxloss import get_auxloss
from models.nn_utils import get_graph_pooling
from samplers.gumbel_scheme import GumbelSampler
from samplers.imle_scheme import IMLESampler
from samplers.simple_scheme import SIMPLESampler


class HybridModel(torch.nn.Module):
    def __init__(self,
                 scorer_model: torch.nn.Module,
                 base2centroid_model: torch.nn.Module,
                 sampler: Union[IMLESampler, SIMPLESampler, GumbelSampler],
                 hetero_gnn: torch.nn.Module,

                 target: str,
                 intra_pred_head: torch.nn.Module,
                 inter_pred_head: torch.nn.Module,
                 intra_graph_pool: str,
                 inter_ensemble_pool: str,
                 auxloss_dict: Union[Config, ConfigDict],
                 ):
        super(HybridModel, self).__init__()

        self.scorer_model = scorer_model
        self.base2centroid_model = base2centroid_model
        self.sampler = sampler
        self.hetero_gnn = hetero_gnn

        self.target = target
        self.inter_ensemble_pool = inter_ensemble_pool
        pool, graph_pool_idx = get_graph_pooling(intra_graph_pool)
        self.intra_graph_pool = pool
        self.graph_pool_idx = graph_pool_idx
        self.inter_pred_head = inter_pred_head
        self.intra_pred_head = intra_pred_head

        self.auxloss = partial(get_auxloss,
                               auxloss_dict=auxloss_dict) if auxloss_dict is not None else None

    def forward(self, data):
        device = data.x.device

        cumsum_nnodes = data._slice_dict['x']
        nnodes_list = cumsum_nnodes[1:] - cumsum_nnodes[:-1]
        if isinstance(self.sampler, IMLESampler):
            self.sampler.nnodes_list = nnodes_list.tolist()
        nnodes_list = nnodes_list.to(device)

        # get scores and samples
        scores = self.scorer_model(data)
        node_mask, marginal = self.sampler(scores) if self.training else self.sampler.validation(scores)
        n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape
        plot_node_mask = node_mask.detach().cpu().numpy()
        plot_scores = scores.detach().cpu().numpy()
        repeats = n_samples * n_ensemble

        # add a dimension for multiply broadcasting
        node_mask = node_mask.permute(0, 3, 2, 1).reshape(repeats, n_centroids, nnodes)[..., None]
        edge_mask = node_mask[:, :, data.edge_index[0], :] * node_mask[:, :, data.edge_index[1], :]

        # map clustered nodes into each centroid
        # repeats, n_centroids, n_graphs, features
        centroid_x = self.base2centroid_model(
            data,
            node_mask,
            edge_mask,
            (n_centroids, data.num_graphs, repeats)
        )

        # construct a heterogeneous hierarchical graph
        # data is constructs like
        # repeat1: [g1: (to_centroid1, to_centroid2, ...), g2: (to_centroid1, to_centroid2, ...)],
        # repeat2: [g1: (to_centroid1, to_centroid2, ...), g2: (to_centroid1, to_centroid2, ...)]

        # low to high hierarchy
        src = torch.arange(data.num_nodes * repeats, device=device).repeat_interleave(n_centroids)
        dst = torch.arange(repeats * data.num_graphs, device=device).repeat_interleave(
            nnodes_list.repeat(repeats)) * n_centroids
        dst = dst[None] + torch.arange(n_centroids, device=device, dtype=torch.long)[:, None]
        dst = dst.t().reshape(-1)

        # edges intra super nodes, assume graphs are undirected
        # todo: try this when converged
        # todo: UnsafeViewBackward0, probably comes from to_undirected
        # idx = np.hstack([np.vstack(np.triu_indices(n_centroids, k=1)),
        #                  np.vstack(np.diag_indices(n_centroids))])

        # cumsum_nedges = data._slice_dict['edge_index'].to(device)
        # nedges_list = cumsum_nedges[1:] - cumsum_nedges[:-1]
        # intra_num_edges = scatter_sum((node_mask[:, idx[0], :, 0][:, :, data.edge_index[0]] *
        #                          node_mask[:, idx[1], :, 0][:, :, data.edge_index[1]]),
        #                         torch.arange(data.num_graphs, device=device).repeat_interleave(nedges_list), dim=2)
        # intra_num_edges[:, -n_centroids:, :] = intra_num_edges[:, -n_centroids:, :] / 2
        #
        # idx, intra_edge_weights = to_undirected(torch.from_numpy(idx).to(device),
        #                                         intra_num_edges.permute(1, 0, 2),
        #                                         reduce='mean')
        # # after: repeats x n_centroids^2 x n_graphs
        # intra_edge_weights = intra_edge_weights.permute(1, 0, 2) / (intra_edge_weights.detach().abs().max() + 1.e-7)

        # dumb edge index
        idx = np.hstack([np.vstack(np.triu_indices(n_centroids, k=1)),
                         np.vstack(np.tril_indices(n_centroids, k=-1))])
        idx = torch.from_numpy(idx).to(device)

        new_data = HeteroData(
            base={'x': data.x,   # will be later filled
                  'batch': data.batch.repeat(repeats) +
                           torch.arange(repeats, device=device).repeat_interleave(nnodes) * data.num_graphs},
            centroid={'x': centroid_x,
                      'batch': torch.arange(repeats * data.num_graphs, device=device).repeat_interleave(n_centroids)},

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
                'edge_weight': node_mask.squeeze(-1).permute(0, 2, 1).reshape(-1)
            },
            centroid__to__base={
                'edge_index': torch.vstack([dst, src]),
                'edge_attr': None,
                'edge_weight': node_mask.squeeze(-1).permute(0, 2, 1).reshape(-1)
            },
            # centroid__to__centroid={
            # 'edge_index': idx.repeat(1, data.num_graphs * repeats) +
            # (torch.arange(data.num_graphs * repeats, device=device) * n_centroids).repeat_interleave(idx.shape[1]),
            # 'edge_attr': None,
            # 'edge_weight': intra_edge_weights.permute(0, 2, 1).reshape(-1)
            # }
            centroid__to__centroid={
                'edge_index': idx.repeat(1, data.num_graphs * repeats) +
                              (torch.arange(data.num_graphs * repeats, device=device) * n_centroids).repeat_interleave(idx.shape[1]),
                'edge_attr': None,
                'edge_weight': None
            }
        )

        base_embedding, centroid_embedding = self.hetero_gnn(
            data,
            new_data,
            hasattr(data, 'edge_attr') and data.edge_attr is not None)

        if self.target == 'base':
            node_embedding = base_embedding.reshape(repeats, -1, base_embedding.shape[-1])
            if self.inter_ensemble_pool == 'mean':
                node_embedding = torch.mean(node_embedding, dim=0)
            elif self.inter_ensemble_pool == 'max':
                node_embedding = torch.max(node_embedding, dim=0).values
            else:
                raise NotImplementedError
            node_embedding = self.inter_pred_head(node_embedding)
            graph_embedding = self.intra_graph_pool(node_embedding, getattr(data, self.graph_pool_idx))
            graph_embedding = self.intra_pred_head(graph_embedding)
        else:
            raise NotImplementedError

        # get auxloss
        if self.training and self.auxloss is not None:
            auxloss = self.auxloss(pool=self.intra_graph_pool,
                                   graph_pool_idx=self.graph_pool_idx,
                                   scores=scores, data=data)
        else:
            auxloss = 0.

        return graph_embedding, plot_node_mask, plot_scores, auxloss
