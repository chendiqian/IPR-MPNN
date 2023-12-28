from functools import partial
from typing import Union, List

import numpy as np
import torch
from ml_collections import ConfigDict
from torch_geometric.data import HeteroData

from data.utils import Config
from models.auxloss import get_auxloss
from models.nn_utils import get_graph_pooling
from samplers.gumbel_scheme import GumbelSampler
from samplers.imle_scheme import IMLESampler
from samplers.simple_scheme import SIMPLESampler

LARGE_NUMBER = 1.e10


class HybridModel(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 scorer_model: torch.nn.Module,
                 list_num_centroids: List,
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
        self.list_num_centroids = list_num_centroids
        self.tensor_num_centroids = torch.tensor(list_num_centroids, dtype=torch.long, device=device)
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
                               list_num_centroids=list_num_centroids,
                               auxloss_dict=auxloss_dict) if auxloss_dict is not None else None

    def forward(self, data, for_plots_only=False):
        device = data.x.device
        plot_scores, plot_node_mask = None, None

        cumsum_nnodes = data._slice_dict['x']
        nnodes_list = (cumsum_nnodes[1:] - cumsum_nnodes[:-1]).to(device)
        n_graphs = data.num_graphs

        # get scores and samples
        scores = self.scorer_model(data)
        nnodes, max_n_centroids, n_ensemble = scores.shape
        if for_plots_only:
            plot_scores = scores.detach().clone().cpu().numpy()

        # this is a mask indexing which centroids are kept in each ensemble
        ens_idx_mask = torch.zeros(n_ensemble, max_n_centroids, dtype=torch.bool, device=device)
        for i, n_ct in enumerate(self.list_num_centroids):
            ens_idx_mask[i, :n_ct] = True

        # for sampling, if there are less n_centroids than max possible num, then pad with a bias
        scores[:, ~ens_idx_mask.t()] = scores[:, ~ens_idx_mask.t()] - LARGE_NUMBER

        # k-subset sampling is carried out as usual, in parallel
        node_mask, _ = self.sampler(scores) if self.training else self.sampler.validation(scores)
        n_samples, nnodes, max_n_centroids, n_ensemble = node_mask.shape
        if for_plots_only:
            plot_node_mask = node_mask.detach().cpu().numpy()
            return None, plot_node_mask, plot_scores, None

        # slice and drop the unnecessary columns
        # nodemask: ns x nn x ncent x ens -> ns x nn x ens x ncent
        # mask: ns x nn x ens x ncent
        node_mask = node_mask.permute(0, 1, 3, 2)[ens_idx_mask.repeat(n_samples, nnodes, 1, 1)].reshape(n_samples, nnodes, -1)
        n_samples, nnodes, sum_n_centroids = node_mask.shape
        repeats = n_samples * n_ensemble

        # map clustered nodes into each centroid
        # node_mask: nx, sum_cents, nnodes, 1
        # add a dimension for multiply broadcasting
        # centx: n_samples, sum_n_centroids, n_graphs, features
        centroid_x = self.base2centroid_model(data, node_mask.permute(0, 2, 1)[..., None])

        # construct a heterogeneous hierarchical graph
        # data is constructs like
        # repeat1: [g1 to_centroid1_1, g2 to_centroid2_1, g1 to_centroid1_2, g2 to_centroid2_2 ...],
        # repeat2: [...]

        # low to high hierarchy edge index
        src = torch.arange(nnodes * repeats, device=device).reshape(-1, nnodes).\
            repeat_interleave(self.tensor_num_centroids.repeat(n_samples), dim=0).reshape(-1)

        dst = torch.cat([torch.arange(n_graphs * nct, device=device).reshape(n_graphs, nct)
                         for nct in self.list_num_centroids], dim=1)
        bias = (torch.hstack([src.new_zeros(1),
                              torch.cumsum(self.tensor_num_centroids, dim=0)[:-1]]) * n_graphs).\
            repeat_interleave(self.tensor_num_centroids)
        dst = dst + bias[None]
        dst_backup = dst  # for the batch idx of the centroids
        dst = (dst.t().reshape(-1)).repeat_interleave(nnodes_list.repeat(sum_n_centroids))
        dst = (dst[None] + torch.arange(n_samples, device=device)[:, None] * sum_n_centroids * n_graphs).reshape(-1)

        # dumb edge index intra centroids: fully connected, no self loop, undirected
        c2c_idx = np.hstack([np.tile(np.vstack(np.triu_indices(n_ct, k=1)), n_graphs) for n_ct in self.list_num_centroids])
        bias = np.concatenate([np.zeros(1, dtype=np.int64), np.repeat(self.list_num_centroids, n_graphs)[:-1]]).cumsum()
        bias = bias.repeat((np.array(self.list_num_centroids) * (np.array(self.list_num_centroids) - 1) // 2).repeat(n_graphs))
        c2c_idx = c2c_idx + bias
        c2c_idx = np.hstack([c2c_idx, c2c_idx[np.array([1, 0])]])
        c2c_idx = torch.from_numpy(c2c_idx).to(device)
        c2c_idx = c2c_idx.repeat(1, n_samples) +\
                  (torch.arange(n_samples, device=device) * sum_n_centroids * n_graphs).repeat_interleave(c2c_idx.shape[1])

        # create batch idx
        original_x_batch = data.batch.repeat(repeats) +\
                           torch.arange(repeats, device=device).repeat_interleave(nnodes) * n_graphs

        # centroids in the same n_sample are considered as one batch, regardless of ensemble
        # otherwise the norm might go wrong, because some n_centroids are 1
        centroid_x_batch = torch.empty(sum_n_centroids * n_graphs, device=device, dtype=torch.long)
        centroid_x_batch[dst_backup.reshape(-1)] = torch.arange(n_graphs, device=device).repeat_interleave(sum_n_centroids)
        centroid_x_batch = centroid_x_batch.repeat(n_samples) + \
                           n_graphs * torch.arange(n_samples, device=device).repeat_interleave(n_graphs * sum_n_centroids)

        new_data = HeteroData(
            base={'x': data.x,   # will be later filled
                  'batch': original_x_batch},
            centroid={'x': centroid_x.reshape(-1, centroid_x.shape[-1]),
                      'batch': centroid_x_batch},

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
                'edge_index': c2c_idx,
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
