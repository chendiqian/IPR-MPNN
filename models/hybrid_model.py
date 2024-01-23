from functools import partial
from typing import Union, List

import numpy as np
import torch
from torch_geometric.data import HeteroData

from data.utils import Config
from models.auxloss import get_auxloss
from models.nn_utils import get_graph_pooling, inter_ensemble_pooling, jumping_knowledge
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

                 jk: str,
                 target: str,
                 intra_pred_head: torch.nn.Module,
                 inter_base_pred_head: torch.nn.Module,
                 inter_cent_pred_head: torch.nn.Module,
                 intra_graph_pool: str,
                 inter_ensemble_pool: str,
                 auxloss_dict: Config,
                 ):
        super(HybridModel, self).__init__()

        self.scorer_model = scorer_model
        self.list_num_centroids = list_num_centroids
        self.tensor_num_centroids = torch.tensor(list_num_centroids, dtype=torch.long, device=device)
        self.base2centroid_model = base2centroid_model
        self.sampler = sampler
        self.hetero_gnn = hetero_gnn

        self.jk = partial(jumping_knowledge, jk=jk)
        self.target = target
        self.inter_ensemble_pool = partial(inter_ensemble_pooling, inter_pool=inter_ensemble_pool)
        if intra_graph_pool == 'root':
            assert target == 'base', "Unable to use centroids"
        pool, graph_pool_idx = get_graph_pooling(intra_graph_pool)
        self.intra_graph_pool = pool
        self.graph_pool_idx = graph_pool_idx
        self.inter_base_pred_head = inter_base_pred_head
        self.inter_cent_pred_head = inter_cent_pred_head
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

        if min(self.list_num_centroids) < max_n_centroids:
            # this is a mask indexing which centroids are kept in each ensemble
            ens_idx_mask = torch.zeros(n_ensemble, max_n_centroids, dtype=torch.bool, device=device)
            for i, n_ct in enumerate(self.list_num_centroids):
                ens_idx_mask[i, :n_ct] = True

            # for sampling, if there are less n_centroids than max possible num, then pad with a bias
            scores[:, ~ens_idx_mask.t()] = scores[:, ~ens_idx_mask.t()] - LARGE_NUMBER

        # k-subset sampling is carried out as usual, in parallel
        node_mask, _ = self.sampler(scores) if self.training else self.sampler.validation(scores)
        if for_plots_only:
            plot_node_mask = node_mask.detach().cpu().numpy()
            return None, plot_node_mask, plot_scores, None

        # when combination with various cluster numbers, some clusters have 0s embedding
        n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape
        repeats = n_samples * n_ensemble

        node_mask = node_mask.permute(0, 3, 2, 1).reshape(repeats, n_centroids, nnodes)

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

        base_embeddings, centroid_embeddings = self.hetero_gnn(
            data,
            new_data,
            hasattr(data, 'edge_attr') and data.edge_attr is not None)

        base_embedding = self.jk(base_embeddings)
        centroid_embedding = self.jk(centroid_embeddings)

        graph_embeddings = []

        if type(self.intra_pred_head) == torch.nn.modules.container.ModuleList: # Only works for centroid embedding right now
            if self.target != 'centroid':
                raise NotImplementedError
            
            preds_list = []
            
            for head, centroid_embedding in zip(self.intra_pred_head, centroid_embeddings):
                centroid_embedding = centroid_embedding.reshape(repeats, n_graphs * n_centroids, centroid_embedding.shape[-1])
                centroid_embedding = self.inter_ensemble_pool(centroid_embedding)
                centroid_embedding = self.intra_graph_pool(centroid_embedding, new_data['centroid']['batch'][:n_graphs * n_centroids])
                centroid_embedding = head(centroid_embedding)

                preds_list.append(centroid_embedding)

            graph_embedding = preds_list
            
        else:
            if self.target in ['base', 'both']:
                base_embedding = base_embedding.reshape(repeats, nnodes, base_embedding.shape[-1])
                base_embedding = self.inter_ensemble_pool(base_embedding)
                base_embedding = self.inter_base_pred_head(base_embedding)
                graph_embedding = self.intra_graph_pool(base_embedding, getattr(data, self.graph_pool_idx))
                graph_embeddings.append(graph_embedding)
            if self.target in ['centroid', 'both']:
                centroid_embedding = centroid_embedding.reshape(repeats, n_graphs * n_centroids, centroid_embedding.shape[-1])
                centroid_embedding = self.inter_ensemble_pool(centroid_embedding)
                centroid_embedding = self.inter_cent_pred_head(centroid_embedding)
                graph_embedding = self.intra_graph_pool(centroid_embedding, new_data['centroid']['batch'][:n_graphs * n_centroids])
                graph_embeddings.append(graph_embedding)

            graph_embedding = self.intra_pred_head(torch.cat(graph_embeddings, dim=1))

        # get auxloss
        if self.training and self.auxloss is not None:
            auxloss = self.auxloss(pool=self.intra_graph_pool,
                                graph_pool_idx=self.graph_pool_idx,
                                scores=scores, data=data)
        else:
            auxloss = 0.

        return graph_embedding, plot_node_mask, plot_scores, auxloss
