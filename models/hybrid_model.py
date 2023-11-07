from typing import Union

import torch
import numpy as np
from torch_scatter import scatter_sum
from torch_geometric.utils import to_undirected

from samplers.gumbel_scheme import GumbelSampler
from samplers.imle_scheme import IMLESampler
from samplers.simple_scheme import SIMPLESampler
from torch_geometric.data import HeteroData


class HybridModel(torch.nn.Module):
    def __init__(self,
                 atom_encoder: torch.nn.Module,
                 bond_encoder: torch.nn.Module,
                 scorer_model: torch.nn.Module,
                 base2centroid_model: torch.nn.Module,
                 sampler: Union[IMLESampler, SIMPLESampler, GumbelSampler],
                 # hier_mpnn: torch.nn.Module,
                 ):
        super(HybridModel, self).__init__()

        self.atom_encoder = atom_encoder
        self.edge_encoder = bond_encoder
        self.scorer_model = scorer_model
        self.base2centroid_model = base2centroid_model
        self.sampler = sampler

    def forward(self, data):
        x = self.atom_encoder(data)
        device = x.device

        # get scores and samples
        scores = self.scorer_model(x, data.edge_index, data.edge_attr)
        node_mask, marginal = self.sampler(scores) if self.training else self.sampler.validation(scores)
        n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape
        repeats = n_samples * n_ensemble

        # add a dimension for multiply broadcasting
        node_mask = node_mask.permute(0, 3, 2, 1).reshape(repeats, n_centroids, nnodes)[..., None]
        edge_mask = node_mask[:, :, data.edge_index[0], :] * node_mask[:, :, data.edge_index[1], :]

        # map clustered nodes into each centroid
        if self.training:
            x = x.repeat(repeats, n_centroids, 1, 1)  # repeats, n_centroids, nnodes, features
            x = x.reshape(-1, x.shape[-1])
            num_graphs = data.num_graphs
            batch = data.batch.repeat(repeats * n_centroids) + \
                    torch.arange(repeats * n_centroids, device=device).repeat_interleave(nnodes) * num_graphs

            # repeats, n_centroids, n_graphs, features
            centroid_x = self.base2centroid_model(x, batch, data.edge_index, data.edge_attr, node_mask, edge_mask)
        else:
            raise NotImplementedError

        # construct a heterogeneous hierarchical graph
        # data is constructs like
        # repeat1: [to_cent1: (g1, g2, g3, g4), to_cent2: (g1, g2, g3, g4), to_cent3: (g1, g2, g3, g4)],
        # repeat2: [to_cent1: (g1, g2, g3, g4), to_cent2: (g1, g2, g3, g4), to_cent3: (g1, g2, g3, g4)]
        cumsum_nnodes = data._slice_dict['x'].to(device)
        nnodes_list = cumsum_nnodes[1:] - cumsum_nnodes[:-1]

        # low to high hierarchy
        src = torch.arange(data.num_nodes * repeats, device=device).repeat_interleave(n_centroids)
        dst = torch.arange(repeats * data.num_graphs, device=device).repeat_interleave(
            nnodes_list.repeat(repeats)) * n_centroids
        dst = dst[None] + torch.arange(n_centroids, device=device, dtype=torch.long)[:, None]
        dst = dst.t().reshape(-1)

        # edges intra super nodes, assume graphs are undirected
        idx = np.hstack([np.vstack(np.triu_indices(n_centroids, k=1)),
                         np.vstack(np.diag_indices(n_centroids))])

        cumsum_nedges = data._slice_dict['edge_index'].to(device)
        nedges_list = cumsum_nedges[1:] - cumsum_nedges[:-1]
        intra_num_edges = scatter_sum((node_mask[:, idx[0], :, 0][:, :, data.edge_index[0]] *
                                 node_mask[:, idx[1], :, 0][:, :, data.edge_index[1]]),
                                torch.arange(data.num_graphs, device=device).repeat_interleave(nedges_list), dim=2)
        intra_num_edges[:, -n_centroids:, :] = intra_num_edges[:, -n_centroids:, :] / 2

        idx, intra_edge_weights = to_undirected(torch.from_numpy(idx).to(device), intra_num_edges.permute(1, 0, 2))
        intra_edge_weights = intra_edge_weights.permute(1, 0, 2)


        new_data = HeteroData(
            base={'x': x.repeat(repeats, 1)},
            centroid={'x': centroid_x},

            base__to__base={'edge_index': data.edge_index.repeat(1, repeats) + \
                                          torch.arange(repeats, device=device).repeat_interleave(data.num_edges) * data.num_nodes,
                            'edge_attr': data.edge_attr.repeat(repeats, 1),
                            'edge_weight': None},
            base__to__centroid={'edge_index': torch.vstack([src, dst]),
                                'edge_attr': None,
                                'edge_weight': node_mask.squeeze(-1).permute(0, 2, 1).reshape(-1)},
            centroid__to__base={'edge_index': torch.vstack([dst, src]),
                                'edge_attr': None,
                                'edge_weight': node_mask.squeeze(-1).permute(0, 2, 1).reshape(-1)},
            centroid__to__centroid={'edge_index': idx.repeat(1, data.num_graphs * repeats) + \
                                                  (torch.arange(data.num_graphs * repeats,
                                                                device=device) * n_centroids).repeat_interleave(idx.shape[1]),
                                    'edge_attr': None,
                                    'edge_weight': intra_edge_weights.permute(0, 2, 1).reshape(-1)}
        )
