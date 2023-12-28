import inspect

import torch
import torch.nn.functional as F
from torch.nn import Identity
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.utils import subgraph
from torch_scatter import scatter_sum, scatter_mean


class GINEConvMultiEdgeset(torch.nn.Module):
    def __init__(self, mlp, bond_encoder):
        super(GINEConvMultiEdgeset, self).__init__()

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))
        self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # 1, 1, nedges, features
        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else torch.zeros(1, 1, device=x.device,
                                                                                                dtype=x.dtype)

        if self.training:
            edge_embedding = edge_embedding.repeat(1, 1, 1, 1)
            repeats, choices, _, _ = edge_weight.shape
            unflatten_x = x.reshape(repeats, choices, -1, x.shape[-1])
            message = F.gelu(unflatten_x[:, :, edge_index[0], :] + edge_embedding)
            message = message * edge_weight
            out = scatter_sum(message, edge_index[1], dim=2, dim_size=unflatten_x.shape[2])
            out = out.reshape(x.shape)
        else:
            # cannot reshape x like that, as the number of edges per centroid / ensemble graph may vary
            message = F.gelu(x[edge_index[0], :] + edge_embedding)
            message = message * edge_weight
            out = scatter_sum(message, edge_index[1], dim=0, dim_size=x.shape[0])

        out = self.mlp((1 + self.eps) * x + out)
        return out


class SAGEConvMultiEdgeset(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bond_encoder):
        super(SAGEConvMultiEdgeset, self).__init__()

        self.lin = torch.nn.Linear(in_channel, in_channel, bias=True)
        self.lin_l = torch.nn.Linear(in_channel, out_channel, bias=True)
        # they use false
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/sage_conv.html#SAGEConv
        self.lin_r = torch.nn.Linear(in_channel, out_channel, bias=False)
        self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # 1, 1, nedges, features
        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else torch.zeros(1, 1, device=x.device,
                                                                                                dtype=x.dtype)

        x_l, x_r = (self.lin(x), x)

        if self.training:
            edge_embedding = edge_embedding.repeat(1, 1, 1, 1)
            repeats, choices, _, _ = edge_weight.shape
            unflatten_x = x_l.reshape(repeats, choices, -1, x_l.shape[-1])
            message = F.gelu(unflatten_x[:, :, edge_index[0], :] + edge_embedding)
            message = message * edge_weight
            # sage use mean aggr by default
            out = scatter_mean(message, edge_index[1], dim=2, dim_size=unflatten_x.shape[2])
            out = out.reshape(x_l.shape)
        else:
            # cannot reshape x like that, as the number of edges per centroid / ensemble graph may vary
            message = F.gelu(x_l[edge_index[0], :] + edge_embedding)
            message = message * edge_weight
            out = scatter_mean(message, edge_index[1], dim=0, dim_size=x_l.shape[0])

        out = self.lin_l(out) + self.lin_r(x_r)
        return out


class GNNMultiEdgeset(torch.nn.Module):
    def __init__(self,
                 conv,
                 atom_encoder_handler,
                 bond_encoder_handler,
                 hidden,
                 num_conv_layers,
                 num_mlp_layers,
                 out_feature,
                 norm,
                 activation,
                 dropout):
        super(GNNMultiEdgeset, self).__init__()

        self.atom_encoder = atom_encoder_handler()
        self.dropout = dropout

        dims = [hidden] * (num_conv_layers + 1)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            # gin also in this case, edge_attr can be None
            if conv in ['gin', 'gine']:
                self.convs.append(
                    GINEConvMultiEdgeset(
                        torch.nn.Sequential(
                            torch.nn.Linear(dim_in, dim_out),
                            torch.nn.GELU(),
                            torch.nn.Linear(dim_out, dim_out),
                        ),
                        bond_encoder_handler(),
                    )
                )
            elif conv == 'sage':
                self.convs.append(
                    SAGEConvMultiEdgeset(dim_in, dim_out, bond_encoder_handler())
                )
            else:
                raise NotImplementedError
            self.norms.append(normalization_resolver(norm, dim_out) if norm is not None else Identity())

        self.supports_norm_batch = False
        if len(self.norms) > 0 and hasattr(self.norms[0], 'forward'):
            norm_params = inspect.signature(self.norms[0].forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        self.mlp = MLP(in_channels=hidden,
                       hidden_channels=hidden,
                       out_channels=out_feature,
                       num_layers=num_mlp_layers,
                       act=activation,
                       norm=norm)

    def forward(self, data, node_mask):
        device = data.x.device
        n_graphs = data.num_graphs
        n_samples, sum_n_centroids, nnodes, _ = node_mask.shape
        batch, edge_index, edge_attr = data.batch, data.edge_index, data.edge_attr
        x = self.atom_encoder(data)

        edge_mask = node_mask[:, :, edge_index[0], :] * node_mask[:, :, edge_index[1], :]

        batch = batch.repeat(n_samples * sum_n_centroids) + \
                torch.arange(n_samples * sum_n_centroids, device=device).repeat_interleave(
                    data.num_nodes) * n_graphs

        if self.training:
            x = x.repeat(n_samples, sum_n_centroids, 1, 1).reshape(-1, x.shape[-1])
        else:
            edge_index = edge_index.repeat(1, n_samples * sum_n_centroids) + \
                         torch.arange(n_samples * sum_n_centroids, device=device).repeat_interleave(
                             data.num_edges) * data.num_nodes
            if edge_attr is not None:
                nnz_edge_idx = edge_mask.squeeze(-1).nonzero()[..., -1]
                edge_attr = edge_attr[nnz_edge_idx]
            else:
                edge_attr = None
            nnz_edge_idx = edge_mask.reshape(-1).nonzero().squeeze()
            edge_index = edge_index[:, nnz_edge_idx]
            edge_mask = edge_mask.reshape(-1)[nnz_edge_idx][..., None]

            # sparsify the nodes
            nnz_x_idx = node_mask.squeeze(-1).nonzero()[..., -1]
            x = x[nnz_x_idx, :]

            nnz_x_idx = node_mask.reshape(-1).nonzero().squeeze()
            node_mask = node_mask.reshape(-1)[nnz_x_idx][..., None]
            batch = batch[nnz_x_idx]

            edge_index, edge_attr, subg_edge_mask = subgraph(nnz_x_idx,
                                                             edge_index,
                                                             edge_attr,
                                                             relabel_nodes=True,
                                                             num_nodes=data.num_nodes * n_samples * sum_n_centroids,
                                                             return_edge_mask=True)

            edge_mask = edge_mask[subg_edge_mask]

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr, edge_mask)
            if self.supports_norm_batch:
                x_new = self.norms[i](x_new, batch)
            else:
                x_new = self.norms[i](x_new)
            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new

        x = self.mlp(x, batch)

        # pooling
        if self.training:
            single_batch = batch[:nnodes]  # it has been repeated
            x = x.reshape(n_samples, sum_n_centroids, nnodes, -1)
            x = scatter_sum(x * node_mask, single_batch, dim=2) / \
                (scatter_sum(node_mask.detach(), single_batch, dim=2) + 1.e-7)
        else:
            # (n_samples * sum_n_centroids * n_graphs) * F
            x = scatter_sum(x * node_mask, batch, dim=0, dim_size=n_samples * n_graphs * sum_n_centroids) / \
                (scatter_sum(node_mask.detach(), batch, dim=0, dim_size=n_samples * n_graphs * sum_n_centroids) + 1.e-7)
            x = x.reshape(n_samples, sum_n_centroids, n_graphs, x.shape[-1])
        return x
