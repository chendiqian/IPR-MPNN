import inspect

import torch
import torch.nn.functional as F
from torch.nn import Identity
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import normalization_resolver
from torch_scatter import scatter_sum


class GINEConvMultiEdgeset(torch.nn.Module):
    def __init__(self, mlp, bond_encoder):
        super(GINEConvMultiEdgeset, self).__init__()

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))
        self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # (repeats * choices * nnodes), features = x.shape
        # repeats, choices, nedges, 1 = edge_weight.shape

        # 1, 1, nedges, features
        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else torch.zeros(1, 1, device=x.device, dtype=x.dtype)

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


class GNNMultiEdgeset(torch.nn.Module):
    def __init__(self,
                 conv,
                 edge_encoder,
                 hidden,
                 num_conv_layers,
                 num_mlp_layers,
                 out_feature,
                 norm,
                 activation,
                 dropout):
        super(GNNMultiEdgeset, self).__init__()

        assert conv == 'gine'
        self.dropout = dropout

        dims = [hidden] * (num_conv_layers + 1)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.convs.append(
                GINEConvMultiEdgeset(
                    torch.nn.Sequential(
                        torch.nn.Linear(dim_in, dim_out),
                        torch.nn.GELU(),
                        torch.nn.Linear(dim_out, dim_out),
                    ),
                    edge_encoder,
                )
            )
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

    def forward(self, x, batch, edge_index, edge_attr, node_mask, edge_mask):
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
        repeats, n_centroids, nnodes, _ = node_mask.shape
        single_batch = batch[:nnodes]  # it has been repeated
        if self.training:
            x = x.reshape(repeats, n_centroids, nnodes, -1)
            x = scatter_sum(x * node_mask, single_batch, dim=2) / \
                (scatter_sum(node_mask.detach(), single_batch, dim=2) + 1.e-7)
            x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1])
        else:
            n_graphs = single_batch.max() + 1
            nnz_x_idx = node_mask.reshape(-1).nonzero().squeeze()
            nnz_node_mask = node_mask.reshape(-1)[nnz_x_idx][..., None]
            x = x[nnz_x_idx, :]
            batch = batch[nnz_x_idx]
            # (repeats * n_centroids * n_graphs) * F
            x = scatter_sum(x * nnz_node_mask, batch, dim=0, dim_size=n_centroids * n_graphs * repeats) / \
                (scatter_sum(nnz_node_mask.detach(), batch, dim=0, dim_size=n_centroids * n_graphs * repeats) + 1.e-7)
            x = x.reshape(repeats, n_centroids, n_graphs, x.shape[-1])
            x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1])
        return x
