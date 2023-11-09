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
        if edge_attr is not None:
            edge_embedding = self.bond_encoder(edge_attr)
            edge_embedding = edge_embedding.repeat(1, 1, 1, 1)  # 1, 1, nedges, features
        else:
            edge_embedding = 0.

        if edge_weight is not None:
            repeats, choices, _, _ = edge_weight.shape
            message = F.gelu(x.reshape(repeats, choices, -1, x.shape[-1])[:, :, edge_index[0], :] + edge_embedding)
            message = message * edge_weight
            out = scatter_sum(message, edge_index[1], dim=2)
            out = out.reshape(x.shape)
        else:
            raise NotImplementedError

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

        x = self.mlp(x)

        # pooling
        if self.training:
            repeats, n_centroids, nnodes, _ = node_mask.shape
            x = x.reshape(repeats, n_centroids, nnodes, -1)
            batch = batch[:nnodes]  # it has been repeated
            x = scatter_sum(x * node_mask, batch, dim=2) / scatter_sum(node_mask.detach(), batch, dim=2)
            x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1])
        else:
            raise NotImplementedError
        return x
