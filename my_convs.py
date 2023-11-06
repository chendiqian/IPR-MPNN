import inspect
from typing import Optional, Union

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch.nn import Identity
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import normalization_resolver
from torch_scatter import scatter_sum

from nn_utils import residual


class GINEConvMultiEdgeset(torch.nn.Module):
    def __init__(self, emb_dim: int,
                 mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 bond_encoder: Optional[Union[MLP, torch.nn.Sequential]] = None):

        super(GINEConvMultiEdgeset, self).__init__()

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

        if bond_encoder is None:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        else:
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


class GINEMultiEdgeset(torch.nn.Module):
    def __init__(self,
                 in_features,
                 num_layers,
                 num_mlp_layers,
                 hidden,
                 out_feature,
                 norm,
                 dropout,
                 use_residual,
                 edge_encoder=None):
        super(GINEMultiEdgeset, self).__init__()

        self.dropout = dropout
        self.use_residual = use_residual

        dims = [in_features] + [hidden] * num_layers

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            self.convs.append(
                GINEConvMultiEdgeset(
                    dim_in,
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
                       norm=norm)

    def forward(self, x, batch, edge_index, edge_attr, node_mask=None):
        repeats, choices, nnodes, features = x.shape
        x = x.reshape(-1, features)
        # x: repeats, choices, nnodes, features
        # node_mask: repeats, choices, nnodes, 1
        edge_mask = node_mask[:, :, edge_index[0], :] * node_mask[:, :, edge_index[1], :] if node_mask is not None else None
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr, edge_mask)
            if self.supports_norm_batch:
                x_new = self.norms[i](x_new, batch)
            else:
                x_new = self.norms[i](x_new)
            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = residual(x, x_new)
            else:
                x = x_new

        x = self.mlp(x)
        if node_mask is not None:
            x = x.reshape(repeats, choices, nnodes, -1)
            x = (x * node_mask).sum(2) / node_mask.detach().sum(2)
        else:
            raise NotImplementedError
        return x
