import inspect
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn import Sequential, GELU, Linear, Identity
from torch_geometric.nn import MLP, GINConv
from torch_geometric.nn.resolver import normalization_resolver

from models.my_conv import GCNConv, SAGEConv, GINEConv


class ScorerGNN(torch.nn.Module):
    def __init__(self,
                 conv: str,
                 conv_cache: bool,
                 atom_encoder_handler: Callable,
                 bond_encoder_handler: Callable,
                 in_feature: int,
                 hidden: int,
                 num_conv_layers: int,
                 num_mlp_layers: int,
                 max_num_centroids: int,
                 num_ensemble: int,
                 norm: str,
                 activation: str,
                 dropout: float
                 ):
        super(ScorerGNN, self).__init__()

        self.atom_encoder = atom_encoder_handler(True)
        self.max_num_centroids = max_num_centroids
        self.num_ensemble = num_ensemble

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = dropout

        in_dims = [in_feature] + [hidden] * max(num_conv_layers - 1, 0)
        for i in range(num_conv_layers):
            if conv == 'gin':
                self.convs.append(
                    GINConv(nn=Sequential(
                        Linear(in_dims[i], hidden),
                        GELU(),
                        Linear(hidden, hidden),
                    ), train_eps=True)
                )
            elif conv == 'gine':
                self.convs.append(
                    GINEConv(
                        bond_encoder_handler(),
                        nn=Sequential(
                        Linear(in_dims[i], hidden),
                        GELU(),
                        Linear(hidden, hidden)),
                        in_channels=in_dims[i])
                )
            elif conv == 'gcn':
                self.convs.append(
                    GCNConv(bond_encoder_handler(),
                            in_channels=in_dims[i],
                            out_channels=hidden,
                            cached=conv_cache)
                )
            elif conv == 'sage':
                self.convs.append(
                    SAGEConv(bond_encoder_handler(),
                             in_channels=in_dims[i],
                             out_channels=hidden)
                )
            else:
                raise NotImplementedError
            self.norms.append(normalization_resolver(norm, hidden) if norm is not None else Identity())

        self.supports_norm_batch = False
        if len(self.norms) > 0 and hasattr(self.norms[0], 'forward'):
            norm_params = inspect.signature(self.norms[0].forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        self.has_edge_attr = False
        if len(self.convs) > 0 and hasattr(self.convs[0], 'forward'):
            params = inspect.signature(self.convs[0].forward).parameters
            self.has_edge_attr = 'edge_attr' in params

        self.mlp = MLP(in_channels=in_feature if num_conv_layers == 0 else hidden,
                       hidden_channels=hidden,
                       out_channels=max_num_centroids * num_ensemble,
                       num_layers=num_mlp_layers,
                       act=activation,
                       norm=norm)

    def forward(self, data):
        batch, edge_index, edge_attr = data.batch, data.edge_index, data.edge_attr
        x = self.atom_encoder(data)

        for i, conv in enumerate(self.convs):
            if self.has_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            if self.supports_norm_batch:
                x = self.norms[i](x, batch)
            else:
                x = self.norms[i](x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.mlp(x, batch)
        x = x.reshape(-1, self.max_num_centroids, self.num_ensemble)
        return x
