from typing import Callable

import torch
from torch_geometric.nn import MLP


class ScorerGNN(torch.nn.Module):
    def __init__(self,
                 gnn: torch.nn.Module,
                 atom_encoder_handler: Callable,
                 hidden: int,
                 num_mlp_layers: int,
                 max_num_centroids: int,
                 num_ensemble: int,
                 norm: str,
                 activation: str):
        super(ScorerGNN, self).__init__()

        self.gnn = gnn

        self.atom_encoder = atom_encoder_handler(True)
        self.max_num_centroids = max_num_centroids
        self.num_ensemble = num_ensemble

        self.mlp = MLP(in_channels=-1,
                       hidden_channels=hidden,
                       out_channels=max_num_centroids * num_ensemble,
                       num_layers=num_mlp_layers,
                       act=activation,
                       norm=norm)

    def forward(self, data):
        batch, edge_index, edge_attr = data.batch, data.edge_index, data.edge_attr
        x = self.atom_encoder(data)

        if self.gnn is not None:
            x = self.gnn(x, edge_index, edge_attr=edge_attr, batch=batch)

        x = self.mlp(x, batch)
        x = x.reshape(-1, self.max_num_centroids, self.num_ensemble)
        return x
