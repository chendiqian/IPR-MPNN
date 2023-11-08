import torch
from torch_geometric.nn import MLP


class ScorerGNN(torch.nn.Module):
    def __init__(self,
                 conv: str,
                 atom_encoder: torch.nn.Module,
                 bond_encoder: torch.nn.Module,
                 hidden: int,
                 # num_layers: int,
                 num_mlp_layers: int,
                 num_centroids: int,
                 num_ensemble: int,
                 norm: str,
                 activation: str,
                 # dropout: float
                 ):
        super(ScorerGNN, self).__init__()

        self.atom_encoder = atom_encoder
        self.edge_encoder = bond_encoder
        self.num_centroids = num_centroids
        self.num_ensemble = num_ensemble

        assert conv == 'mlp'

        self.mlp = MLP(in_channels=hidden,
                       hidden_channels=hidden,
                       out_channels=num_centroids * num_ensemble,
                       num_layers=num_mlp_layers,
                       act=activation,
                       norm=norm)

    def forward(self, x, *args, **kwargs):
        x = self.mlp(x)
        x = x.reshape(-1, self.num_centroids, self.num_ensemble)
        return x