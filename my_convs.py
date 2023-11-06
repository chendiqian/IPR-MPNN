from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, MLP
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder


class GINEConv(MessagePassing):
    def __init__(self, emb_dim: int = 64,
                 mlp: Optional[Union[MLP, torch.nn.Sequential]] = None,
                 bond_encoder: Optional[Union[MLP, torch.nn.Sequential]] = None):

        super(GINEConv, self).__init__(aggr="add")

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

        if bond_encoder is None:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        else:
            self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr, edge_weight):
        repeats, nnodes, features = x.shape
        if edge_attr is not None:
            edge_embedding = self.bond_encoder(edge_attr)
            edge_embedding = edge_embedding.repeat(repeats, 1, 1)
        else:
            edge_embedding = None

        x = x.reshape(-1, features)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out
