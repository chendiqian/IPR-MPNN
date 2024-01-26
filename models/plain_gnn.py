from typing import Final

import torch
from torch_geometric.nn import MLP, GINEConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data


class GINE(BasicGNN):
    # missing from PyG
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True  # this is important
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)


class PlainGNN(torch.nn.Module):
    def __init__(self,
                 bond_encoder: torch.nn.Module,
                 node_encoder: torch.nn.Module,
                 prediction_mlp: torch.nn.Module,
                 gnn: torch.nn.Module):
        super(PlainGNN, self).__init__()
        self.bond_encoder = bond_encoder
        self.node_encoder = node_encoder
        self.prediction_mlp = prediction_mlp
        self.gnn = gnn

    def forward(self, data: Data, *args):
        batch, edge_index, edge_attr = data.batch, data.edge_index, data.edge_attr
        x = self.node_encoder(data)
        if edge_attr is not None and self.bond_encoder is not None:
            edge_attr = self.bond_encoder(edge_attr)

        x = self.gnn(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = self.prediction_mlp(x, None, batch, None)
        return x, None, None, 0.
