from typing import Final

import torch
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data

from models.my_conv import GINEConv, GCNConv, SAGEConv


class GINE(BasicGNN):
    # missing from PyG
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True  # this is important
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        edge_encoder_handler = kwargs['edge_encoder']
        kwargs.setdefault('in_channels', in_channels)
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(edge_encoder_handler(), nn=mlp, **kwargs)


class GCN(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        edge_encoder_handler = kwargs['edge_encoder']
        return GCNConv(edge_encoder_handler(), in_channels=in_channels, out_channels=out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int,
                  out_channels: int, **kwargs) -> MessagePassing:
        edge_encoder_handler = kwargs['edge_encoder']
        return SAGEConv(edge_encoder_handler(), in_channels=in_channels, out_channels=out_channels, **kwargs)


class PlainGNN(torch.nn.Module):
    def __init__(self,
                 node_encoder: torch.nn.Module,
                 prediction_mlp: torch.nn.Module,
                 gnn: torch.nn.Module):
        super(PlainGNN, self).__init__()
        self.node_encoder = node_encoder
        self.prediction_mlp = prediction_mlp
        self.gnn = gnn

    def forward(self, data: Data, *args):
        batch, edge_index, edge_attr = data.batch, data.edge_index, data.edge_attr
        x = self.node_encoder(data)
        x = self.gnn(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = self.prediction_mlp(x, None, batch, None)
        return x, None, None, 0.
