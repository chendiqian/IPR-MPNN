from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import GCNConv as PyGGCNConv, SAGEConv as PyGSAGEConv, GINEConv as PyGGINEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import (
    scatter,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

from models.nn_utils import add_self_loop_multi_target


def gcn_norm(  # noqa: F811
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_attr, edge_weight = add_self_loop_multi_target(
            edge_index,
            num_nodes,
            0,
            edge_attr,
            edge_weight
        )

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_attr, edge_weight


class GCNConv(PyGGCNConv):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor, Tensor]]

    def __init__(
        self,
        bond_encoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bond_encoder = torch.nn.Sequential(bond_encoder, Linear(-1, kwargs['out_channels']))


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: OptTensor = None,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_attr, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_attr, edge_weight, x.size(self.node_dim),
                        self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_attr, edge_weight)
                else:
                    edge_index, edge_attr, edge_weight = cache[0], cache[1], cache[2]

        if self.bond_encoder is not None and edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConv(PyGSAGEConv):
    def __init__(
        self,
        bond_encoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bond_encoder = torch.nn.Sequential(bond_encoder, Linear(-1, kwargs['in_channels']))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        if self.bond_encoder is not None and edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        return x_j if edge_attr is None else x_j + edge_attr


class GINEConv(PyGGINEConv):

    def __init__(self,
                 bond_encoder,
                 **kwargs):
        super().__init__(**kwargs)
        self.bond_encoder = torch.nn.Sequential(bond_encoder, Linear(-1, kwargs['in_channels']))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.bond_encoder is not None and edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        return x_j if edge_attr is None else x_j + edge_attr
