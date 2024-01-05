import warnings
from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import ModuleList, functional as F
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType, Adj
from torch_geometric.utils.hetero import check_add_self_loops

from models.nn_utils import residual


class HeteroConv(torch.nn.Module):
    def __init__(
        self,
        convs: Dict[EdgeType, Tuple[MessagePassing, int]],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, (module, _) in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict({'__'.join(k): v[0] for k, v in convs.items()})
        conv_rank = {'__'.join(k): v[1] for k, v in convs.items()}
        sorted_rank_value = sorted(set(conv_rank.values()))  # small value has high priority

        self.ranked_convs = ModuleList([])
        for i, rank in enumerate(sorted_rank_value):
            module_dict = ModuleDict({})
            for k, v in conv_rank.items():
                if v == rank:
                    module_dict[k] = self.convs[k]
            self.ranked_convs.append(module_dict)

        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # for conv in self.convs.values():
        #     conv.reset_parameters()
        raise NotImplementedError

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Dict[EdgeType, Tensor],
        edge_weight_dict: Dict[EdgeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        for cur_rank, cur_convs in enumerate(self.ranked_convs):
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                str_edge_type = '__'.join(edge_type)

                if str_edge_type not in cur_convs:
                    continue

                conv = cur_convs[str_edge_type]
                out = conv((x_dict[src], x_dict[dst]),
                           edge_index,
                           edge_attr_dict.get(edge_type, None),
                           edge_weight_dict.get(edge_type, None),
                           batch_dict[dst])
                out_dict[dst].append(out)

            for key, value in out_dict.items():
                x_dict[key] = group(value, self.aggr)

        return x_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'


class HeteroGINEConv(MessagePassing):
    def __init__(self, hid_dim, edge_encoder, num_mlp_layers, norm, act):
        super(HeteroGINEConv, self).__init__(aggr="add")

        self.lin_src = Linear(-1, hid_dim)
        self.lin_dst = Linear(-1, hid_dim)
        self.edge_encoder = edge_encoder
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm, act=act)
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, edge_index, edge_attr, edge_weight=None, batch=None):
        x = (self.lin_src(x[0]), x[1])

        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)

        x_dst = (1 + self.eps) * x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        return self.mlp(out, batch)

    def message(self, x_j, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out


class HeteroSAGEConv(MessagePassing):
    def __init__(self, hid_dim, edge_encoder, num_mlp_layers, norm, act):
        super(HeteroSAGEConv, self).__init__(aggr="mean")

        self.lin_src = Linear(-1, hid_dim)
        self.lin_dst = Linear(-1, hid_dim)
        self.edge_encoder = edge_encoder
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm, act=act)

    def forward(self, x, edge_index, edge_attr, edge_weight=None, batch=None):
        x = (self.lin_src(x[0]), x[1])

        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        x_dst = self.lin_dst(x[1])
        out = out + x_dst
        return self.mlp(out, batch)

    def message(self, x_j, edge_attr, edge_weight):
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]
        m = F.gelu(x_j + edge_attr) if edge_attr is not None else x_j
        return m * edge_weight if edge_weight is not None else m

    def update(self, aggr_out):
        return aggr_out


class HeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 atom_encoder_handler,
                 bond_encoder_handler,
                 hid_dim,
                 num_conv_layers,
                 num_mlp_layers,
                 dropout,
                 norm,
                 activation,
                 use_res,
                 aggr,
                 parallel):
        super(HeteroGNN, self).__init__()

        self.atom_encoder = atom_encoder_handler()
        self.dropout = dropout
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if conv == 'gine':
            f_conv = HeteroGINEConv
        elif conv == 'sage':
            f_conv = HeteroSAGEConv
        else:
            raise NotImplementedError

        b2b, b2c, c2c, c2b = (0, 0, 0, 0) if parallel else (1, 0, 0, 1)
        self.gnn_convs = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gnn_convs.append(
                HeteroConv({
                    ('base', 'to', 'base'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2b),
                    ('base', 'to', 'centroid'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2c),
                    ('centroid', 'to', 'centroid'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2c),
                    ('centroid', 'to', 'base'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2b),
                },
                    aggr=aggr))

    def forward(self, old_data, data, has_edge_attr):
        edge_index_dict, edge_weight_dict = data.edge_index_dict, data.edge_weight_dict
        edge_attr_dict = data.edge_attr_dict if has_edge_attr else {}
        batch_dict = data.batch_dict
        x_dict = data.x_dict
        repeats = batch_dict['base'].shape[0] // x_dict['base'].shape[0]
        x_dict['base'] = self.atom_encoder(old_data).repeat(repeats, 1)

        for i in range(self.num_layers):
            h1 = x_dict
            h2 = self.gnn_convs[i](x_dict, edge_index_dict, edge_attr_dict, edge_weight_dict, batch_dict)
            keys = h2.keys()
            if self.use_res:
                x_dict = {k: residual(h1[k], F.gelu(h2[k])) for k in keys}
            else:
                x_dict = {k: F.gelu(h2[k]) for k in keys}
            x_dict = {k: F.dropout(x_dict[k], p=self.dropout, training=self.training) for k in keys}

        return x_dict['base'], x_dict['centroid']
