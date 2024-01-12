from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from numpy import argsort, unique, cumsum, split
import torch
from torch import Tensor
from torch.nn import ModuleList, functional as F
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType, Adj

from models.nn_utils import residual


class HeteroConv(torch.nn.Module):
    def __init__(
        self,
        typed_convs: Dict[Tuple[EdgeType, str], Tuple[MessagePassing, int]],
        delay: Optional[int] = 0,
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        # order them
        edge_types = []
        ranks = []
        for (etype, tmp), (_, rank) in typed_convs.items():
            edge_types.append((etype, tmp))
            ranks.append(rank)

        # sort the ranks
        rank_sort_idx = argsort(ranks)
        _, rank_counts = unique(ranks, return_counts=True)
        split_rank_sort_idx = split(rank_sort_idx, cumsum(rank_counts)[:-1])

        self.ranked_convs = ModuleList([])
        for blk in split_rank_sort_idx:
            module_dict = ModuleDict({})
            for idx in blk:
                etype, tmp = edge_types[idx]
                module_dict['__'.join([*etype, tmp])] = typed_convs[(etype, tmp)][0]
            self.ranked_convs.append(module_dict)

        self.delay = delay
        self.aggr = aggr

    def forward(
        self,
        list_x_dict: List[Dict[NodeType, Tensor]],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Dict[EdgeType, Tensor],
        edge_weight_dict: Dict[EdgeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ):
        new_x_dict = {k: None for k in list_x_dict[-1].keys()}
        for cur_convs in self.ranked_convs:
            out_dict = defaultdict(list)
            for edge_tmp, conv in cur_convs.items():
                src, rel, dst, temp = edge_tmp.split('__')
                edge_type = (src, rel, dst)

                if temp == 'delay':
                    if self.delay and len(list_x_dict) >= self.delay + 1:
                        x_src = list_x_dict[-(self.delay + 1)][src]
                        x_dst = list_x_dict[-(self.delay + 1)][dst]
                    else:
                        continue
                else:
                    x_src = new_x_dict[src] if new_x_dict[src] is not None else list_x_dict[-1][src]
                    x_dst = new_x_dict[dst] if new_x_dict[dst] is not None else list_x_dict[-1][dst]

                out = conv((x_src, x_dst),
                           edge_index_dict[edge_type],
                           edge_attr_dict.get(edge_type, None),
                           edge_weight_dict.get(edge_type, None),
                           batch_dict[dst])
                out_dict[edge_type].append(out)

            # inter-delay tensors must be aggregated with shape-preserving aggr func
            # otherwise shapes mismatch as earlier layers have no delayed input
            merged_src_out_dict = defaultdict(list)
            for (src, rel, dst), tensors in out_dict.items():
                merged_dst = torch.stack(tensors, dim=0).mean(0) if len(tensors) > 1 else tensors[0]
                merged_src_out_dict[dst].append(merged_dst)

            for key, value in merged_src_out_dict.items():
                new_x_dict[key] = group(value, self.aggr)

        return new_x_dict


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
                 delay,
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
                    (('base', 'to', 'base'), 'current'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2b),
                    (('base', 'to', 'centroid'), 'current'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2c),
                    (('base', 'to', 'centroid'), 'delay'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2c),
                    (('centroid', 'to', 'centroid'), 'current'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2c),
                    (('centroid', 'to', 'base'), 'current'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2b),
                    (('centroid', 'to', 'base'), 'delay'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2b),
                },
                    delay=delay,
                    aggr=aggr))

    def forward(self, old_data, data, has_edge_attr):
        edge_index_dict, edge_weight_dict = data.edge_index_dict, data.edge_weight_dict
        edge_attr_dict = data.edge_attr_dict if has_edge_attr else {}
        batch_dict = data.batch_dict
        x_dict = data.x_dict
        repeats = batch_dict['base'].shape[0] // x_dict['base'].shape[0]
        x_dict['base'] = self.atom_encoder(old_data).repeat(repeats, 1)

        list_x_dict = [x_dict]

        for i in range(self.num_layers):
            h1 = list_x_dict[-1]
            h2 = self.gnn_convs[i](list_x_dict, edge_index_dict, edge_attr_dict, edge_weight_dict, batch_dict)
            keys = h2.keys()
            if self.use_res:
                new_x_dict = {k: residual(h1[k], F.gelu(h2[k])) for k in keys}
            else:
                new_x_dict = {k: F.gelu(h2[k]) for k in keys}
            new_x_dict = {k: F.dropout(new_x_dict[k], p=self.dropout, training=self.training) for k in keys}
            list_x_dict.append(new_x_dict)

        base_embeddings = [xd['base'] for xd in list_x_dict]
        centroid_embeddings = [xd['centroid'] for xd in list_x_dict]
        return base_embeddings, centroid_embeddings
