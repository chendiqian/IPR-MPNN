from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from functools import partial
import torch
from numpy import argsort, unique, cumsum, split
from torch import Tensor
from torch.nn import ModuleList, functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, MLP, Linear
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType, Adj, OptTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)

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
                        # from some previous src, but current dst
                        x_src = list_x_dict[-(self.delay + 1)][src]
                        x_dst = new_x_dict[dst] if new_x_dict[dst] is not None else list_x_dict[-1][dst]
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
    def __init__(self, hid_dim, edge_encoder, num_mlp_layers, norm, act, aggr='add'):
        super(HeteroGINEConv, self).__init__(aggr=aggr)

        self.lin_src = Linear(-1, hid_dim)
        self.lin_dst = Linear(-1, hid_dim)
        self.edge_encoder = torch.nn.Sequential(edge_encoder, Linear(-1, hid_dim))
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


class HeteroGATv2Conv(MessagePassing):
    """
    this is supposed to be used for c2c_conv only
    """
    def __init__(
            self,
            hid_dim,
            edge_encoder,
            num_mlp_layers,
            norm,
            act,

            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.heads = heads
        self.hid_dim = hid_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        self.lin_l = Linear(-1, heads * hid_dim, bias=bias, weight_initializer='glorot')
        self.lin_r = Linear(-1, heads * hid_dim, bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, hid_dim))
        glorot(self.att)

        self.lin_edge = torch.nn.Sequential(edge_encoder, Linear(-1, hid_dim))
        self.mlp = MLP([-1] + [hid_dim] * num_mlp_layers, norm=norm, act=act)

    def forward(self, x, edge_index, edge_attr, edge_weight=None, batch=None):
        # a heuristic whether src are same type as dst. might fail
        is_hetero = x[0].shape[0] != x[1].shape[0]
        H, C = self.heads, self.hid_dim

        x_l = self.lin_l(x[0]).view(-1, H, C)
        x_r = self.lin_r(x[1]).view(-1, H, C)

        if self.add_self_loops and not is_hetero:
            num_nodes = x_l.shape[0]
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r),
                                  edge_attr=edge_attr)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, edge_weight=edge_weight)

        if is_hetero:
            out = out + x_r

        if self.concat:
            out = out.view(-1, self.heads * self.hid_dim)
        else:
            out = out.mean(dim=1)

        return self.mlp(out, batch)

    def edge_update(self, x_j: Tensor,
                    x_i: Tensor,
                    edge_attr: OptTensor,
                    index: Tensor,
                    ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor, edge_weight: Tensor=None) -> Tensor:
        if edge_weight is not None and edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None, None]
        # alpha: E x H
        # x_j: E x H x C
        m = x_j * alpha[:, None]
        return m * edge_weight if edge_weight is not None else m


class HeteroMLP(torch.nn.Module):
    """
    used as an intermediate NN without message passing
    """
    def __init__(self, hid_dim, edge_encoder, num_mlp_layers, norm, act):
        super(HeteroMLP, self).__init__()
        self.mlp = MLP([-1] + [hid_dim] * num_mlp_layers, norm=norm, act=act)

    def forward(self, x, edge_index, edge_attr, edge_weight=None, batch=None):
        return self.mlp(x[0], batch)


class HeteroSAGEConv(MessagePassing):
    def __init__(self, hid_dim, edge_encoder, num_mlp_layers, norm, act, aggr='mean'):
        super(HeteroSAGEConv, self).__init__(aggr=aggr)

        self.lin_src = Linear(-1, hid_dim)
        self.lin_dst = Linear(-1, hid_dim)
        self.edge_encoder = torch.nn.Sequential(edge_encoder, Linear(-1, hid_dim))
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
                 b2c_conv,
                 c2b_conv,
                 c2c_conv,
                 atom_encoder_handler,
                 bond_encoder_handler,
                 hid_dim,
                 centroid_hid_dim,
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

        def conv_distributor(cv):
            if cv == 'gine':
                f = HeteroGINEConv
            elif cv == 'gine_mean':
                f = partial(HeteroGINEConv, aggr='mean')
            elif cv == 'sage':
                f = HeteroSAGEConv
            elif cv == 'mlp':
                f = HeteroMLP
            elif cv == 'gat':
                f = HeteroGATv2Conv
            else:
                raise NotImplementedError
            return f

        f_conv = conv_distributor(conv)
        f_b2c_conv = conv_distributor(b2c_conv)
        f_c2b_conv = conv_distributor(c2b_conv)
        f_c2c_conv = conv_distributor(c2c_conv)

        b2b, b2c, c2c, c2b = (0, 0, 0, 0) if parallel else (1, 0, 0, 1)
        self.gnn_convs = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gnn_convs.append(
                # use GNN's default neighborhood aggr, while between centroid and base we use mean aggr
                HeteroConv({
                    (('base', 'to', 'base'), 'current'):
                        (f_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2b),
                    (('base', 'to', 'centroid'), 'current'):
                        (f_b2c_conv(centroid_hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2c),
                    (('base', 'to', 'centroid'), 'delay'):
                        (f_b2c_conv(centroid_hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), b2c),
                    (('centroid', 'to', 'centroid'), 'current'):
                        (f_c2c_conv(centroid_hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2c),
                    (('centroid', 'to', 'base'), 'current'):
                        (f_c2b_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2b),
                    (('centroid', 'to', 'base'), 'delay'):
                        (f_c2b_conv(hid_dim, bond_encoder_handler(), num_mlp_layers, norm, activation), c2b),
                },
                    delay=delay,
                    # aggr across different heterogeneity, e.g., cent and base to base
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
        
        list_x_dict = list_x_dict[1:] # Ignore input

        base_embeddings = [xd['base'] for xd in list_x_dict]
        centroid_embeddings = [xd['centroid'] for xd in list_x_dict]
        return base_embeddings, centroid_embeddings
