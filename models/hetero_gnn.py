import torch
import torch.nn.functional as F

# from torch_geometric.nn import MLP
from models.hetero_convs import HeteroConv, HeteroGINEConv
from models.nn_utils import residual


def get_conv_layer(conv: str,
                   in_dim: int,
                   hid_dim: int,
                   edge_encoder: torch.nn.Module,
                   num_mlp_layers: int,
                   norm: str,
                   act: str):
    if conv.lower() == 'gine':
        def get_conv():
            return HeteroGINEConv(in_dim=in_dim,
                                  hid_dim=hid_dim,
                                  edge_encoder=edge_encoder,
                                  num_mlp_layers=num_mlp_layers,
                                  norm=norm,
                                  act=act)
    else:
        raise NotImplementedError
    return get_conv


class HeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 in_place,
                 edge_encoder,
                 in_feature,
                 hid_dim,
                 num_conv_layers,
                 # num_pred_layers,
                 num_mlp_layers,
                 # num_classes,
                 dropout,
                 norm,
                 activation,
                 use_res):
        super(HeteroGNN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers
        self.use_res = use_res

        get_conv_first_layer = get_conv_layer(conv, in_feature, hid_dim, edge_encoder, num_mlp_layers, norm, activation)
        get_conv = get_conv_layer(conv, hid_dim, hid_dim, edge_encoder, num_mlp_layers, norm, activation)
        self.gnn_convs = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gnn_convs.append(
                HeteroConv({
                    ('base', 'to', 'base'): (get_conv_first_layer() if layer == 0 else get_conv(), 0),
                    ('base', 'to', 'centroid'): (get_conv_first_layer() if layer == 0 else get_conv(), 1),
                    ('centroid', 'to', 'centroid'): (get_conv_first_layer() if layer == 0 else get_conv(), 2),
                    ('centroid', 'to', 'base'): (get_conv_first_layer() if layer == 0 else get_conv(), 3),
                },
                    in_place=in_place,
                    aggr='mean'))

        # self.pred_base = MLP([hid_dim] * num_pred_layers + [num_classes])
        # self.pred_cent = MLP([hid_dim] * num_pred_layers + [num_classes])

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict, edge_attr_dict, edge_weight_dict = data.edge_index_dict, data.edge_attr_dict, data.edge_weight_dict

        for i in range(self.num_layers):
            h1 = x_dict
            h2 = self.gnn_convs[i](x_dict, edge_index_dict, edge_attr_dict, edge_weight_dict)
            keys = h2.keys()
            if self.use_res:
                x_dict = {k: residual(h1[k], F.gelu(h2[k])) for k in keys}
            else:
                x_dict = {k: F.gelu(h2[k]) for k in keys}
            x_dict = {k: F.dropout(x_dict[k], p=self.dropout, training=self.training) for k in keys}

        # base_embedding = self.pred_base(x_dict['base'])
        # cen_embedding = self.pred_cent(x_dict['centroid'])
        # return base_embedding, cen_embedding

        return x_dict['base'], x_dict['centroid']
