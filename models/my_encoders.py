import torch
from torch import nn as nn
from torch_geometric.nn import MLP
from ogb.graphproppred.mol_encoder import AtomEncoder as OGB_AtomEncoder, BondEncoder as OGB_BondEncoder

from data.utils import Config


class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr)
        else:
            return None


class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        return self.embedding(data.x)


class EXPAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(EXPAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        return self.embedding(data.x)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_feature, hidden):
        super(LinearEncoder, self).__init__()
        self.embedding = torch.nn.Linear(in_feature, hidden)

    def forward(self, data):
        return self.embedding(data.x)
    
class BiEmbedding(torch.nn.Module):
    def __init__(self,
                 dim_in,
                 hidden,):
        super(BiEmbedding, self).__init__()
        self.layer0_keys = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)
        self.layer0_values = nn.Embedding(num_embeddings=dim_in + 1, embedding_dim=hidden)

    def forward(self, data):
        x = data.x
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed
        return x

    def reset_parameters(self):
        self.layer0_keys.reset_parameters()
        self.layer0_values.reset_parameters()

class BiEmbeddingCat(torch.nn.Module):
    def __init__(self,
                 n_nodes,
                 n_features,
                 hidden,):
        super(BiEmbeddingCat, self).__init__()
        self.emb_node = nn.Embedding(num_embeddings=n_nodes, embedding_dim=hidden)
        self.emb_feature = nn.Embedding(num_embeddings=n_features, embedding_dim=hidden)

    def forward(self, data):
        x = data.x
        x_node, x_feature = x[:, 0], x[:, 1]
        node_emb = self.emb_node(x_node)
        feature_emb = self.emb_feature(x_feature)
        x = torch.cat([node_emb, feature_emb], dim=-1)
        return x

    def reset_parameters(self):
        self.layer0_keys.reset_parameters()
        self.layer0_values.reset_parameters()


class LinearBondEncoder(torch.nn.Module):
    def __init__(self, in_feature, hidden):
        super(LinearBondEncoder, self).__init__()
        self.embedding = torch.nn.Linear(in_feature, hidden)
    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr)
        else:
            return None


class MyOGBAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(MyOGBAtomEncoder, self).__init__()
        self.embedding = OGB_AtomEncoder(hidden)

    def forward(self, data):
        return self.embedding(data.x)


class COCONodeEncoder(torch.nn.Module):
    # https://github.com/toenshoff/LRGB/blob/main/graphgps/encoder/voc_superpixels_encoder.py#L68
    def __init__(self, emb_dim):
        super().__init__()

        node_x_mean = torch.tensor([
            4.6977347e-01, 4.4679317e-01, 4.0790915e-01, 7.0808627e-02,
            6.8686441e-02, 6.8498217e-02, 6.7777938e-01, 6.5244222e-01,
            6.2096798e-01, 2.7554795e-01, 2.5910738e-01, 2.2901227e-01,
            2.4261935e+02, 2.8985367e+02
        ])
        node_x_std = torch.tensor([
            2.6218116e-01, 2.5831082e-01, 2.7416739e-01, 5.7440419e-02,
            5.6832556e-02, 5.7100497e-02, 2.5929087e-01, 2.6201612e-01,
            2.7675411e-01, 2.5456995e-01, 2.5140920e-01, 2.6182330e-01,
            1.5152475e+02, 1.7630779e+02
        ])

        self.register_buffer('node_x_mean', node_x_mean)
        self.register_buffer('node_x_std', node_x_std)
        self.encoder = torch.nn.Linear(14, emb_dim)

    def forward(self, batch):
        x = batch.x - self.node_x_mean.view(1, -1)
        x /= self.node_x_std.view(1, -1)
        x = self.encoder(x)
        return x


class COCOEdgeEncoder(torch.nn.Module):
    # https://github.com/toenshoff/LRGB/blob/main/graphgps/encoder/voc_superpixels_encoder.py#L97
    def __init__(self, emb_dim):
        super().__init__()
        edge_x_mean = torch.tensor([0.07848548, 43.68736])
        edge_x_std = torch.tensor([0.08902349, 28.473562])
        self.register_buffer('edge_x_mean', edge_x_mean)
        self.register_buffer('edge_x_std', edge_x_std)
        self.encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, edge_attr):
        edge_attr = edge_attr - self.edge_x_mean.view(1, -1)
        edge_attr /= self.edge_x_std.view(1, -1)
        edge_attr = self.encoder(edge_attr)
        return edge_attr


class FeatureEncoder(torch.nn.Module):

    def __init__(self,
                 dim_in,
                 hidden,
                 type_encoder: str,
                 lap_encoder: Config = None,
                 rw_encoder: Config = None,
                 partition_encoder: Config = None):
        super(FeatureEncoder, self).__init__()

        lin_hidden = hidden
        if lap_encoder is not None:
            lin_hidden -= lap_encoder.dim_pe
        if rw_encoder is not None:
            lin_hidden -= rw_encoder.dim_pe
        if partition_encoder is not None:
            lin_hidden -= partition_encoder.dim_pe

        assert lin_hidden > 0

        self.linear_embed = get_atom_encoder(type_encoder, lin_hidden, dim_in)

        if lap_encoder is not None:
            self.lap_encoder = LapPENodeEncoder(hidden,
                                                hidden - (rw_encoder.dim_pe if rw_encoder is not None else 0)
                                                - (partition_encoder.dim_pe if partition_encoder is not None else 0),
                                                lap_encoder,
                                                expand_x=False)
        else:
            self.lap_encoder = None

        if rw_encoder is not None:
            self.rw_encoder = RWSENodeEncoder(hidden,
                                              hidden - (partition_encoder.dim_pe if partition_encoder is not None else 0),
                                              rw_encoder,
                                              expand_x=False)
        else:
            self.rw_encoder = None

        if partition_encoder is not None:
            self.part_encoder = PartitionInfoEncoder(hidden, hidden, partition_encoder.dim_pe, expand_x=False)
        else:
            self.part_encoder = None

    def forward(self, data):
        x = self.linear_embed(data)
        if self.lap_encoder is not None:
            x = self.lap_encoder(x, data)
        if self.rw_encoder is not None:
            x = self.rw_encoder(x, data)
        if self.part_encoder is not None:
            x = self.part_encoder(x, data)
        return x


class LapPENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/laplace_pos_encoder.py
    """Laplace Positional Embedding node encoder.
    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()

        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        max_freqs = pecfg.max_freqs  # Num. eigenvectors (frequencies)

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = torch.nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if pecfg.raw_norm_type is None or pecfg.raw_norm_type == 'None':
            raw_norm = torch.nn.Identity()
        elif pecfg.raw_norm_type.lower() == 'batchnorm':
            raw_norm = torch.nn.BatchNorm1d(max_freqs)
        else:
            raise ValueError

        self.pe_encoder = torch.nn.Sequential(
            raw_norm,
            MLP([max_freqs] + (n_layers - 1) * [2 * dim_pe] + [dim_pe])
        )

    def forward(self, x, batch):
        if not hasattr(batch, 'EigVecs'):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        pos_enc = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(pos_enc.size(1), device=pos_enc.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            pos_enc = pos_enc * sign_flip.unsqueeze(0)

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors)
        # pos_enc = (self.pe_encoder(eigvecs) + self.pe_encoder(-eigvecs)) * 0.5  # (Num nodes) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        return x


class RWSENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/kernel_pos_encoder.py
    """Configurable kernel-based Positional Encoding node encoder.
    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.
    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = 'RWSE'  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = pecfg.kernel
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        n_layers = pecfg.layers

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = torch.nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = torch.nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        self.pe_encoder = MLP([num_rw_steps] + (n_layers - 1) * [2 * dim_pe] + [dim_pe])

    def forward(self, x, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        return x


class PartitionInfoEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, dim_pe, expand_x=True):
        super().__init__()

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"Part size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = torch.nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # todo: temporary set 20
        self.pe_encoder = torch.nn.Embedding(20, dim_pe)
        torch.nn.init.xavier_uniform_(self.pe_encoder.weight.data)

    def forward(self, x, batch):
        if not hasattr(batch, 'partition'):
            raise ValueError("Precomputed partitions are "
                             f"required for {self.__class__.__name__}")
        pos_enc = self.pe_encoder(batch.partition)

        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        x = torch.cat((h, pos_enc), 1)
        return x


def get_atom_encoder(atom_encoder: str,
                     hidden: int,
                     in_feature: int = None,
                     lap_args: Config = None,
                     rw_args: Config = None,
                     partition_args: Config = None):
    if lap_args is not None or rw_args is not None or partition_args is not None:
        return FeatureEncoder(in_feature, hidden, atom_encoder, lap_args, rw_args, partition_args)
    else:
        if atom_encoder == 'zinc':
            return ZINCAtomEncoder(hidden)
        elif atom_encoder == 'ogb':
            return MyOGBAtomEncoder(hidden)
        elif atom_encoder == 'exp':
            return EXPAtomEncoder(hidden)
        elif atom_encoder == 'linear':
            return LinearEncoder(in_feature, hidden)
        elif atom_encoder == 'bi_embedding':
            return BiEmbedding(in_feature, hidden)
        elif atom_encoder == 'bi_embedding_cat':
            assert hidden % 2 == 0, 'hidden size must be even'
            return BiEmbeddingCat(n_nodes=in_feature, n_features=2, hidden=hidden // 2)
        elif atom_encoder == 'coco':
            return COCONodeEncoder(hidden)
        else:
            raise NotImplementedError


def get_bond_encoder(bond_encoder: str, hidden: int, in_features: int = None):
    if bond_encoder == 'zinc':
        return ZINCBondEncoder(hidden)
    elif bond_encoder == 'ogb':
        return OGB_BondEncoder(hidden)
    elif bond_encoder == 'linear':
        return LinearBondEncoder(in_features, hidden)
    elif bond_encoder is None:
        return None
    elif bond_encoder == 'coco':
        return COCOEdgeEncoder(hidden)
    else:
        raise NotImplementedError
