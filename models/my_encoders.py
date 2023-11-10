import torch


class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr.squeeze())
        else:
            return None


class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        return self.embedding(data.x.squeeze())


class LinearEncocer(torch.nn.Module):
    def __init__(self, in_feature, hidden):
        super(LinearEncocer, self).__init__()
        self.embedding = torch.nn.Linear(in_feature, hidden)

    def forward(self, data):
        return self.embedding(data.x)


def get_atom_encoder(atom_encoder: str, hidden: int, in_feature: int = None):
    if atom_encoder == 'zinc':
        return ZINCAtomEncoder(hidden)
    elif atom_encoder == 'linear':
        return LinearEncocer(in_feature, hidden)
    else:
        raise NotImplementedError


def get_bond_encoder(bond_encoder: str, hidden: int):
    if bond_encoder == 'zinc':
        return ZINCBondEncoder(hidden)
    elif bond_encoder is None:
        return None
    else:
        raise NotImplementedError
