import torch


class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        return self.embedding(x)


class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        return self.embedding(x)


def get_atom_encoder(atom_encoder: str, hidden: int):
    if atom_encoder == 'zinc':
        return ZINCAtomEncoder(hidden)
    else:
        raise NotImplementedError


def get_bond_encoder(bond_encoder: str, hidden: int):
    if bond_encoder == 'zinc':
        return ZINCBondEncoder(hidden)
    else:
        raise NotImplementedError
