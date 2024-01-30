from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    sort_edge_index
)


class AugmentWithPartition:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/cluster.html#ClusterData
    """
    def __init__(self, num_parts, recursive=True):
        super(AugmentWithPartition, self).__init__()
        self.num_parts = num_parts
        self.recursive = recursive

    def __call__(self, graph: Data):
        row, index = sort_edge_index(graph.edge_index, num_nodes=graph.num_nodes, sort_by_row=True)
        indptr = index2ptr(row, size=graph.num_nodes)

        cluster = torch.ops.torch_sparse.partition(
            indptr.cpu(),
            index.cpu(),
            None,
            self.num_parts,
            self.recursive,
        ).to(graph.edge_index.device)

        graph.partition = cluster
        return graph


class AugmentWithDumbAttr:
    def __call__(self, graph: Data):
        graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float)
        graph.edge_attr = torch.ones(graph.num_edges, 1, dtype=torch.float)
        return graph


class RenameLabel:
    # dumb class to rename edge_label to y
    def __call__(self, graph: Data):
        graph.y = graph.edge_label.float()  # for BCE loss
        del graph.edge_label
        return graph


class AddLaplacianEigenvectorPE:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=self.k + 1,
                which='SR' if not self.is_undirected else 'SA',
                return_eigenvectors=True,
                **self.kwargs,
            )

        sort_idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:, sort_idx])

        data.EigVecs = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        data.EigVals = torch.from_numpy(np.real(eig_vals[sort_idx][1:self.k + 1]))
        return data
