import torch
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr


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
