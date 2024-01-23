from typing import Callable
import torch


class Predictor(torch.nn.Module):
    """
    A function that takes both node and centroid embeddings,
    and select or merge them into graph embedding.
    """
    def __init__(self,
                 pred_target: str,
                 inter_ensemble_pool: Callable,
                 inter_base_pred_head: torch.nn.Module,
                 inter_cent_pred_head: torch.nn.Module,
                 intra_graph_pool: Callable,
                 intra_pred_head: torch.nn.Module):
        super(Predictor, self).__init__()
        self.pred_target = pred_target
        self.inter_ensemble_pool = inter_ensemble_pool
        self.inter_base_pred_head = inter_base_pred_head
        self.inter_cent_pred_head = inter_cent_pred_head
        self.intra_graph_pool = intra_graph_pool
        self.intra_pred_head = intra_pred_head

    def forward(self, node_embedding, centroid_embedding, node_batch, centroid_batch):
        # potentially we have to concat node and centroid embeddings
        graph_embeddings = []

        if self.pred_target in ['base', 'both']:
            # merge the ensembles
            node_embedding = self.inter_ensemble_pool(node_embedding)
            # MLP
            node_embedding = self.inter_base_pred_head(node_embedding)
            # pool the nodes into root nodes or graph level embedding
            graph_embedding = self.intra_graph_pool(node_embedding, node_batch)
            graph_embeddings.append(graph_embedding)
        if self.pred_target in ['centroid', 'both']:
            centroid_embedding = self.inter_ensemble_pool(centroid_embedding)
            centroid_embedding = self.inter_cent_pred_head(centroid_embedding)
            graph_embedding = self.intra_graph_pool(centroid_embedding, centroid_batch)
            graph_embeddings.append(graph_embedding)

        graph_embedding = self.intra_pred_head(torch.cat(graph_embeddings, dim=1))
        return graph_embedding
