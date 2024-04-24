from typing import Union, Dict
from collections import defaultdict

import torch
from data.metrics import Evaluator

from torch_geometric.utils import to_dense_batch


class Trainer:
    def __init__(self,
                 task: str,
                 criterion: torch.nn.modules.loss,
                 evaluator: Evaluator,
                 target_metric: str,
                 device: Union[str, torch.device]):
        super(Trainer, self).__init__()

        if task == 'edge':
            self.train = self.train_link_pred
            self.test = self.test_link_pred
        else:
            self.train = self.train_node_graph_prediction
            self.test = self.test_node_graph_prediction
        self.criterion = criterion
        self.evaluator = evaluator
        self.target_metric = target_metric
        self.device = device
        self.clear_stats()

    def train_node_graph_prediction(self, train_loader, model, optimizer):
        model.train()

        train_losses = 0.
        preds = []
        labels = []
        num_instances = 0

        for data in train_loader.loader:
            data = data.to(self.device)
            y = data.y
            num_instances += y.shape[0]

            optimizer.zero_grad()
            outputs, _, _, auxloss = model(data)
            if type(outputs) == list:
                head_losses = [self.criterion(output, y) for output in outputs]
                loss = torch.sum(torch.stack(head_losses))
                preds.append(torch.stack(outputs, dim=0).detach().mean(dim=0))
            else:
                loss = self.criterion(outputs, y)
                preds.append(outputs.detach())
            train_losses += loss.detach() * y.shape[0]
            loss = loss + auxloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            labels.append(y)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric: Dict = self.evaluator(labels, preds)
        return train_losses.item() / num_instances, train_metric

    @torch.no_grad()
    def test_node_graph_prediction(self, loader, model, scheduler, epoch=None):
        model.eval()

        val_losses = 0.
        preds = []
        labels = []
        num_instances = 0

        for data in loader.loader:
            data = data.to(self.device)
            y = data.y
            num_instances += y.shape[0]

            outputs, *_ = model(data)
            if type(outputs) == list:
                head_losses = [self.criterion(output, y) for output in outputs]
                loss = torch.sum(torch.stack(head_losses))
                preds.append(torch.stack(outputs, dim=0).detach().mean(dim=0))
            else: 
                loss = self.criterion(outputs, y)
                preds.append(outputs.detach())

            val_losses += loss.detach() * y.shape[0]
            labels.append(y)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        preds = preds * loader.std.to(self.device) if loader.std is not None else preds
        labels = labels * loader.std.to(self.device) if loader.std is not None else labels
        val_metric: Dict = self.evaluator(labels, preds)
        if scheduler is not None:
            scheduler.step(epoch) if 'LambdaLR' in str(type(scheduler)) else scheduler.step(val_metric[self.target_metric])
        return val_losses.item() / num_instances, val_metric

    def get_sensitivity(self, loader, model):
        from functools import partial
        from torch_geometric.data import Batch
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        import numpy as np
        from torch.autograd.functional import jacobian
        from tqdm import tqdm
        import time

        norms_same = []
        norms_diff = []

        for data in tqdm(loader.loader, desc='Computing sensitivity...'):
            data = data.to(self.device)
            graphs = Batch.to_data_list(data)

            #this could be vmapped I think
            for g in tqdm(graphs, desc='Computing sensitivity for each graph in batch...'):
                g = g.cpu()
                mat = csr_matrix(
                    (
                        np.ones(g.edge_index.shape[1]),
                        (g.edge_index[0].numpy(), g.edge_index[1].numpy()),
                    ),
                    shape=(g.num_nodes, g.num_nodes),
                )

                mat = shortest_path(mat, directed=False, return_predecessors=False)
                mat[np.isinf(mat)] = -1.
                mat[mat == -1] = mat.max() + 1

                candidate_idx = np.vstack(np.triu_indices(g.num_nodes, k=1))

                distances = mat[candidate_idx[0], candidate_idx[1]]
                most_distant_pair = candidate_idx[:, np.argsort(distances)[-1]]
                u, v = most_distant_pair

                batch_of_1 = Batch.from_data_list([g]).to(self.device)

                (_, data_hetero, has_edge_attr), base_embeddings, centroid_embeddings = model(batch_of_1, return_for_sensitivity=True)

                # get rid of final embedding
                base_embeddings = base_embeddings[:-1]
                centroid_embeddings = centroid_embeddings[:-1]

                hetero_model = model.hetero_gnn

                layers_list = [[i for i in range(start_layer, len(base_embeddings))] for start_layer in range(len(base_embeddings))]

                def forward_fn(x, centr_emb, data, has_edge_attr, layer_list):
                    input_embs = {
                        "base": x,
                        "centroid": centr_emb
                    }
                    base_embeddings = hetero_model.partial_forward(input_embs, data=data, has_edge_attr=has_edge_attr, layer_list=layer_list)
                    return base_embeddings[-1][(u,v), :] #return the embeddings of the most distant pair
                
                layerwise_norms_same = []
                layerwise_norms_diff = []

                for layer_list, x, centr_emb in zip(layers_list, base_embeddings, centroid_embeddings):
                    model_fwd = partial(forward_fn, centr_emb=centr_emb, data=data_hetero, has_edge_attr=has_edge_attr, layer_list=layer_list)
                    out_u_idx, out_v_idx = 0, 1

                    # tick = time.time()
                    # j_i = jacobian(model_fwd, x, vectorize=True)
                    # tock = time.time()
                    # print('time taken for vectorized=true: ', tock - tick)
                    # tick = time.time()
                    # j_i_1 = jacobian(model_fwd, x)
                    # tock = time.time()
                    # print('time taken for vectorized=false: ', tock - tick)
                    # #check if they are close
                    # input_norm_1 = torch.linalg.norm(j_i_1[out_u_idx, :, v, :], ord='fro')
                    # input_norm = torch.linalg.norm(j_i[out_u_idx, :, v, :], ord='fro')
                    # print('Are jacobians norms close?', torch.allclose(input_norm, input_norm_1))
                    # print('\n\n')

                    j_i = jacobian(model_fwd, x, vectorize=True)

                    norm_vv = torch.linalg.norm(j_i[out_v_idx, :, v, :], ord='fro')
                    norm_uu = torch.linalg.norm(j_i[out_u_idx, :, u, :], ord='fro')
                    norm_vv_plus_uu = norm_vv + norm_uu

                    norm_vu = torch.linalg.norm(j_i[out_v_idx, :, u, :], ord='fro')
                    norm_uv = torch.linalg.norm(j_i[out_u_idx, :, v, :], ord='fro')
                    norm_vu_plus_uv = norm_vu + norm_uv

                    layerwise_norms_same.append(norm_vv_plus_uu)
                    layerwise_norms_diff.append(norm_vu_plus_uv)

                norms_same.append(layerwise_norms_same)
                norms_diff.append(layerwise_norms_diff)

            print('Done computing sensitivity for batch...')
            break

        return norms_same, norms_diff


    def train_link_pred(self, train_loader, model, optimizer):
        model.train()

        train_losses = 0.
        train_metrics = defaultdict(float)
        num_labels = 0
        num_graphs = 0

        for data in train_loader.loader:
            data = data.to(self.device)
            y = data.y
            num_labels += y.shape[0]
            num_graphs += data.num_graphs

            npreds = (data._slice_dict['y'][1:] - data._slice_dict['y'][:-1]).to(self.device)
            nnodes = (data._slice_dict['x'][1:] - data._slice_dict['x'][:-1]).to(self.device)

            optimizer.zero_grad()
            outputs, _, _, auxloss = model(data)

            def get_pred(output_emb):
                return (output_emb[data.edge_label_index[0]] * output_emb[data.edge_label_index[1]]).sum(1)

            if type(outputs) == list:
                head_losses = [self.criterion(get_pred(output), y) for output in outputs]
                loss = torch.sum(torch.stack(head_losses))
                outputs = outputs[-1]
            else:
                loss = self.criterion(get_pred(outputs), y)
            train_losses += loss.detach() * y.shape[0]
            loss = loss + auxloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            reshaped_pred, _ = to_dense_batch(
                outputs.detach(),
                batch=data.batch,
                max_num_nodes=nnodes.max())
            all_prediction = torch.einsum('bnf,bmf->bnm', reshaped_pred, reshaped_pred)

            train_metric = self.evaluator(y, all_prediction,
                                          npreds, nnodes,
                                          data.edge_label_index)
            for k, v in train_metric.items():
                train_metrics[k] += v * data.num_graphs

        for k, v in train_metrics.items():
            train_metrics[k] = v.item() / num_graphs

        return train_losses.item() / num_labels, train_metrics

    @torch.no_grad()
    def test_link_pred(self, loader, model, scheduler, epoch=None):
        model.eval()

        val_losses = 0.
        val_metrics = defaultdict(float)
        num_labels = 0
        num_graphs = 0

        for data in loader.loader:
            data = data.to(self.device)
            y = data.y
            num_graphs += data.num_graphs
            num_labels += y.shape[0]

            npreds = (data._slice_dict['y'][1:] - data._slice_dict['y'][:-1]).to(self.device)
            nnodes = (data._slice_dict['x'][1:] - data._slice_dict['x'][:-1]).to(self.device)

            def get_pred(output_emb):
                return (output_emb[data.edge_label_index[0]] * output_emb[data.edge_label_index[1]]).sum(1)

            outputs, *_ = model(data)

            if type(outputs) == list:
                head_losses = [self.criterion(get_pred(output), y) for output in outputs]
                loss = torch.sum(torch.stack(head_losses))
                outputs = outputs[-1]
            else:
                loss = self.criterion(get_pred(outputs), y)

            val_losses += loss.detach() * y.shape[0]

            reshaped_pred, _ = to_dense_batch(
                outputs.detach(),
                batch=data.batch,
                max_num_nodes=nnodes.max())
            all_prediction = torch.einsum('bnf,bmf->bnm', reshaped_pred, reshaped_pred)

            val_metric = self.evaluator(y, all_prediction,
                                          npreds, nnodes,
                                          data.edge_label_index)
            for k, v in val_metric.items():
                val_metrics[k] += v * data.num_graphs

        for k, v in val_metrics.items():
            val_metrics[k] = v.item() / num_graphs

        if scheduler is not None:
            scheduler.step(epoch) if 'LambdaLR' in str(type(scheduler)) else scheduler.step(val_metrics[self.target_metric])
        return val_losses.item() / num_labels, val_metrics

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
