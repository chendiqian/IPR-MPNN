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
