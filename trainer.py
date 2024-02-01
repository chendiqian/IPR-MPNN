from typing import Union
import torch
from data.metrics import Evaluator

from torch_geometric.utils import to_dense_batch


class Trainer:
    def __init__(self,
                 task: str,
                 criterion: torch.nn.modules.loss,
                 evaluator: Evaluator,
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
        train_metric = self.evaluator(labels, preds)
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
        val_metric = self.evaluator(labels, preds)
        if scheduler is not None:
            scheduler.step(epoch) if 'LambdaLR' in str(type(scheduler)) else scheduler.step(val_metric)
        return val_losses.item() / num_instances, val_metric

    def train_link_pred(self, train_loader, model, optimizer):
        model.train()

        train_losses = 0.
        train_metrics = 0.
        num_instances = 0

        for data in train_loader.loader:
            data = data.to(self.device)
            y = data.y
            num_instances += y.shape[0]

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
            train_metrics += train_metric * y.shape[0]

        return train_losses.item() / num_instances, train_metrics.item() / num_instances

    @torch.no_grad()
    def test_link_pred(self, loader, model, scheduler, epoch=None):
        model.eval()

        val_losses = 0.
        val_metrics = 0.
        num_instances = 0

        for data in loader.loader:
            data = data.to(self.device)
            y = data.y
            num_instances += y.shape[0]

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
            val_metrics += val_metric * y.shape[0]

        val_metric = val_metrics.item() / num_instances

        if scheduler is not None:
            scheduler.step(epoch) if 'LambdaLR' in str(type(scheduler)) else scheduler.step(val_metric)
        return val_losses.item() / num_instances, val_metric

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = None
        self.patience = 0
