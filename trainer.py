import pdb
from typing import Union
import torch

from data.metrics import Evaluator


class Trainer:
    def __init__(self,
                 criterion: torch.nn.modules.loss,
                 evaluator: Evaluator,
                 device: Union[str, torch.device]):
        super(Trainer, self).__init__()

        self.criterion = criterion
        self.evaluator = evaluator
        self.device = device
        self.clear_stats()


    def train(self, train_loader, model, optimizer):
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
            outputs = model(data)

            pdb.set_trace()

            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_losses += loss.detach() * y.shape[0]
            preds.append(outputs.detach())
            labels.append(y)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_metric = self.evaluator(labels, preds)
        return train_losses.item() / num_instances, train_metric

    @torch.no_grad()
    def test(self, loader, model, scheduler):
        model.eval()

        val_losses = 0.
        preds = []
        labels = []
        num_instances = 0

        for data in loader.loader:
            data = data.to(self.device)
            y = data.y
            num_instances += y.shape[0]

            outputs = model(data)
            loss = self.criterion(outputs, y)

            val_losses += loss.detach() * y.shape[0]
            preds.append(outputs.detach())
            labels.append(y)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        val_metric = self.evaluator(labels, preds)
        if scheduler is not None:
            scheduler.step(val_metric)
        return val_losses.item() / num_instances, val_metric

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = 0.
        self.patience = 0
