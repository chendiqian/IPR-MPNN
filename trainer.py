from typing import Union
import torch

class Trainer:
    def __init__(self,
                 criterion: torch.nn.modules.loss,
                 device: Union[str, torch.device]):
        super(Trainer, self).__init__()

        self.criterion = criterion
        self.device = device
        self.clear_stats()


    def train(self, train_loader, model, optimizer):
        model.train()

        corrects = 0
        total_loss = 0.
        num_nodes = 0
        for data in train_loader:
            data = data.to(self.device)
            y = data.y

            optimizer.zero_grad()
            outputs = model(data)[getattr(data, self.node_mask)]
            loss = self.criterion(outputs, y)
            loss.backward()

            optimizer.step()
            pred = torch.argmax(outputs.detach(), dim=1)
            corrects += pred.eq(y).sum()
            total_loss += loss.detach() * len(y)
            num_nodes += len(y)
        return total_loss.item() / num_nodes, corrects.item() / num_nodes

    @torch.no_grad()
    def test(self, loader, model, scheduler):
        model.eval()

        corrects = 0
        total_loss = 0
        num_nodes = 0
        for data in loader:
            data = data.to(self.device)
            y = data.y

            outputs = model(data)[getattr(data, self.node_mask)]
            loss = self.criterion(outputs, y)
            pred = torch.argmax(outputs.detach(), dim=1)
            corrects += pred.eq(y).sum()
            total_loss += loss.detach() * len(y)
            num_nodes += len(y)

        acc = corrects.item() / num_nodes
        if scheduler is not None:
            scheduler.step(acc)
        return total_loss.item() / num_nodes, acc

    def clear_stats(self):
        self.best_val_loss = 1e5
        self.best_val_metric = 0.
        self.patience = 0
