import os
from re import split as re_split
from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
            outputs, _, _, auxloss = model(data)

            loss = self.criterion(outputs, y)
            train_losses += loss.detach() * y.shape[0]
            loss = loss + auxloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

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

            outputs, *_ = model(data)
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
        self.best_val_metric = None
        self.patience = 0


class Plotter:
    def __init__(self, device, plot_args):
        self.device = device

        if plot_args is None:
            self.plot_every = 10000000000
            return

        if hasattr(plot_args, 'plot_folder'):
            self.plot_folder = plot_args.plot_folder
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
        else:
            self.plot_folder = None

        self.plot_every = plot_args.plot_every
        self.plot_mask = hasattr(plot_args, 'mask') and plot_args.mask
        self.plot_score = hasattr(plot_args, 'score') and plot_args.score

    def __call__(self, epoch, train_loader, val_loader, model, wandb):
        if epoch % self.plot_every == 0:
            self.visualize(epoch, train_loader, val_loader, model, wandb)
        else:
            return

    @torch.no_grad()
    def visualize(self, epoch, train_loader, val_loader, model, wandb):
        model.eval()

        train_data = next(iter(train_loader.loader))
        val_data = next(iter(val_loader.loader))

        data_dict = {'train': train_data.to(self.device),
                     'val': val_data.to(self.device)}

        for phase, data in data_dict.items():
            _, node_mask, scores, _ = model(data)

            # plot mask
            if self.plot_mask:
                n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape

                vmin = np.min(node_mask)
                vmax = np.max(node_mask)

                fig, axs = plt.subplots(ncols=n_samples * n_ensemble + 1,
                                        figsize=(n_centroids * n_samples * n_ensemble * 1.2, nnodes),
                                        gridspec_kw=dict(width_ratios=[1.] * n_samples * n_ensemble + [0.3]))

                for ens in range(n_ensemble):
                    for ns in range(n_samples):
                        # nnodes, n_centroids
                        mask = node_mask[ns, :, :, ens]

                        axs[ens * n_samples + ns].set_axis_off()
                        sns.heatmap(mask, cbar=False, vmin=vmin, vmax=vmax, ax=axs[ens * n_samples + ns],
                                    linewidths=0.1, linecolor='yellow')
                        axs[ens * n_samples + ns].title.set_text(f'ens{ens}, ns{ns}')

                fig.colorbar(axs[0].collections[0], cax=axs[-1])

                if self.plot_folder is not None:
                    fig.savefig(
                        os.path.join(self.plot_folder,
                                     f'mask_epoch{epoch}_{phase}.png'),
                        bbox_inches='tight')
                plt.close(fig)

                wandb.log({"plot_mask": wandb.Image(fig)}, step=epoch)

            # plot score
            if self.plot_score:
                nnodes, n_centroids, n_ensemble = scores.shape

                vmin = np.min(scores)
                vmax = np.max(scores)

                fig, axs = plt.subplots(ncols=n_ensemble + 1,
                                        figsize=(n_centroids * n_ensemble * 1.2, nnodes),
                                        gridspec_kw=dict(width_ratios=[1.] * n_ensemble + [0.3]))

                for ens in range(n_ensemble):
                    # nnodes, n_centroids
                    mask = scores[:, :, ens]

                    axs[ens].set_axis_off()
                    sns.heatmap(mask, cbar=False, vmin=vmin, vmax=vmax, ax=axs[ens],
                                linewidths=0.1, linecolor='yellow')
                    axs[ens].title.set_text(f'ens{ens}')

                fig.colorbar(axs[0].collections[0], cax=axs[-1])

                if self.plot_folder is not None:
                    path = os.path.join(self.plot_folder, f'scores_epoch{epoch}_{phase}.png')
                    fig.savefig(path, bbox_inches='tight')
                    wandb.log({"plot_score": wandb.Image(path)}, step=epoch)
                else:
                    tmp_path = f'sc_{epoch}_{phase}_{"".join(re_split(r"[ :.-]", str(datetime.now())))}.png'
                    fig.savefig(tmp_path, bbox_inches='tight')
                    wandb.log({"plot_score": wandb.Image(tmp_path)}, step=epoch)
                    os.unlink(tmp_path)

                plt.close(fig)
