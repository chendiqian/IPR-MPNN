import os
from datetime import datetime
from re import split as re_split

import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch_geometric
import warnings
from matplotlib import pyplot as plt
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, to_undirected


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
        self.plot_graph = hasattr(plot_args, 'graph') and plot_args.graph
        self.total_resistance = hasattr(plot_args, 'total_resistance') and plot_args.total_resistance

        for key in plot_args:
            if key not in ['mask', 'score', 'graph', 'plot_folder', 'plot_every']:
                warnings.warn(f'Key {key} is not a valid plotting option.')

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

        data_dict = {
            'train': train_data.to(self.device),
            'val': val_data.to(self.device)
        }

        for phase, data in data_dict.items():
            _, node_mask, scores, _ = model(data, True)

            # plot mask
            if self.plot_mask:
                n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape

                vmin = np.min(node_mask)
                vmax = np.max(node_mask)

                # vmin = 0
                # vmax = np.max(node_mask.sum(1))

                fig, axs = plt.subplots(ncols=n_samples * n_ensemble + 1,
                                        figsize=(n_centroids * n_samples * n_ensemble * 1.2, nnodes),
                                        # figsize=(n_centroids * n_samples * n_ensemble * 1.2, 1.),
                                        gridspec_kw=dict(width_ratios=[1.] * n_samples * n_ensemble + [0.3]))

                for ens in range(n_ensemble):
                    for ns in range(n_samples):
                        # nnodes, n_centroids
                        mask = node_mask[ns, :, :, ens]
                        # mask = node_mask[ns, :, :, ens].sum(0, keepdims=True)

                        axs[ens * n_samples + ns].set_axis_off()
                        sns.heatmap(mask, cbar=False, vmin=vmin, vmax=vmax, ax=axs[ens * n_samples + ns],
                                    linewidths=0.1, linecolor='yellow')
                        axs[ens * n_samples + ns].title.set_text(f'phase {phase} ens{ens}, ns{ns}')

                fig.colorbar(axs[0].collections[0], cax=axs[-1])

                if self.plot_folder is not None:
                    path = os.path.join(self.plot_folder, f'masks_epoch{epoch}_{phase}.png')
                    fig.savefig(path, bbox_inches='tight')
                    wandb.log({f"plot_mask_phase_{phase}": wandb.Image(path)}, step=epoch)
                else:
                    tmp_path = f'msk_{epoch}_{phase}_{"".join(re_split(r"[ :.-]", str(datetime.now())))}.png'
                    fig.savefig(tmp_path, bbox_inches='tight')
                    wandb.log({f"plot_mask_phase_{phase}": wandb.Image(tmp_path)}, step=epoch)
                    os.unlink(tmp_path)

                plt.close(fig)

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
                    axs[ens].title.set_text(f'phase {phase} ens{ens}')

                fig.colorbar(axs[0].collections[0], cax=axs[-1])

                if self.plot_folder is not None:
                    path = os.path.join(self.plot_folder, f'scores_epoch{epoch}_{phase}.png')
                    fig.savefig(path, bbox_inches='tight')
                    wandb.log({f"plot_score_phase_{phase}": wandb.Image(path)}, step=epoch)
                else:
                    tmp_path = f'sc_{epoch}_{phase}_{"".join(re_split(r"[ :.-]", str(datetime.now())))}.png'
                    fig.savefig(tmp_path, bbox_inches='tight')
                    wandb.log({f"plot_score_phase_{phase}": wandb.Image(tmp_path)}, step=epoch)
                    os.unlink(tmp_path)

                plt.close(fig)

            if phase == 'val':
                if self.total_resistance:
                    
                    def compute_total_resistance(graph):
                        graph_nx = to_networkx(graph, to_undirected=True)
                        lap = torch.tensor(nx.laplacian_matrix(graph_nx).todense().astype('float'))
                        pinv = torch.linalg.pinv(lap, hermitian=True)
                        pinv_diag = torch.diagonal(pinv)
                        resistance_matrix = pinv_diag.unsqueeze(0) + pinv_diag.unsqueeze(1) - 2 * pinv
                        total_resistance = resistance_matrix.sum()
                        return total_resistance

                    n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape
                    graphs = Batch.to_data_list(data)
                    rewired_ds_resistance = []
                    initial_ds_resistance = []
                    print('Computing total resistance...')
                    for g in graphs:
                        initial_total_resistance = compute_total_resistance(g)
                        initial_ds_resistance.append(initial_total_resistance.item())

                        mask_idx = node_mask[0, :, :, 0] #doesn't support multiple samples and ensembles
                        updated_edge_index = g.edge_index.clone()
                        updated_node_list = g.x.clone()
                        # print(updated_edge_index.shape)
                        for cluster in range(mask_idx.shape[1]):
                            mask = torch.tensor(mask_idx[:, cluster])
                            nodes_connected_to_cluster = torch.where(mask)[0]
                            new_node_idx = g.edge_index.max() + cluster + 1
                            new_edges = torch.stack([torch.full_like(nodes_connected_to_cluster, new_node_idx), nodes_connected_to_cluster])
                            updated_edge_index = torch.cat([updated_edge_index, to_undirected(new_edges).to(g.edge_index.device)], dim=1)
                            new_random_node = torch.rand((1, g.x.shape[1]), device=g.x.device)
                            updated_node_list = torch.cat([updated_node_list, new_random_node], dim=0)
                        
                        # print(updated_edge_index.shape)
                        # print(updated_node_list.shape)
                        g_updated = torch_geometric.data.Data(x=updated_node_list, edge_index=updated_edge_index)
                        # g_nx = to_networkx(g_updated, to_undirected=True)
                        # lap = torch.tensor(nx.laplacian_matrix(g_nx).todense().astype('float'))
                        # pinv = torch.linalg.pinv(lap, hermitian=True)
                        # pinv_diag = torch.diagonal(pinv)
                        # resistance_matrix = pinv_diag.unsqueeze(0) + pinv_diag.unsqueeze(1) - 2 * pinv
                        # total_resistance = resistance_matrix.sum()
                        rewired_total_resistance = compute_total_resistance(g_updated)
                        rewired_ds_resistance.append(rewired_total_resistance.item())
                    
                    print(f'Initial total resistance: {sum(initial_ds_resistance)}')
                    print(f'Rewired total resistance: {sum(rewired_ds_resistance)}')
                    wandb.log({f"initial_total_resistance_{phase}": sum(initial_ds_resistance)}, step=epoch)
                    wandb.log({f"rewired_total_resistance_{phase}": sum(rewired_ds_resistance)}, step=epoch)


            if self.plot_graph:
                n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape

                graphs = Batch.to_data_list(data)
                g = graphs[0]
                g_nx = to_networkx(g, to_undirected=True)

                node_mask = node_mask > 0.
                if node_mask.sum(2).max() == 1:
                    # 1 cluster per node
                    # n_samples, nnodes, n_ensemble
                    mask = np.argmax(node_mask, axis=2)

                    fig, axs = plt.subplots(ncols=n_samples,
                                            nrows=n_ensemble,
                                            figsize=(n_samples * 5, n_ensemble * 5),
                                            squeeze=False)

                    for ens in range(n_ensemble):
                        for ns in range(n_samples):
                            axs[ens, ns].set_axis_off()
                            nx.draw_kamada_kawai(g_nx,
                                                 node_color=mask[ns, :, ens],
                                                 ax=axs[ens, ns],
                                                 node_size=4500 // g.num_nodes)  # empirical number
                            axs[ens, ns].title.set_text(f'phase {phase}, ens{ens}, ns{ns}')
                else:
                    # more than 1 cluster per node
                    fig, axs = plt.subplots(ncols=n_centroids,
                                            nrows=n_ensemble * n_samples,
                                            figsize=(n_centroids * 5, n_ensemble * n_samples * 5),
                                            squeeze=False)

                    for ens in range(n_ensemble):
                        for ns in range(n_samples):
                            for kl in range(n_centroids):
                                row_id = ens * n_samples + ns
                                axs[row_id, kl].set_axis_off()
                                mask = np.array(['w'] * g.num_nodes, dtype=object)
                                mask[node_mask[ns, :, kl, ens]] = 'k'
                                nx.draw_kamada_kawai(g_nx,
                                                     node_color=mask,
                                                     edgecolors='k',
                                                     ax=axs[row_id, kl],
                                                     node_size=4500 // g.num_nodes)  # empirical number
                                axs[row_id, kl].title.set_text(f'phase {phase}, ens{ens}, ns{ns}, centroid{kl}')

                if self.plot_folder is not None:
                    fig.savefig(
                        os.path.join(self.plot_folder,
                                     f'graphs_epoch{epoch}_{phase}.png'),
                        bbox_inches='tight')
                plt.close(fig)

                wandb.log({f"plot_graph_phase_{phase}": wandb.Image(fig)}, step=epoch)
