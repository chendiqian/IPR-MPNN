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

        self.initial_total_resistance_mean = []
        self.initial_total_resistance_std = []
        self.rewired_total_resistance_mean = []
        self.rewired_total_resistance_std = []

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

        train_data = next(iter(train_loader[0].loader))
        val_data = next(iter(val_loader[0].loader))

        rewired_ds_resistance = []
        initial_ds_resistance = []

        for train_data, val_data in zip(train_loader[0].loader, val_loader[0].loader):
        
            data_dict = {
                'train': train_data.to(self.device),
                'val': val_data.to(self.device)
            }

            for phase, data in data_dict.items():
                _, node_mask, scores, _ = model(data, True)

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
                        # also get masks batch
                        node_mask = node_mask[0, :, :, 0]
                        node_masks_list = []
                        num_graphs = data.batch.max().item() + 1  # Get the number of graphs in the batch

                        for i in range(num_graphs):
                            mask = node_mask[data.batch.cpu() == i]  # Mask out nodes belonging to graph i
                            node_masks_list.append(mask)

                        print('Computing total resistance...')
                        for g, mask in zip(graphs, node_masks_list):
                            initial_total_resistance = compute_total_resistance(g)
                            initial_ds_resistance.append(initial_total_resistance.item())

                            # mask_idx = node_mask[0, :, :, 0] #doesn't support multiple samples and ensembles
                            mask_idx = mask
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
                            
                            g_updated = torch_geometric.data.Data(x=updated_node_list, edge_index=updated_edge_index)
                            rewired_total_resistance = compute_total_resistance(g_updated)
                            rewired_ds_resistance.append(rewired_total_resistance.item())
                    
        print(f'Initial total resistance: {np.mean(initial_ds_resistance)}')
        print(f'Rewired total resistance: {np.mean(rewired_ds_resistance)}')
        wandb.log({f"initial_total_resistance_{phase}": np.mean(initial_ds_resistance)}, step=epoch)
        wandb.log({f"rewired_total_resistance_{phase}": np.mean(rewired_ds_resistance)}, step=epoch)

        self.initial_total_resistance_mean.append(np.mean(initial_ds_resistance))
        self.initial_total_resistance_std.append(np.std(initial_ds_resistance))

        self.rewired_total_resistance_mean.append(np.mean(rewired_ds_resistance))
        self.rewired_total_resistance_std.append(np.std(rewired_ds_resistance))
