import os
import pdb

import torch
from collections import defaultdict
from functools import partial
from typing import List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.datasets import (ZINC, WebKB, TUDataset,
                                      LRGBDataset,
                                      GNNBenchmarkDataset,
                                      HeterophilousGraphDataset)
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import PrefetchLoader

from torch_geometric.transforms import (Compose,
                                        AddRandomWalkPE,
                                        ToUndirected,
                                        AddRemainingSelfLoops)

from data.data_preprocess import AugmentWithPartition, AugmentWithDumbAttr, AddLaplacianEigenvectorPE, RenameLabel
from data.planarsatpairsdataset import PlanarSATPairsDataset
from data.utils import Config, AttributedDataLoader, get_all_split_idx, separate_data
from data.qm9 import QM9
from data.alchemy import MyTUDataset
from data.tree_dataset import MyTreeDataset, MyLeafColorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 0

DATASET = (ZINC,
           WebKB,
           LRGBDataset,
           PlanarSATPairsDataset,
           GNNBenchmarkDataset,
           Subset,
           HeterophilousGraphDataset,
           TUDataset,
           MyTreeDataset,
           MyLeafColorDataset,
           QM9,
           PygGraphPropPredDataset)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    ToUndirected: 99,
    AddRemainingSelfLoops: 100,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPartition: 98,
    AugmentWithDumbAttr: 98,
    RenameLabel: 0,
}


def get_additional_path(args: Config):
    extra_path = ''
    if hasattr(args.encoder, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.encoder, 'lap'):
        extra_path += 'lap_'
    if (hasattr(args, 'auxloss') and hasattr(args.auxloss, 'partition')) or hasattr(args.encoder, 'partition'):
        extra_path += f'partition{args.scorer_model.num_centroids}_'
    return extra_path if len(extra_path) else None


def get_transform(args: Config):
    transform = []
    if transform:
        return Compose(transform)
    else:
        return None


def get_pretransform(args: Config, extra_pretransforms: Optional[List] = None):
    pretransform = []
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.encoder, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.encoder.rwse.kernel, 'pestat_RWSE'))
    if hasattr(args.encoder, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.encoder.lap.max_freqs, is_undirected=True))
    if (hasattr(args, 'auxloss') and hasattr(args.auxloss, 'partition')) or hasattr(args.encoder, 'partition'):
        if isinstance(args.scorer_model.num_centroids, list):
            assert len(set(args.scorer_model.num_centroids)) == 1
            num_centroids = args.scorer_model.num_centroids[0]
        else:
            num_centroids = args.scorer_model.num_centroids
        pretransform.append(AugmentWithPartition(num_centroids))

    if pretransform:
        pretransform = sorted(pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
        return Compose(pretransform)
    else:
        return None


def get_data(args: Config, force_subset):
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    task = 'graph'
    if args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, std = get_zinc(args, force_subset)
    elif args.dataset.lower() == 'alchemy':
        train_set, val_set, test_set, std = get_alchemy(args, force_subset)
    elif args.dataset.lower().startswith('tree'):
        train_set, val_set, test_set, std = get_treedataset(args, force_subset)
    elif args.dataset.lower().startswith('leafcolor'):
        train_set, val_set, test_set, std = get_leafcolordataset(args, force_subset=force_subset)
    elif args.dataset.lower() in ['cornell', 'texas', 'wisconsin']:
        train_set, val_set, test_set, std = get_webkb(args, force_subset)
        task = 'node'
    elif args.dataset.lower() in ['amazon-ratings', 'roman-empire']:
        train_set, val_set, test_set, std = get_hetero(args, force_subset)
        task = 'node'
    elif args.dataset.lower().startswith('peptides'):
        train_set, val_set, test_set, std = get_lrgb(args, force_subset)
    elif args.dataset.lower() == 'coco-sp':
        task = 'node'
        train_set, val_set, test_set, std = get_lrgb(args, force_subset)
    elif args.dataset.lower() == 'pcqm-contact':
        task = 'edge'
        train_set, val_set, test_set, std = get_lrgb(args, force_subset)
    elif args.dataset.lower() == 'exp':
        train_set, val_set, test_set, std = get_exp_dataset(args, force_subset)
    elif args.dataset.lower() == 'csl':
        train_set, val_set, test_set, std = get_CSL(args, force_subset)
    elif args.dataset in ['PROTEINS_full', 'MUTAG', 'PTC_MR', 'NCI1', 'NCI109', 'IMDB-MULTI', 'IMDB-BINARY']:
        train_set, val_set, test_set, std = get_TU(args, force_subset)
    elif args.dataset.lower() == 'qm9':
        train_set, val_set, test_set, std = get_qm9(args, force_subset)
    elif args.dataset.lower().startswith('ogb'):
        train_set, val_set, test_set, std = get_ogbg_data(args, force_subset)
    else:
        raise ValueError

    dataloader = partial(PyGDataLoader,
                         batch_size=args.batch_size,
                         shuffle=not args.debug,
                         num_workers=NUM_WORKERS)

    if not force_subset:  # potentially multi split training
        assert isinstance(train_set, (list, DATASET))
        if isinstance(train_set, DATASET):
            train_set = [train_set]
            val_set = [val_set]
            test_set = [test_set]

        train_loaders = [AttributedDataLoader(loader=PrefetchLoader(dataloader(t), device=device), std=std, task=task)
                         for i, t in enumerate(train_set)]
        val_loaders = [AttributedDataLoader(loader=PrefetchLoader(dataloader(t), device=device), std=std, task=task) for
                       i, t in enumerate(val_set)]
        test_loaders = [AttributedDataLoader(loader=PrefetchLoader(dataloader(t), device=device), std=std, task=task)
                        for i, t in enumerate(test_set)]
    else:  # for plots
        assert isinstance(train_set, DATASET)
        train_loaders = AttributedDataLoader(loader=dataloader(train_set), std=std, task=task)
        val_loaders = AttributedDataLoader(loader=dataloader(val_set), std=std, task=task)
        test_loaders = None

    return train_loaders, val_loaders, test_loaders, task


def get_ogbg_data(args: Config, force_subset: bool):
    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    # if there are specific pretransforms, create individual folders for the dataset
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      root=datapath,
                                      transform=transform,
                                      pre_transform=pre_transform)
    dataset.data.y = dataset.data.y.float()
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    if args.debug or force_subset:
        train_idx = train_idx[:1]
        val_idx = train_idx[:1]
        test_idx = test_idx[:1]

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    return train_set, val_set, test_set, None


def get_leafcolordataset(args: Config, force_subset: bool):
    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8

    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyLeafColorDataset(data_path, True, 11, depth, transform=transform, pre_transform=pre_transform)
    val_set = MyLeafColorDataset(data_path, False, 11, depth, transform=transform, pre_transform=pre_transform)
    test_set = val_set

    if args.debug or force_subset:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    args['num_classes'] = max([s.y.item() for s in train_set]) + 1

    return train_set, val_set, test_set, None


def get_treedataset(args: Config, force_subset: bool):
    depth = int(args.dataset.lower().split('_')[1])
    assert 2 <= depth <= 8

    pre_transform = get_pretransform(args)
    # pre_transform = get_pretransform(args, extra_pretransforms=[GraphCoalesce(), GraphRedirect(depth)])
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, args.dataset)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = MyTreeDataset(data_path, True, 11, depth, transform=transform, pre_transform=pre_transform)
    val_set = MyTreeDataset(data_path, False, 11, depth, transform=transform, pre_transform=pre_transform)
    # min is 1
    train_set.data.y -= 1
    val_set.data.y -= 1
    test_set = val_set

    if args.debug or force_subset:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    return train_set, val_set, test_set, None

def get_TU(args: Config, force_subset: bool):
    from torch.utils.data import random_split

    if args.dataset.startswith('IMDB'):
        extra = [AugmentWithDumbAttr()]
    else:
        extra = None
    pre_transform = get_pretransform(args, extra_pretransforms=extra)
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = TUDataset(data_path,
                        name=args.dataset,
                        transform=transform,
                        pre_transform=pre_transform)

    labels = dataset.data.y.tolist()

    if args.dataset != 'IMDB-MULTI':  # imdb-multi is multi class clf
        dataset.data.y = dataset.data.y.float().unsqueeze(1)

    # num_training = int(len(dataset) * 0.8)
    # num_val = int(len(dataset) * 0.1)
    # num_test = len(dataset) - num_val - num_training
    # train_set, val_set, test_set = random_split(dataset,
    #                                             [num_training, num_val, num_test],
    #                                             generator=torch.Generator().manual_seed(0))

    train_splits = []
    val_splits = []

    for fold_idx in range(0, 10):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_idx)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        train_splits.append(Subset(dataset, train_idx))
        val_splits.append(Subset(dataset, test_idx))

    test_splits = val_splits

    if args.debug or force_subset:
        train_splits = train_splits[0][:1]
        val_splits = val_splits[0][:1]
        test_splits = test_splits[0][:1]

    return train_splits, val_splits, test_splits, None


def get_qm9(args: Config, force_subset: bool):
    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, 'QM9')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    if hasattr(args, 'task_id'):
        if isinstance(args.task_id, int):
            assert 0 <= args.task_id <= 12
            task_id = args.task_id
        else:
            raise TypeError
    else:
        raise ValueError('task_id not specified')

    dataset_lists = defaultdict(list)

    for split in ['train', 'valid', 'test']:

        dataset = QM9(data_path,
                      split=split,
                      transform=transform,
                      pre_transform=pre_transform)

        new_data = Data()
        for k, v in dataset._data._store.items():
            if k != 'y':
                setattr(new_data, k, v)
            else:
                setattr(new_data, k, v[:, task_id:task_id + 1])

        d = QM9(data_path,
                split=split,
                return_data=False,
                transform=transform,
                pre_transform=pre_transform)
        d.data = new_data
        dataset_lists[split].append(d)

    train_set = dataset_lists['train'][0]
    val_set = dataset_lists['valid'][0]
    test_set = dataset_lists['test'][0]

    if args.debug or force_subset:
        train_set = train_set[:1]
        val_set = val_set[:1]
        test_set = test_set[:1]

    # https://github.com/radoslav11/SP-MPNN/blob/main/src/experiments/run_gr.py#L22
    norm_const = [
        0.066513725,
        0.012235489,
        0.071939046,
        0.033730778,
        0.033486113,
        0.004278493,
        0.001330901,
        0.004165489,
        0.004128926,
        0.00409976,
        0.004527465,
        0.012292586,
        0.037467458,
    ]
    std = 1. / torch.tensor(norm_const, dtype=torch.float)

    return train_set, val_set, test_set, std[task_id]


def get_alchemy(args: Config, force_subset: bool):
    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    infile = open("datasets/indices/train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("datasets/indices/val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("datasets/indices/test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    dataset = MyTUDataset(data_path,
                          name="alchemy_full",
                          index=indices_train + indices_val + indices_test,
                          transform=transform,
                          pre_transform=pre_transform)

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std

    train_set = dataset[:len(indices_train)]
    val_set = dataset[len(indices_train): len(indices_train) + len(indices_val)]
    test_set = dataset[-len(indices_test):]

    if args.debug or force_subset:
        train_set = train_set[:1]
        val_set = val_set[:1]
        test_set = test_set[:1]

    return train_set, val_set, test_set, std


def get_zinc(args: Config, force_subset: bool):
    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    data_path = os.path.join(args.data_path, 'ZINC')
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    train_set = ZINC(data_path,
                     split='train',
                     subset=True,
                     transform=transform,
                     pre_transform=pre_transform)

    val_set = ZINC(data_path,
                   split='val',
                   subset=True,
                   transform=transform,
                   pre_transform=pre_transform)

    test_set = ZINC(data_path,
                    split='test',
                    subset=True,
                    transform=transform,
                    pre_transform=pre_transform)

    for sp in [train_set, val_set, test_set]:
        sp.data.x = sp.data.x.squeeze()
        sp.data.edge_attr = sp.data.edge_attr.squeeze()
        sp.data.y = sp.data.y[:, None]

    # from torch_geometric.utils import k_hop_subgraph
    # from torch_geometric.data import Data
    # gs = []
    # g = train_set[0]
    # idx = [0, 9, 1]
    # for i in range(3):
    #     subset, edge_index, _, edge_mask = k_hop_subgraph(node_idx=idx[i], num_hops=2, edge_index=g.edge_index,
    #                    relabel_nodes=True)
    #     gs.append(Data(x=g.x[subset],
    #                    edge_index=edge_index,
    #                    edge_attr=g.edge_attr[edge_mask],
    #                    y=g.y))
    #
    # if args.debug:
    #     train_set = gs
    #     val_set = gs
    #     test_set = gs

    if args.debug or force_subset:
        train_set = train_set[:1]
        val_set = val_set[:1]
        test_set = test_set[:1]

    return train_set, val_set, test_set, None


def get_webkb(args, force_subset):
    datapath = os.path.join(args.data_path, args.dataset.lower())
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    extra_pretransforms = [ToUndirected(reduce='mean')]

    pre_transforms = get_pretransform(args, extra_pretransforms=extra_pretransforms)
    transform = get_transform(args)

    splits = {'train': [], 'val': [], 'test': []}

    folds = range(10)
    for split in ['train', 'val', 'test']:
        for fold in folds:
            dataset = WebKB(root=datapath,
                            name=args.dataset.lower(),
                            transform=transform,
                            pre_transform=pre_transforms)
            mask = getattr(dataset.data, f'{split}_mask')
            mask = mask[:, fold]
            dataset.data.y = dataset.data.y[mask]
            dataset.data.output_mask = mask
            del dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask

            splits[split].append(dataset)

    train_set, val_set, test_set = splits['train'], splits['val'], splits['test']

    if args.debug or force_subset:
        train_set = train_set[0]
        val_set = val_set[0]
        test_set = test_set[0]

    return train_set, val_set, test_set, None


def get_hetero(args, force_subset):
    datapath = os.path.join(args.data_path, args.dataset.lower())
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    extra_pretransforms = None

    pre_transforms = get_pretransform(args, extra_pretransforms=extra_pretransforms)
    transform = get_transform(args)

    splits = {'train': [], 'val': [], 'test': []}

    folds = range(10)
    for split in ['train', 'val', 'test']:
        for fold in folds:
            dataset = HeterophilousGraphDataset(root=datapath,
                                                name=args.dataset.lower(),
                                                transform=transform,
                                                pre_transform=pre_transforms)
            mask = getattr(dataset.data, f'{split}_mask')
            mask = mask[:, fold]
            dataset.data.y = dataset.data.y[mask]
            dataset.data.output_mask = mask
            del dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask

            splits[split].append(dataset)

    train_set, val_set, test_set = splits['train'], splits['val'], splits['test']

    if args.debug or force_subset:
        train_set = train_set[0]
        val_set = val_set[0]
        test_set = test_set[0]

    return train_set, val_set, test_set, None


def get_lrgb(args: Config, force_subset):
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    if args.dataset.lower() == 'pcqm-contact':
        extra_pretransforms = [RenameLabel()]
    else:
        extra_pretransforms = None
    pre_transform = get_pretransform(args, extra_pretransforms=extra_pretransforms)
    transform = get_transform(args)

    train_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='train',
                            transform=transform, pre_transform=pre_transform)
    val_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='val',
                          transform=transform, pre_transform=pre_transform)
    test_set = LRGBDataset(root=datapath, name=args.dataset.lower(), split='test',
                           transform=transform, pre_transform=pre_transform)

    if args.debug:
        train_set = train_set[:16]
        val_set = val_set[:16]
        test_set = test_set[:16]

    if force_subset:
        train_set = train_set[:1]
        val_set = val_set[:1]
        test_set = test_set[:1]

    return train_set, val_set, test_set, None


def get_exp_dataset(args, force_subset, num_fold=10):
    extra_path = get_additional_path(args)
    extra_path = extra_path if extra_path is not None else 'normal'
    pre_transform = get_pretransform(args, extra_pretransforms=[ToUndirected()])
    transform = get_transform(args)

    dataset = PlanarSATPairsDataset(os.path.join(args.data_path, args.dataset.upper()),
                                    extra_path,
                                    transform=transform,
                                    pre_transform=pre_transform)
    dataset._data.y = dataset._data.y.float()[:, None]
    dataset._data.x = dataset._data.x.squeeze()

    train_sets, val_sets, test_sets = [], [], []
    for idx in range(num_fold):
        train, val, test = separate_data(idx, dataset, num_fold)
        train_set = dataset[train]
        val_set = dataset[val]
        test_set = dataset[test]

        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    if args.debug:
        train_sets = train_sets[0]
        val_sets = val_sets[0]
        test_sets = test_sets[0]
        if force_subset:
            train_sets = train_sets[:1]
            val_sets = val_sets[:1]
            test_sets = test_sets[:1]
    else:
        if force_subset:
            train_sets = train_sets[0][:1]
            val_sets = val_sets[0][:1]
            test_sets = test_sets[0][:1]

    return train_sets, val_sets, test_sets, None


def get_CSL(args, force_subset):
    pre_transform = get_pretransform(args, extra_pretransforms=[AugmentWithDumbAttr()])
    transform = get_transform(args)

    data_path = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        data_path = os.path.join(data_path, extra_path)

    dataset = GNNBenchmarkDataset(data_path,
                                  name='CSL',
                                  transform=transform,
                                  pre_transform=pre_transform)

    splits = get_all_split_idx(dataset)

    train_sets = [Subset(dataset, splits['train'][i]) for i in range(5)]
    val_sets = [Subset(dataset, splits['val'][i]) for i in range(5)]
    test_sets = [Subset(dataset, splits['test'][i]) for i in range(5)]

    if args.debug:
        train_sets = train_sets[0]
        val_sets = val_sets[0]
        test_sets = test_sets[0]
        if force_subset:
            train_sets = train_sets[:1]
            val_sets = val_sets[:1]
            test_sets = test_sets[:1]
    else:
        if force_subset:
            train_sets = train_sets[0][:1]
            val_sets = val_sets[0][:1]
            test_sets = test_sets[0][:1]

    return train_sets, val_sets, test_sets, None
