import os
from argparse import Namespace
from functools import partial
from typing import Union, List, Optional

from ml_collections import ConfigDict
from torch.utils.data import Subset
from torch_geometric.datasets import ZINC, WebKB, LRGBDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import (Compose,
                                        AddRandomWalkPE,
                                        AddLaplacianEigenvectorPE,
                                        ToUndirected,
                                        AddRemainingSelfLoops)

from data.data_preprocess import AugmentWithPartition, AugmentWithDumbAttr
from data.planarsatpairsdataset import PlanarSATPairsDataset
from data.utils import AttributedDataLoader, get_all_split_idx, separate_data

NUM_WORKERS = 1

DATASET = (ZINC,
           WebKB,
           LRGBDataset,
           PlanarSATPairsDataset,
           GNNBenchmarkDataset,
           Subset)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    ToUndirected: 99,
    AddRemainingSelfLoops: 100,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPartition: 98,
    AugmentWithDumbAttr: 98,
}


def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if hasattr(args.encoder, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.encoder, 'lap'):
        extra_path += 'lap_'
    if (hasattr(args, 'auxloss') and hasattr(args.auxloss, 'partition')) or hasattr(args.encoder, 'partition'):
        extra_path += f'partition{args.scorer_model.num_centroids}_'
    return extra_path if len(extra_path) else None


def get_transform(args: Union[Namespace, ConfigDict]):
    transform = []
    if transform:
        return Compose(transform)
    else:
        return None


def get_pretransform(args: Union[Namespace, ConfigDict], extra_pretransforms: Optional[List] = None):
    pretransform = []
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args.encoder, 'rwse'):
        pretransform.append(AddRandomWalkPE(args.encoder.rwse.kernel, 'pestat_RWSE'))
    if hasattr(args.encoder, 'lap'):
        pretransform.append(AddLaplacianEigenvectorPE(args.encoder.lap.max_freqs, 'EigVecs', is_undirected=True))
    if (hasattr(args, 'auxloss') and hasattr(args.auxloss, 'partition')) or hasattr(args.encoder, 'partition'):
        pretransform.append(AugmentWithPartition(args.scorer_model.num_centroids))

    if pretransform:
        pretransform = sorted(pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
        return Compose(pretransform)
    else:
        return None


def get_data(args: Union[Namespace, ConfigDict], force_subset):
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    task = 'graph'
    if args.dataset.lower() == 'zinc':
        train_set, val_set, test_set, std = get_zinc(args, force_subset)
    elif args.dataset.lower().startswith('hetero'):
        train_set, val_set, test_set, std = get_heterophily(args, force_subset)
        task = 'node'
    elif args.dataset.lower().startswith('peptides'):
        train_set, val_set, test_set, std = get_lrgb(args, force_subset)
    elif args.dataset.lower() == 'exp':
        train_set, val_set, test_set, std = get_exp_dataset(args, force_subset)
    elif args.dataset.lower() == 'csl':
        train_set, val_set, test_set, std = get_CSL(args, force_subset)
    else:
        raise ValueError

    assert std is None
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

        train_loaders = [AttributedDataLoader(loader=dataloader(t), std=std, task=task) for i, t in enumerate(train_set)]
        val_loaders = [AttributedDataLoader(loader=dataloader(t), std=std, task=task) for i, t in enumerate(val_set)]
        test_loaders = [AttributedDataLoader(loader=dataloader(t), std=std, task=task) for i, t in enumerate(test_set)]
    else:  # for plots
        assert isinstance(train_set, DATASET)
        train_loaders = AttributedDataLoader(loader=dataloader(train_set), std=std, task=task)
        val_loaders = AttributedDataLoader(loader=dataloader(val_set), std=std, task=task)
        test_loaders = None

    return train_loaders, val_loaders, test_loaders


def get_zinc(args: Union[Namespace, ConfigDict], force_subset: bool):
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

    if args.debug or force_subset:
        train_set = train_set[:1]
        val_set = val_set[:1]
        test_set = test_set[:1]

    return train_set, val_set, test_set, None


def get_heterophily(args, force_subset):
    dataset_name = args.dataset.lower().split('_')[1]
    datapath = os.path.join(args.data_path, 'hetero_' + dataset_name)
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transforms = get_pretransform(args, extra_pretransforms=[ToUndirected(reduce='mean')])
    transform = get_transform(args)

    splits = {'train': [], 'val': [], 'test': []}

    folds = range(10)
    for split in ['train', 'val', 'test']:
        for fold in folds:
            dataset = WebKB(root=datapath,
                            name=dataset_name,
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


def get_lrgb(args: Union[Namespace, ConfigDict], force_subset):
    datapath = args.data_path
    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)
    pre_transform = get_pretransform(args, extra_pretransforms=None)
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
