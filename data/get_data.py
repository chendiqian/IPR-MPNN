import os
from argparse import Namespace
from functools import partial
from typing import Union, List, Optional

from ml_collections import ConfigDict
from torch_geometric.datasets import ZINC, WebKB
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE, ToUndirected, AddRemainingSelfLoops
from data.utils import AttributedDataLoader


NUM_WORKERS = 1

DATASET = (ZINC,
           WebKB,)

# sort keys, some pre_transform should be executed first
PRETRANSFORM_PRIORITY = {
    ToUndirected: 99,
    AddRemainingSelfLoops: 100,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
}

def get_additional_path(args: Union[Namespace, ConfigDict]):
    extra_path = ''
    if hasattr(args.encoder, 'rwse'):
        extra_path += 'rwse_'
    if hasattr(args.encoder, 'lap'):
        extra_path += 'lap_'
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

    train_set.data.y = train_set.data.y[:, None]
    val_set.data.y = val_set.data.y[:, None]
    test_set.data.y = test_set.data.y[:, None]

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
