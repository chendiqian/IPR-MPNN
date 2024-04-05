import hashlib
import json
import os
from ast import literal_eval
from collections import namedtuple, defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from multimethod import multimethod
from sklearn.model_selection import StratifiedKFold, train_test_split

AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'std',
        'task',
    ])

def create_nested_dict(flat_dict):
    nested_dict = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(".")
        d = nested_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return nested_dict

class Config(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            extension = os.path.splitext(fpath)[1]
            while os.path.dirname(fpath) != fpath:
                fpath = os.path.dirname(fpath)
                fpaths.append(os.path.join(fpath, 'default' + extension))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath) as f:
                    self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Config):
                    self[key] = Config()
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith('--'):
                opt = opt[2:]
            if '=' in opt:
                key, value = opt.split('=', 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split('.')
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Config())
            current[subkeys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.to_dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Config):
                seperator = '\n'
            else:
                seperator = ' '
            text = key + ':' + seperator + str(value)
            lines = text.split('\n')
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (' ' * 2) + line
            texts.extend(lines)
        return '\n'.join(texts)


def args_canonize(args: Union[Config, Dict]):
    for k, v in args.items():
        if isinstance(v, Union[Config, Dict]):
            args[k] = args_canonize(v)
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
            elif v.lower() == 'none':
                args[k] = None
    return args


def args_unify(args: Config):
    if hasattr(args, 'scorer_model') and args.scorer_model is not None and hasattr(args, 'sampler') and args.sampler is not None:
        if isinstance(args.scorer_model.num_centroids, int):
            assert args.sampler.sample_k <= args.scorer_model.num_centroids
            args.scorer_model.num_centroids = [args.scorer_model.num_centroids] * args.sampler.num_ensemble
        elif isinstance(args.scorer_model.num_centroids, str):
            num_centroids = eval(args.scorer_model.num_centroids)
            assert isinstance(num_centroids, list)
            assert args.sampler.sample_k <= min(num_centroids)
            args.scorer_model.num_centroids = sorted(num_centroids)
            args.sampler.num_ensemble = len(num_centroids)
        else:
            raise TypeError
    return args


class IsBetter:
    """
    A comparator for different metrics, to unify >= and <=

    """
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self, val1: float, val2: Optional[float]) -> Tuple[bool, float]:
        if val2 is None:
            return True, val1

        if self.task_type in ['rmse', 'mae']:
            better = val1 < val2
            the_better = val1 if better else val2
            return better, the_better
        elif self.task_type in ['rocauc', 'acc', 'f1_macro', 'ap', 'mrr', 'mrr_self_filtered']:
            better = val1 > val2
            the_better = val1 if better else val2
            return better, the_better
        else:
            raise ValueError


def separate_data(fold_idx, dataset, num_folds):
    assert 0 <= fold_idx < num_folds, f"fold_idx must be from 0 to {num_folds - 1}."
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    labels = dataset._data.y.numpy()
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return torch.tensor(train_idx), torch.tensor(test_idx), torch.tensor(test_idx)


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 3:1:1
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 5 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 5 fold have unique test set.
    """

    k_splits = 5
    all_idx = defaultdict(list)

    cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=0)
    labels = dataset.data.y.squeeze().numpy()

    for indexes in cross_val_fold.split(labels, labels):
        remain_index, test_index = indexes[0], indexes[1]

        # Gets final 'train' and 'val'
        train, val, _, _ = train_test_split(remain_index,
                                             range(len(remain_index)),
                                             test_size=0.25,
                                             stratify=labels[remain_index])
        all_idx['train'].append(train)
        all_idx['val'].append(val)
        all_idx['test'].append(test_index)

    return all_idx


def weighted_cross_entropy(pred, y):
    """Weighted cross-entropy for unbalanced classes.
    """

    # calculating label weights for weighted loss computation
    V = y.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(y)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(y)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, y, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, y.float(),
                                                  weight=weight[y])
        return loss
