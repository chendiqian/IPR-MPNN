import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


def pre_proc(y1, y2):
    if len(y1.shape) == 1:
        y1 = y1[:, None]
    if len(y2.shape) == 1:
        y2 = y2[:, None]
    if isinstance(y1, torch.Tensor):
        y1 = y1.detach().cpu().numpy()
    if isinstance(y2, torch.Tensor):
        y2 = y2.detach().cpu().numpy()
    return y1, y2


def eval_rocauc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        compute ROC-AUC averaged across tasks
    """
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.any(y_true[:, i] == 1) and np.any(y_true[:, i] == 0):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def eval_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    eval accuracy (potentially multi task)

    :param y_true:
    :param y_pred:
    :return:
    """
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)


def eval_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mae_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        mae_list.append(np.abs(y_true[is_labeled, i] - y_pred[is_labeled, i]).mean())

    return sum(mae_list) / len(mae_list)

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
        From:
        https://github.com/XiaoxinHe/Graph-MLPMixer/blob/48cd68f9e92a7ecbf15aea0baf22f6f338b2030e/train/peptides_func.py
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


def eval_F1macro(y_true: np.ndarray, y_pred: np.ndarray):
    f1s = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        f1 = f1_score(y_true[is_labeled, i], y_pred[is_labeled, i], average='macro')
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def _eval_mrr(y_pred_pos, y_pred_neg, type_info):
    """ Compute Hits@k and Mean Reciprocal Rank (MRR).

    Implementation from OGB:
    https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

    Args:
        y_pred_neg: array with shape (batch size, num_entities_neg).
        y_pred_pos: array with shape (batch size, )
    """

    if type_info == 'torch':
        y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
        argsort = torch.argsort(y_pred, dim=1, descending=True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        # hits1_list = (ranking_list <= 1).to(torch.float)
        # hits3_list = (ranking_list <= 3).to(torch.float)
        # hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1. / ranking_list.to(torch.float)

        # return {f'hits@1{suffix}_list': hits1_list,
        #         f'hits@3{suffix}_list': hits3_list,
        #         f'hits@10{suffix}_list': hits10_list,
        #         f'mrr{suffix}_list': mrr_list}
        return mrr_list.mean().item()
    else:
        y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                axis=1)
        argsort = np.argsort(-y_pred, axis=1)
        ranking_list = (argsort == 0).nonzero()
        ranking_list = ranking_list[1] + 1
        # hits1_list = (ranking_list <= 1).astype(np.float32)
        # hits3_list = (ranking_list <= 3).astype(np.float32)
        # hits10_list = (ranking_list <= 10).astype(np.float32)
        mrr_list = 1. / ranking_list.astype(np.float32)

        # return {'hits@1_list': hits1_list,
        #         'hits@3_list': hits3_list,
        #         'hits@10_list': hits10_list,
        #         'mrr_list': mrr_list}
        return mrr_list.mean().item()


def eval_mrr(y_true: torch.Tensor,
             y_pred: torch.Tensor,
             npreds: torch.Tensor,
             nnodes: torch.Tensor,
             edge_label_idx: torch.Tensor):
    device = y_true.device
    y_true = torch.split(y_true, npreds.cpu().tolist(), dim=0)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = edge_label_idx - offset[None]
    split_edge_label_idx_list = torch.split(edge_label_idx, npreds.cpu().tolist(), dim=1)

    mrr_list = []
    for pred, truth, edge_label_idx, nnode in zip(y_pred, y_true, split_edge_label_idx_list, nnodes):
        pred = pred[:nnode, :nnode]

        pos_edge_index = edge_label_idx[:, truth.squeeze() == 1]
        num_pos_edges = pos_edge_index.shape[1]

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]

        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, nnode], dtype=torch.bool, device=device)
            neg_mask[torch.arange(num_pos_edges, device=device), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            mrr = _eval_mrr(pred_pos, pred_neg, 'torch')
        else:
            # Return empty stats.
            mrr = _eval_mrr(pred_pos, pred_pos, 'torch')

        mrr_list.append(mrr)

    return np.mean(mrr_list)


def eval_mrr_batch(y_true: torch.Tensor,
                   y_pred: torch.Tensor,
                   npreds: torch.Tensor,
                   nnodes: torch.Tensor,
                   edge_label_idx: torch.Tensor):
    device = y_true.device
    num_graphs = len(nnodes)

    offset = torch.cat([nnodes.new_zeros(1), torch.cumsum(nnodes, dim=0)[:-1]])
    offset = torch.repeat_interleave(offset, npreds)
    edge_label_idx = edge_label_idx - offset[None]

    # common tensors
    arange_num_graphs = torch.arange(num_graphs, device=device)

    edge_batch_index = torch.repeat_interleave(arange_num_graphs, npreds)

    # get positive edges
    pos_edge_index = edge_label_idx[:, y_true == 1]
    num_pos_edges = scatter(y_true.long(), edge_batch_index, dim=0, reduce='sum')
    pos_edge_batch_index = edge_batch_index[y_true == 1]
    assert num_pos_edges.min() > 0
    pred_pos = y_pred[pos_edge_batch_index, pos_edge_index[0], pos_edge_index[1]]

    # get negative edges: npreds * (nnodes - 1)
    neg_mask = torch.ones(num_graphs, num_pos_edges.max(), nnodes.max(), dtype=torch.bool, device=device)
    neg_mask[arange_num_graphs.repeat_interleave(nnodes.max() - nnodes), :,
             torch.cat([torch.arange(n, nnodes.max(), device=device) for n in nnodes])] = False

    _, real_edge_mask = to_dense_batch(pos_edge_index[0],
                                       torch.repeat_interleave(arange_num_graphs, num_pos_edges))
    fake_edge_idx = (~real_edge_mask).nonzero().t()
    neg_mask[fake_edge_idx[0], fake_edge_idx[1], :] = False
    cat_arange_pos_edges = torch.cat([torch.arange(n, device=device) for n in num_pos_edges], dim=0)
    neg_mask[pos_edge_batch_index, cat_arange_pos_edges, pos_edge_index[1]] = False
    pred_neg = y_pred[pos_edge_batch_index, pos_edge_index[0]][neg_mask[pos_edge_batch_index, cat_arange_pos_edges]]

    # commen tensors
    list_arange_posedge = [torch.arange(num_pos_edges[i], device=device) for i in range(num_graphs)]

    # construct a concatenated matrix
    concat_edges = torch.ones(num_graphs, num_pos_edges.max(), nnodes.max(), device=device) * -1.e10
    # fill in negative edges
    idx0 = torch.repeat_interleave(arange_num_graphs, num_pos_edges * (nnodes - 1))
    idx1 = torch.hstack([list_arange_posedge[i].repeat_interleave(nnodes[i] - 1) for i in range(num_graphs)])
    idx2 = torch.hstack([torch.arange(1, nnodes[i], device=device).repeat(num_pos_edges[i]) for i in range(num_graphs)])
    concat_edges[idx0, idx1, idx2] = pred_neg
    # fill in positive edges
    idx0 = torch.repeat_interleave(arange_num_graphs, num_pos_edges)
    idx1 = torch.cat(list_arange_posedge)
    concat_edges[idx0, idx1, 0] = pred_pos

    # predict
    argsort = torch.argsort(concat_edges, dim=-1, descending=True)
    ranking_list = argsort == 0.
    batch_mmr = scatter(1 / (ranking_list[real_edge_mask].nonzero()[:, -1] + 1), idx0, reduce='mean', dim=0)

    return batch_mmr.mean()


class Evaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self,
                 y_true: torch.Tensor,
                 y_pred: torch.Tensor,
                 npreds: torch.Tensor = None,
                 nnodes: torch.Tensor = None,
                 edge_label_idx: torch.Tensor = None):
        if self.task_type == 'rocauc':
            func = eval_rocauc
        elif self.task_type == 'rmse':
            func = eval_rmse
        elif self.task_type == 'acc':
            if y_pred.shape[1] == 1:
                # binary
                y_pred = (y_pred > 0.).to(torch.int)
            else:
                if y_true.dim() == 1 or y_true.shape[1] == 1:
                    # multi class
                    y_pred = torch.argmax(y_pred, dim=1)
                else:
                    # multi label
                    raise NotImplementedError
            func = eval_acc
        elif self.task_type == 'f1_macro':
            assert y_pred.shape[1] > 1, "assumed not binary"
            y_pred = torch.argmax(y_pred, dim=1)
            func = eval_F1macro
        elif self.task_type == 'mae':
            func = eval_mae
        elif self.task_type == 'ap':
            func = eval_ap
        elif self.task_type == 'mrr':
            return eval_mrr_batch(y_true, y_pred, npreds, nnodes, edge_label_idx)
        else:
            raise ValueError(f"Unexpected task type {self.task_type}")

        y_true, y_pred = pre_proc(y_true, y_pred)
        metric = func(y_true, y_pred)
        return metric
