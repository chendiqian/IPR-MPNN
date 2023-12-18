from typing import List

from ortools.linear_solver import pywraplp
import torch


def get_solution_as_tensor(x: List[List[pywraplp.Variable]], num_nodes: int, device: torch.device = 'cpu'):
    xs = [[x[j][i].solution_value() for i in range(len(x[0]))] for j in range(num_nodes)]
    return torch.tensor(xs, device=device, dtype=torch.float32)


def get_or_assignment(scores: torch.Tensor, nnodes_list: List, k: int) -> torch.Tensor:
    device = scores.device

    nnodes, n_centroids, n_ensemble = scores.shape
    if k >= n_centroids:
        return torch.ones_like(scores, dtype=torch.float32, device=device)

    max_nnodes = max(nnodes_list)
    scores_list = torch.split(scores, nnodes_list, dim=0)
    num_graphs = len(nnodes_list)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    # we can reuse the vars and constraints
    x = [[solver.IntVar(0, 1, f'x[{j}][{i}]') for i in range(n_centroids)] for j in range(max_nnodes)]

    # create first, fill value for each graph
    objective = solver.Objective()

    # coverage
    for j in range(n_centroids):
        solver.Add(sum([x[i][j] for i in range(max_nnodes)]) >= 1)
    # k subset
    for i in range(max_nnodes):
        solver.Add(sum([x[i][j] for j in range(n_centroids)]) == k)

    mask = []
    for ig in range(num_graphs):
        mask_ig = []

        # reset the constraints
        if ig == 0:
            last_nnodes = max_nnodes
        else:
            last_nnodes = nnodes_list[ig - 1]

        # reset some ones from the constraints
        if nnodes_list[ig] > last_nnodes:
            for cons_id in range(n_centroids):
                for node_id in range(last_nnodes, nnodes_list[ig]):
                    solver.constraint(cons_id).SetCoefficient(x[node_id][cons_id], 1.)
        # remove some ones from the constraints
        elif nnodes_list[ig] < last_nnodes:
            for cons_id in range(n_centroids):
                for node_id in range(nnodes_list[ig], last_nnodes):
                    solver.constraint(cons_id).SetCoefficient(x[node_id][cons_id], 0.)

        for ens in range(n_ensemble):
            score_ens = scores_list[ig][..., ens].cpu().tolist()

            # https://github.com/google/or-tools/blob/stable/ortools/linear_solver/linear_solver.cc#L249
            objective.Clear()
            for i in range(nnodes_list[ig]):
                for j in range(n_centroids):
                    objective.SetCoefficient(x[i][j], score_ens[i][j])
            objective.SetMaximization()

            status = solver.Solve()
            assert status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE
            solution = get_solution_as_tensor(x, nnodes_list[ig], device)
            mask_ig.append(solution)
        mask_ig = torch.stack(mask_ig, dim=2)
        mask.append(mask_ig)
    mask = torch.cat(mask, dim=0)

    return mask
