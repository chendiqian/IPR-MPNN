import argparse
import copy
import os
from datetime import datetime
import logging

import numpy as np
import torch
import wandb
import yaml
from torch import optim
from tqdm import tqdm

from data.const import CRITERION_DICT, TASK_TYPE_DICT, SCHEDULER_MODE
from data.metrics import Evaluator
from data.get_data import get_data
from data.utils import IsBetter
from data.utils import Config, args_canonize
from models.get_model import get_model
from trainer import Trainer


def args_parser():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    config = Config()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    return args, config


def main(args, wandb):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exp_time = "".join(str(datetime.now()).split(":"))
        log_folder_name = f'logs/{args.dataset}_{args.model}_{exp_time}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    logging.basicConfig(level=logging.INFO)
    train_loaders, val_loaders, test_loaders = get_data(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(criterion=CRITERION_DICT[args.dataset.lower()],
                      evaluator=Evaluator(CRITERION_DICT[args.dataset.lower()]),
                      device=device)
    comparison = IsBetter(TASK_TYPE_DICT[args.dataset.lower()])

    best_val_metrics = [[] for _ in range(args.num_runs)]
    test_metrics = [[] for _ in range(args.num_runs)]
    test_metrics_ensemble = [[] for _ in range(args.num_runs)]

    for _run in range(args.num_runs):
        for _fold, (train_loader, val_loader, test_loader) in enumerate(
                zip(train_loaders, val_loaders, test_loaders)):
            if args.ckpt:
                run_id = f'run{_run}_fold{_fold}'
                run_folder = os.path.join(log_folder_name, run_id)
                os.mkdir(run_folder)

            logging.info(f'===========starting {_run}th run, {_fold}th fold=================')
            model = get_model(args, device)
            best_model = copy.deepcopy(model.state_dict())
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=SCHEDULER_MODE[TASK_TYPE_DICT[args.dataset.lower()]],
                                                             factor=0.5, patience=50, min_lr=1.e-5)

            pbar = tqdm(range(args.max_epoch))
            for epoch in pbar:
                train_loss, train_metric = trainer.train(train_loader, model, optimizer)

                with torch.no_grad():
                    val_loss, val_metric = trainer.test(val_loader, model, scheduler)

                is_better, the_better = comparison(val_metric, trainer.best_val_metric)
                if is_better:
                    trainer.patience = 0
                    trainer.best_val_metric = the_better
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(run_folder, 'best_model.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > args.patience:
                    break

                log_dict = {'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_metric': train_metric,
                            'val_metric': val_metric,
                            'lr': scheduler.optimizer.param_groups[0]["lr"]}
                pbar.set_postfix(log_dict)
                wandb.log(log_dict)

            model.load_state_dict(best_model)
            with torch.no_grad():
                test_loss, test_metric = trainer.test(test_loader, model, None)
            test_metrics[_run].append(test_metric)
            best_val_metrics[_run].append(trainer.best_val_metric)

            logging.info(f'Best val metric: {trainer.best_val_metric}')
            logging.info(f'test metric: {test_metric}')
            # logging.info(f'test metric ensemble: {test_metric_ensemble}')

            trainer.clear_stats()

    test_metrics = np.array(test_metrics)
    best_val_metrics = np.array(best_val_metrics)
    # test_metrics_ensemble = np.array(test_metrics_ensemble)

    wandb.run.summary['best_val_metric'] = np.mean(best_val_metrics)
    wandb.run.summary['best_val_metric_std'] = np.std(best_val_metrics)
    wandb.run.summary['test_metric'] = np.mean(test_metrics)
    wandb.run.summary['test_metric_std'] = np.std(test_metrics)
    # wandb.run.summary['test_metric_ensemble'] = np.mean(test_metrics_ensemble)
    # wandb.run.summary['test_metric_ensemble_std'] = np.std(test_metrics_ensemble)


if __name__ == '__main__':
    _, args = args_parser()
    args = args_canonize(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.use_wandb else "disabled",
               config=vars(args),
               entity=args.wandb.entity)  # use your own entity

    main(args, wandb)
