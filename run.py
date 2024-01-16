import argparse
import copy
import os
from re import split as re_split
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
from data.utils import Config, args_canonize, args_unify
from models.get_model import get_model
from trainer import Trainer
from visualize import Plotter


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
        exp_time = "".join(re_split(r'[ :.-]', str(datetime.now())))
        log_folder_name = f'logs/{args.dataset}_{exp_time}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    # log wandb config
    wandb.config.update(args)

    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')

    train_loaders, val_loaders, test_loaders = get_data(args, False)

    # for visualization
    plotter = Plotter(device, args.plots if hasattr(args, 'plots') else None)
    if hasattr(args, 'plots') and args.plots is not None:
        plot_train_loader, plot_val_loader, _ = get_data(args, True)
    else:
        plot_train_loader, plot_val_loader = None, None

    trainer = Trainer(criterion=CRITERION_DICT[args.dataset.lower()],
                      evaluator=Evaluator(TASK_TYPE_DICT[args.dataset.lower()]),
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

            if args.wd_params == 'downstream':
                no_wd = []
                wd = []
                for name, param in model.named_parameters():
                    if 'hetero_gnn' in name or 'inter' in name or 'intra' in name:
                        wd.append(param)
                    else:
                        no_wd.append(param)
            else:
                wd = model.parameters()
                no_wd = []

            # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer = optim.Adam([{'params': no_wd, 'weight_decay': 0.},
                                    {'params': wd, 'weight_decay': args.weight_decay}],
                                   lr=args.lr)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=SCHEDULER_MODE[TASK_TYPE_DICT[args.dataset.lower()]],
                                                             factor=0.5, patience=50, min_lr=1.e-5)

            # wandb.watch(model, log='all', log_freq=1)

            pbar = tqdm(range(1, args.max_epoch + 1))
            for epoch in pbar:
                train_loss, train_metric = trainer.train(train_loader, model, optimizer)

                with torch.no_grad():
                    val_loss, val_metric = trainer.test(val_loader, model, scheduler)

                if hasattr(args, 'log_test') and args.log_test:
                    with torch.no_grad():
                        test_loss, test_metric = trainer.test(test_loader, model, scheduler)

                with torch.no_grad():
                    plotter(epoch, plot_train_loader, plot_val_loader, model, wandb)

                is_better, the_better = comparison(val_metric, trainer.best_val_metric)
                if is_better:
                    trainer.patience = 0
                    trainer.best_val_metric = the_better
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(run_folder, 'best_model.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > args.patience and epoch > args.min_epoch:
                    break

                log_dict = {'train_loss': train_loss,
                            'val_loss': val_loss,
                            'test_loss': test_loss if (hasattr(args, 'log_test') and args.log_test) else 0.,
                            'train_metric': train_metric,
                            'val_metric': val_metric,
                            'test_metric': test_metric if (hasattr(args, 'log_test') and args.log_test) else 0.,
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
    args = args_unify(args_canonize(args))

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if hasattr(args.wandb, 'name') else None,
               mode="online" if args.wandb.use_wandb else "disabled",
               config=vars(args),
               entity=args.wandb.entity)  # use your own entity

    main(args, wandb)
