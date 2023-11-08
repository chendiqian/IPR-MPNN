import argparse
import copy
import os
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from torch import optim
from tqdm import tqdm
from data.utils import Config, args_canonize

from data.get_data import get_data

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

    train_loaders, val_loaders, test_loaders = get_data(args)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # trainer = Trainer(mode=args.mode,
    #                   criterion=CRITERION_DICT[args.dataset.lower()],
    #                   device=device)
    #
    # best_val_metrics = []
    # test_metrics = []
    #
    # for _run in range(args.runs):
    #     for _fold, (train_loader, val_loader, test_loader) in enumerate(
    #             zip(train_loaders, val_loaders, test_loaders)):
    #         if args.ckpt:
    #             run_id = f'run{_run}_fold{_fold}'
    #             run_folder = os.path.join(log_folder_name, run_id)
    #             os.mkdir(run_folder)
    #
    #         model = get_model(args, device)
    #         best_model = copy.deepcopy(model.state_dict())
    #         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, min_lr=1.e-5)
    #
    #         pbar = tqdm(range(args.epoch))
    #         for epoch in pbar:
    #             train_loss, train_metric = trainer.train(train_loader, model, optimizer)
    #
    #             with torch.no_grad():
    #                 val_loss, val_metric = trainer.test(val_loader, model, scheduler)
    #
    #             if trainer.best_val_metric < val_metric:
    #                 trainer.patience = 0
    #                 trainer.best_val_metric = val_metric
    #                 best_model = copy.deepcopy(model.state_dict())
    #                 if args.ckpt:
    #                     torch.save(model.state_dict(), os.path.join(run_folder, 'best_model.pt'))
    #             else:
    #                 trainer.patience += 1
    #
    #             if trainer.patience > args.patience:
    #                 break
    #
    #             log_dict = {'train_loss': train_loss,
    #                         'val_loss': val_loss,
    #                         'train_metric': train_metric,
    #                         'val_metric': val_metric,
    #                         'lr': scheduler.optimizer.param_groups[0]["lr"]}
    #             pbar.set_postfix(log_dict)
    #             wandb.log(log_dict)
    #
    #         best_val_metrics.append(trainer.best_val_metric)
    #
    #         model.load_state_dict(best_model)
    #         with torch.no_grad():
    #             test_loss, test_metric = trainer.test(test_loader, model, None)
    #         test_metrics.append(test_metric)
    #
    #         trainer.clear_stats()
    #
    # wandb.log({
    #     'val_metrics_mean': np.mean(best_val_metrics),
    #     'val_metrics_std': np.std(best_val_metrics),
    #     'test_metrics_mean': np.mean(test_metrics),
    #     'test_metrics_std': np.std(test_metrics),
    # })


if __name__ == '__main__':
    _, args = args_parser()
    args = args_canonize(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.use_wandb else "disabled",
               config=vars(args),
               entity=args.wandb.entity)  # use your own entity

    main(args, wandb)
