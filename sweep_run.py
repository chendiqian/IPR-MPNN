import wandb
from data.utils import args_canonize, args_unify
from run import main
from ml_collections import ConfigDict


hyperparameter_defaults = {}


if __name__ == '__main__':
    wandb.init(
        config=hyperparameter_defaults,
        mode="online",
    )

    args = ConfigDict(args_unify(args_canonize(wandb.config._as_dict())))
    main(args, wandb)
