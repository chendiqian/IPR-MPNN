import wandb
from data.utils import Config, args_canonize, args_unify
from run import main


hyperparameter_defaults = {}


if __name__ == '__main__':
    wandb.init(
        config=hyperparameter_defaults,
        mode="online",
    )

    args = args_canonize(wandb.config._as_dict())
    config = Config()
    config.update(args)
    config = args_unify(config)
    main(config, wandb)
