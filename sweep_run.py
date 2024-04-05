import wandb
from data.utils import Config, args_canonize, args_unify, create_nested_dict
from run import main


hyperparameter_defaults = {}


if __name__ == '__main__':
    wandb.init(
        config=hyperparameter_defaults,
        mode="online",
    )

    wandb_config = create_nested_dict(wandb.config._as_dict())
    args = args_canonize(wandb_config)
    config = Config()
    config.update(args)
    config = args_unify(config)
    main(config, wandb)
