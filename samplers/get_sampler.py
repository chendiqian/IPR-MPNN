from samplers.gumbel_scheme import GumbelSampler
from samplers.imle_scheme import IMLESampler
from samplers.simple_scheme import SIMPLESampler


def get_sampler(sampler_args, device):
    if sampler_args is None:
        return None

    if sampler_args.name == 'simple':
        return SIMPLESampler(
            sampler_args.sample_k,
            device=device,
            val_ensemble=sampler_args.val_samples,
            train_ensemble=sampler_args.train_samples
        )
    elif sampler_args.name == 'imle':
        return IMLESampler(
            sample_k=sampler_args.sample_k,
            device=device,
            train_ensemble=sampler_args.train_samples,
            val_ensemble=sampler_args.val_samples,
            noise_scale=sampler_args.noise_scale,
            beta=sampler_args.beta
        )
    elif sampler_args.name == 'gumbel':
        return GumbelSampler(
            k=sampler_args.sample_k,
            train_ensemble=sampler_args.train_samples,
            val_ensemble=sampler_args.val_samples,
            tau=sampler_args.tau
        )
    else:
        raise ValueError(f"Unexpected sampler {sampler_args.name}")
