

import torch

from crosscoders.dataclasses.configs.autoencoders import DeadNeuronMetrics




def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:

    # (bs, nl, dm) -> (nl, dm) -> 1
    # can we assume independence to sum vars?
    variance = x.var(dim=0).sum()
    residual_variance = (x - x_hat).var(dim=0).sum()


    return (1 - (residual_variance / variance)).item()


def dead_neurons(x_enc: torch.Tensor):

    # why use any? why not count how many tokens each neuron is dead for?
    all_tokens = x_enc.all(dim=0).sum()             # fires on all tokens
    one_token  = x_enc.any(dim=0).sum()             # fires on at least one token
    no_token   = (x_enc == 0).all(dim=0).sum()      # fires on no tokens


    return DeadNeuronMetrics(*map(
        lambda _: (_ / x_enc.shape[-1]).item(),
        (all_tokens, one_token, no_token)
    ))