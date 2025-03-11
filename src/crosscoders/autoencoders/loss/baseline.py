

from crosscoders.autoencoders.loss import dead_neurons, explained_variance
from crosscoders.dataclasses.autoencoders.baseline import BaselineLossMetrics


def loss(y, y_hat, x_enc, W_dec, lambda_s) -> BaselineLossMetrics:

    error = (
        (y - y_hat).pow(2)
        .sum((-2, -1))  # over layers, d_model
        .mean()         # over tokens
    )

    l0 = (
        (x_enc > 0)
        .sum(-1)    # over latents
        .type_as(x_enc)
        .mean()     # over tokens
    )

    W_dec_norm = (
        W_dec
        .norm(dim=-1)   # over d_model
        .sum(-1)        # over layers
    )

    l1 = (
        (x_enc.abs() * W_dec_norm)
        .sum(-1)    # over latents
        .mean()     # over tokens
    )


    loss = (
        error +
        lambda_s * l1
    )


    return loss, BaselineLossMetrics(
        loss               = loss.item(),
        error              = error.item(),
        l1                 = l1.item(),
        l0                 = l0.item(),
        explained_variance = explained_variance(y, y_hat),
        dead_neurons       = dead_neurons(x_enc),
    )
