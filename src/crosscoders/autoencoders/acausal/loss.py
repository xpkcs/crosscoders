

import torch
import einops


from crosscoders.abc import LossABC
from crosscoders.configs import LossMetrics




class AcausalLoss(LossABC):

    def __call__(self, target: torch.Tensor, predicted: torch.Tensor, **kwargs: dict) -> LossMetrics:

        per_layer_l2_norm = torch.norm(target - predicted, dim=-1)
        reconstruction_error = einops.reduce(
            per_layer_l2_norm,
            '... n_layers -> ...',
            'sum'
        ).mean()


        feature_decoder_norms = einops.reduce(
            torch.norm(kwargs['W_dec'], dim=-1),
            'd_coder n_layers -> d_coder',
            'sum'
        )
        regularization_penalty_l1 = einops.einsum(
            kwargs['x_enc'], feature_decoder_norms,
            '... d_coder , d_coder -> ...'
        ).mean()


        regularization_penalty_l0 = 0 # TODO


        return LossMetrics(
            loss=reconstruction_error + regularization_penalty_l1,
            reconstruction_error=reconstruction_error,
            regularization_penalty_l1=regularization_penalty_l1,
            regularization_penalty_l0=regularization_penalty_l0
        )
