

from typing import overload
import torch
import einops


from crosscoders.abc.loss import LossABC
from crosscoders.constants import CONSTANTS
from crosscoders.dataclasses.configs.runner import RunnerConfig
from crosscoders.dataclasses.metrics.loss import LossMetrics




class AcausalLoss(LossABC):

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, x_enc: torch.Tensor, W_dec: torch.Tensor) -> LossMetrics:

        # per_layer_l2_norm = einops.reduce(
        #     (x - x_hat).pow(2),
        #     '... n_layers d_model -> ... n_layers',
        #     'sum'
        # )
        # reconstruction_error = einops.reduce(
        #     per_layer_l2_norm,
        #     '... n_layers -> ...',
        #     'sum'
        # ).mean()

        # reconstruction_error = einops.reduce(
        #     (x - x_hat).pow(2),
        #     '... n_layers d_model -> ...',
        #     'sum'
        # ).mean()

        reconstruction_error = (x - x_hat).pow(2).sum(dim=(-2, -1)).mean()
        # reconstruction_error = (x - x_hat).pow(2).mean(-1).sum(dim=-1).mean()
        # reconstruction_error = (x - x_hat).norm(dim=-1).sum(dim=-1).mean()

        # reconstruction_error = (x - x_hat).norm(dim=-1).mean()


        # feature_decoder_norms = einops.reduce(
        #     torch.norm(kwargs['W_dec'], dim=-1),
        #     'd_coder n_layers -> d_coder',
        #     'sum'
        # )
        # l1 = einops.einsum(
        #     # kwargs['x_enc'].abs(), feature_decoder_norms,
        #     kwargs['x_enc'], feature_decoder_norms,
        #     '... d_coder , d_coder -> ...'
        # ).mean()

        # l1 = x_enc.norm(p=1, dim=-1).mean()
        l1 = (x_enc.abs() @ W_dec.norm(dim=-1).sum(dim=-1)).mean()


        l0 = (x_enc > 0).sum(-1).to(CONSTANTS.EXPERIMENT.HARDWARE.dtype).mean()


        loss = reconstruction_error + self.cfg.lambda_s * l1


        return loss, LossMetrics(
            loss=loss.item(),
            error=reconstruction_error.item(),
            l1=l1.item(),
            l0=l0.item(),
            explained_variance=self.explained_variance(x, x_hat),
            dead_neurons=self.dead_neurons(x_enc)
        )


class AcausalLossJumpReLU(LossABC):

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, x_enc: torch.Tensor, W_dec: torch.Tensor, t: torch.Tensor) -> LossMetrics:

        reconstruction_error = (x - x_hat).pow(2).sum(dim=(-2, -1)).mean()
        # reconstruction_error = (x - x_hat).pow(2).mean(-1).sum(-1).mean()

        W_dec_norm = W_dec.norm(dim=-1).sum(dim=-1)

        l1 = torch.nn.functional.tanh(self.cfg.c * x_enc.abs() * W_dec_norm).sum(-1).mean()

        lp = (torch.nn.functional.relu(torch.exp(t) - x_enc) * W_dec_norm).sum(-1).mean()

        l0 = (x_enc > 0).sum(-1).to(CONSTANTS.EXPERIMENT.HARDWARE.dtype).mean()


        # loss = reconstruction_error + self.cfg.L1_COEFFICIENT * l1
        loss = reconstruction_error + self.cfg.lambda_s * l1 + self.cfg.lambda_p * lp


        return loss, LossMetrics(
            loss=loss.item(),
            error=reconstruction_error.item(),
            l1=l1.item(),
            lp=lp.item(),
            l0=l0.item(),
            explained_variance=self.explained_variance(x, x_hat),
            dead_neurons=self.dead_neurons(x_enc)
        )
