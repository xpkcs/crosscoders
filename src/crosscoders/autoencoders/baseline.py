

import einops
import torch

from crosscoders.abc.model import AutoencoderABC
from crosscoders.config import get_config
from crosscoders.dataclasses.configs.autoencoders import AutoencoderInitConfig
from crosscoders.dataclasses.configs.config import Config
from crosscoders.dataclasses.metrics.loss import LossMetrics

CONFIG = get_config()




class BaselineAutoencoder(AutoencoderABC):


    cfg: AutoencoderInitConfig

    def __init__(self, cfg: AutoencoderInitConfig):

        super().__init__(cfg)


        # dec
        self.W_dec = torch.nn.Parameter(torch.empty(
            self.cfg.d_coder, self.cfg.n_layers, self.cfg.d_model))

        self.b_dec = torch.nn.Parameter(torch.zeros(
            self.cfg.n_layers, self.cfg.d_model))

        # enc
        self.W_enc = torch.nn.Parameter(torch.empty(
            self.cfg.d_model, self.cfg.n_layers, self.cfg.d_coder))

        self.b_enc = torch.nn.Parameter(torch.zeros(
            self.cfg.d_coder))


        # init W_dec
        torch.nn.init.normal_(self.W_dec)
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=0, keepdim=True) * 0.1

        # torch.nn.init.kaiming_uniform_(self.W_dec)

        # init W_enc
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            'd_coder n_layers d_model -> d_model n_layers d_coder'
        ).contiguous()


    def _encode(self, x):

        x_enc = super()._encode(x)
        x_enc = torch.nn.functional.relu(x_enc)

        return x_enc


    def loss(self, y, y_hat, lambda_s) -> LossMetrics:

        error = (
            (y - y_hat).pow(2)
            .sum((-2, -1))  # over layers, d_model
            .mean()         # over tokens
        )

        l0 = (
            (self.x_enc > 0)
            .sum(-1)    # over latents
            .type_as(self.x_enc)
            .mean()     # over tokens
        )

        W_dec_norm = (
            self.W_dec
            .norm(dim=-1)   # over d_model
            .sum(-1)        # over layers
        )

        l1 = (
            (self.x_enc.abs() * W_dec_norm)
            .sum(-1)    # over latents
            .mean()     # over tokens
        )


        loss = (
            error +
            lambda_s * l1
        )


        return loss, LossMetrics(
            loss               = loss.item(),
            error              = error.item(),
            l0                 = l0.item(),
            l1                 = l1.item(),
            explained_variance = self.explained_variance(y, y_hat),
            dead_neurons       = self.dead_neurons(self.x_enc)
        )
