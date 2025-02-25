











from dataclasses import dataclass, field
from typing import Literal
import einops
import torch
from crosscoders.abc.dataclass import DataclassABC
from crosscoders.abc.model import AutoencoderABC
from crosscoders.constants import CONSTANTS
from crosscoders.dataclasses.configs.globals import HardwareConfig

from torch import nn

from crosscoders.dataclasses.metrics.loss import LossMetrics

from torch.nn.functional import relu, tanh





@dataclass(repr=False)
class BaselineModelConfig(DataclassABC):

    N_LAYERS: int = 4
    D_MODEL: int = 768
    D_CODER: int = 24576

    lambda_s: float = 2

    HARDWARE: HardwareConfig = field(default_factory=HardwareConfig)



class BaselineAutoencoder(AutoencoderABC):

    def __init__(self, cfg: BaselineModelConfig = BaselineModelConfig()):

        super().__init__(cfg)

        torch.manual_seed(314159)

        # kaiming uniform init
        # W_dec
        self.W_dec = torch.nn.Parameter(torch.empty
            ((self.cfg.D_CODER, self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict()))

        self.b_dec = torch.nn.Parameter(torch.zeros(
            (self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict()))

        # W_enc
        self.W_enc = torch.nn.Parameter(torch.empty
            ((self.cfg.D_MODEL, self.cfg.N_LAYERS, self.cfg.D_CODER),
            **self.cfg.HARDWARE.asdict()))

        self.b_enc = torch.nn.Parameter(torch.zeros(
            (self.cfg.D_CODER,),
            **self.cfg.HARDWARE.asdict()))


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


    def loss(self, y, y_hat, **kwargs) -> LossMetrics:

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
            kwargs.get('lambda_s', self.cfg.lambda_s) * l1
        )


        return loss, LossMetrics(
            loss               = loss.item(),
            error              = error.item(),
            l0                 = l0.item(),
            l1                 = l1.item(),
            explained_variance = self.explained_variance(y, y_hat),
            dead_neurons       = self.dead_neurons(self.x_enc)
        )
