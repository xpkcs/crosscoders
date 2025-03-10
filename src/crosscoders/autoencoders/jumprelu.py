
from dataclasses import dataclass, field
from crosscoders.dataclasses.metrics.loss import LossMetrics
import einops
import torch
from crosscoders.abc.dataclass import DataclassABC
from crosscoders.abc.model import AutoencoderABC
from crosscoders.constants import CONSTANTS
# from crosscoders.dataclasses.configs.globals import HardwareConfig

from torch.nn.functional import relu, tanh





@dataclass(repr=False)
class JumpReLUModelConfig(DataclassABC):

    N_LAYERS: int = 4
    D_MODEL: int = 768
    D_CODER: int = 24576

    lambda_s: float = 10
    lambda_p: float = 3e-6
    c: float = 4
    eps: float = 2

    # HARDWARE: HardwareConfig = field(default_factory=HardwareConfig)


@dataclass(repr=False)
class JumpReLULossMetrics(LossMetrics):

    lp: torch.Tensor


class JumpReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, t, eps):
        threshold = torch.exp(t)
        # dx_mask = input > threshold
        # dt_mask = ((input - threshold) / eps).abs() < 0.5
        # ctx.save_for_backward(dx_mask, dt_mask, t, torch.tensor(eps, device=input.get_device()))     # save for backward computation

        ctx.save_for_backward(input, t)     # save for backward computation
        ctx.eps = eps
        output = torch.where(input > threshold, input, 0)       # if input > exp(t), pass through

        return output


    @staticmethod
    def backward(ctx, grad_output):
        input, t = ctx.saved_tensors

        threshold = torch.exp(t)
        dx_mask = input > threshold
        dt_mask = ((input - threshold) / ctx.eps).abs() < 0.5

        # derivative wrt input: 1 if input > threshold, 0 o/w
        grad_input = grad_output * dx_mask.type_as(grad_output)

        # derivative wrt t: exp(t) if input <= threshold, 0 o/w
        grad_t = grad_output * (dt_mask.type_as(grad_output) * (-threshold / ctx.eps))

        return grad_input, grad_t, None


class JumpReLUAutoencoder(AutoencoderABC):

    def __init__(self, cfg: JumpReLUModelConfig, reconstruction: bool = False):

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


        self.t = torch.nn.Parameter(torch.zeros(
            (self.cfg.D_CODER,),
            **self.cfg.HARDWARE.asdict()))


        # init W_dec
        # torch.nn.init.uniform_(self.W_dec, a = - 1 / self.cfg.D_MODEL, b = 1 / self.cfg.D_MODEL)
        # torch.nn.init.uniform_(self.W_enc, a = - 1 / self.cfg.D_CODER, b = 1 / self.cfg.D_CODER)

        torch.nn.init.uniform_(self.W_enc, a = - 1 / self.cfg.D_MODEL, b = 1 / self.cfg.D_MODEL)
        torch.nn.init.uniform_(self.W_dec, a = - 1 / self.cfg.D_CODER, b = 1 / self.cfg.D_CODER)

        # # init W_enc
        # self.W_enc.data = einops.rearrange(
        #     self.W_dec.data.clone(),
        #     'd_coder n_layers d_model -> d_model n_layers d_coder'
        # ).contiguous()

        # if reconstruction:
        #     self.W_enc = (self.cfg.D_MODEL / self.cfg.D_CODER) * self.W_enc
        # else:
        #     torch.nn.init.uniform_(self.W_dec, a = - 1 / self.cfg.D_CODER, b = 1 / self.cfg.D_CODER)

        # TODO: b_dec, b_enc init

        self.t.data = 0.1 + self.t.data


    def _encode(self, x):

        x_enc = super()._encode(x)
        x_enc = JumpReLUFunction.apply(x_enc, self.t, self.cfg.eps)

        return x_enc


    def loss(self, y, y_hat, **kwargs) -> JumpReLULossMetrics:

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
            tanh(self.cfg.c * self.x_enc.abs() * W_dec_norm)
            .sum(-1)    # over latents
            .mean()     # over tokens
        )

        lp = (
            (relu(torch.exp(self.t) - self.x_enc) * W_dec_norm) \
            .sum(-1)    # over latents
            .mean()     # over tokens
        )

        loss = (
            error +
            kwargs.get('lambda_s', self.cfg.lambda_s) * l1 +
            kwargs.get('lambda_p', self.cfg.lambda_p) * lp
        )


        return loss, JumpReLULossMetrics(
            loss               = loss.item(),
            error              = error.item(),
            l0                 = l0.item(),
            l1                 = l1.item(),
            lp                 = lp.item(),
            explained_variance = self.explained_variance(y, y_hat),
            dead_neurons       = self.dead_neurons(self.x_enc)
        )
