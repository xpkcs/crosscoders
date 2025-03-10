



from abc import abstractmethod

import einops
import torch

from crosscoders.config import get_config
from crosscoders.dataclasses.metrics.loss import DeadNeuronMetrics

# from typing import Any, Callable, List, Literal, Optional, Tuple, TypeVar, Union, overload


# from crosscoders.dataclasses.configs.runner import ModelConfig

CONFIG = get_config()


class AutoencoderABC(torch.nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        torch.manual_seed(CONFIG.seed)



    # def __init__(self, cfg: ModelConfig):

    #     super().__init__(cfg)

    #     # self._init()


    # @abstractmethod
    # def _init(self):
    #     ...


    @abstractmethod
    def _encode(self, x):
        ...


    @abstractmethod
    def _decode(self, h):
        ...


    def _encode(self, x):

        x_enc = einops.einsum(
            x, self.W_enc,
            '... n_layers d_model , d_model n_layers d_coder -> ... d_coder'
        ) + self.b_enc


        # missing activation so that child classes can implement

        return x_enc


    def _decode(self, x_enc):

        x_dec = einops.einsum(
            x_enc, self.W_dec,
            '... d_coder , d_coder n_layers d_model -> ... n_layers d_model'
        ) + self.b_dec

        return x_dec


    def forward(self, x):

        h = self._encode(x)
        x = self._decode(h)

        self.x_enc = h

        return x


    @staticmethod
    def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:

        # (bs, nl, dm) -> (nl, dm) -> 1
        # can we assume independence to sum vars?
        variance = x.var(dim=0).sum()
        residual_variance = (x - x_hat).var(dim=0).sum()

        return (1 - (residual_variance / variance)).item()


    @staticmethod
    def dead_neurons(x_enc: torch.Tensor):

        # why use any? why not count how many tokens each neuron is dead for?
        all_tokens = x_enc.all(dim=0).sum()             # fires on all tokens
        one_token  = x_enc.any(dim=0).sum()             # fires on at least one token
        no_token   = (x_enc == 0).all(dim=0).sum()      # fires on no tokens

        return DeadNeuronMetrics(*map(lambda _: (_ / x_enc.shape[-1]).item(), (all_tokens, one_token, no_token)))
        return DeadNeuronMetrics(*map(lambda _: (_ / x_enc.shape[-1]).item(), (all_tokens, one_token, no_token)))