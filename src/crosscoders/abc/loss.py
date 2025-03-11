

from abc import abstractmethod

import torch

from crosscoders.abc.base import BaseABC




class LossABC(BaseABC):


    # cfg: LossConfig

    @abstractmethod
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, **kwargs: dict):
        ...
