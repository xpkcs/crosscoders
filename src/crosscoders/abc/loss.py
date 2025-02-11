



from abc import abstractmethod
from crosscoders.abc.base import BaseABC
from crosscoders.dataclasses.configs.runner import LossConfig
from crosscoders.dataclasses.metrics.loss import DeadNeuronMetrics, LossMetrics


import torch




class LossABC(BaseABC):

    cfg: LossConfig


    @abstractmethod
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, **kwargs: dict) -> LossMetrics:
        ...

