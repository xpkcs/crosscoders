



from abc import abstractmethod
from crosscoders.configs.loss import LossMetrics


import torch




class LossABC:

    # def __init__(self):
    #     ...

    @abstractmethod
    def __call__(self, target: torch.Tensor, predicted: torch.Tensor, **kwargs: dict) -> LossMetrics:
        ...
