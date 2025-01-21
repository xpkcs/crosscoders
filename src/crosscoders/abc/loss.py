



from abc import abstractmethod


import torch




class LossABC:

    # def __init__(self):
    #     ...

    @abstractmethod
    # def __call__(self, target: torch.Tensor[DTYPE], predicted: torch.Tensor[DTYPE]):
    def __call__(self, target: torch.Tensor, predicted: torch.Tensor):
        ...
