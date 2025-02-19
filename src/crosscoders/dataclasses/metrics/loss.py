



# from typing import NamedTuple
from dataclasses import dataclass

import torch

from crosscoders.abc.dataclass import DataclassABC




@dataclass(repr=False)
class DeadNeuronMetrics(DataclassABC):
    
    all_tokens: float
    one_token : float
    no_token  : float



@dataclass(repr=False)
class LossMetrics(DataclassABC):

    loss: torch.Tensor       # only required output to run backward()

    error: torch.Tensor
    l1: torch.Tensor
    lp: torch.Tensor
    l0: torch.Tensor

    explained_variance: float
    dead_neurons: DeadNeuronMetrics


    # def asdict(self):
        
    #     return {_.item() for _ in vars(self)}

# class LossOutput(NamedTuple):
#     l2_loss: torch.Tensor
#     l1_loss: torch.Tensor
#     l0_loss: torch.Tensor
#     # explained_variance: torch.Tensor
#     # explained_variance_A: torch.Tensor
#     # explained_variance_B: torch.Tensor
