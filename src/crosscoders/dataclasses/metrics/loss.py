



# from typing import NamedTuple
from dataclasses import dataclass

import torch

from crosscoders.abc.dataclass import DataclassABC




@dataclass(repr=False)
class LossMetrics(DataclassABC):

    loss: torch.Tensor       # only required output to run backward()

    reconstruction_error: torch.Tensor
    regularization_penalty_l1: torch.Tensor
    regularization_penalty_l0: torch.Tensor

# class LossOutput(NamedTuple):
#     l2_loss: torch.Tensor
#     l1_loss: torch.Tensor
#     l0_loss: torch.Tensor
#     # explained_variance: torch.Tensor
#     # explained_variance_A: torch.Tensor
#     # explained_variance_B: torch.Tensor
