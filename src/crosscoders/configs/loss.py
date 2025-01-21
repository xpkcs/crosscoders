



from typing import NamedTuple

import torch




class LossOutput(NamedTuple):
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    # explained_variance: torch.Tensor
    # explained_variance_A: torch.Tensor
    # explained_variance_B: torch.Tensor
