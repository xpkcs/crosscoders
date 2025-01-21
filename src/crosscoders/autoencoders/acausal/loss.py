



from typing import overload

import torch
import einops


from crosscoders.abc import LossABC
# from crosscoders.constants import DTYPE




class AcausalLoss(LossABC):

    def __call__(self, target: torch.Tensor, predicted: torch.Tensor):


        per_layer_l2_norm = einops.reduce(
            (target - predicted).pow(2),
            '... n_layers d_model -> ... n_layers',
            'sum'
        ).sqrt()
        self.reconstruction_error = einops.reduce(
            per_layer_l2_norm,
            '... n_layers -> ...',
            'sum'
        ).mean()

        self.regularization_penalty_l1 = 0 # TODO
        self.regularization_penalty_l0 = 0 # TODO


        return self.reconstruction_error, self.regularization_penalty_l1, self.regularization_penalty_l0
