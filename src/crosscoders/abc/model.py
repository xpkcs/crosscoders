



from abc import abstractmethod

import torch
# from typing import Any, Callable, List, Literal, Optional, Tuple, TypeVar, Union, overload


from crosscoders.abc.base import BaseABC
from crosscoders.dataclasses.configs.runner import ModelConfig




class AutoencoderABC(torch.nn.Module):

    def __init__(self, cfg: ModelConfig):

        super().__init__()

        self.cfg = cfg




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


    def forward(self, x):

        h = self._encode(x)
        x = self._decode(h)

        return x
