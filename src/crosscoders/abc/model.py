



from abc import abstractmethod
# from typing import Any, Callable, List, Literal, Optional, Tuple, TypeVar, Union, overload


from crosscoders.dataclasses.configs.model import ModelConfig




class AutoencoderABC:
    """
    """

    def __init__(self, cfg: ModelConfig):

        super().__init__()

        self.cfg = cfg

    #     self._init()


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
