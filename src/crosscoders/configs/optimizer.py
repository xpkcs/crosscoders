



from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import torch

from crosscoders.abc import DataclassABC



@dataclass(repr=False)
class OptimizerParameters(DataclassABC):

    lr: float = 1e-3
    betas: Tuple[float,float] = (.9,.999)


@dataclass(repr=False)
class OptimizerConfig(DataclassABC):

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    parameters: OptimizerParameters = field(default_factory=OptimizerParameters)
