



from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import torch




class OptimizerParameters(NamedTuple):

    lr: float = 1e-3
    betas: Tuple[float,float] = (.9,.999)


@dataclass
class OptimizerConfig:

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    parameters: OptimizerParameters = field(default_factory=OptimizerParameters)
