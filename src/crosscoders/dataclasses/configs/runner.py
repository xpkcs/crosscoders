



from dataclasses import MISSING, dataclass, field


from crosscoders.dataclasses.configs.globals import HardwareConfig
from crosscoders.abc.dataclass import DataclassABC

from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import torch



@dataclass(repr=False)
class LossConfig(DataclassABC):

    pass
    # L1_COEFFICIENT: float = 8e-5
    # L1_COEFFICIENT: float = 1.
    c: float = 4
    lambda_s: float = 10
    lambda_p: float = 3e-6


@dataclass(repr=False)
class ModelConfig(DataclassABC):

    CAUSALITY: Literal['acausal', 'weak', 'strict']
    LOCALITY: Literal['global', 'local', 'skip'] = 'global'

    ACTIVATION_FUNCTION: Literal['relu', 'jumprelu'] = 'jumprelu'
    eps: float = 2

    N_LAYERS: int = 12
    D_MODEL: int = 768
    D_CODER: int = 16384

    HARDWARE: HardwareConfig = field(default_factory=HardwareConfig)


@dataclass(repr=False)
class OptimizerParameters(DataclassABC):

    lr: float = 0.0004  # 1e-4
    betas: Tuple[float,float] = (.9,.999)
    fused: bool = True


@dataclass(repr=False)
class OptimizerConfig(DataclassABC):

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    parameters: OptimizerParameters = field(default_factory=OptimizerParameters)


@dataclass(repr=False)
class RunnerConfig(DataclassABC):

    MODEL: ModelConfig

    LOSS: Optional[LossConfig] = None   # will be set in post init

    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)

    INPUT_NAME: str = 'resid_post'
    OUTPUT_NAME: str = 'resid_post'

    def __post_init__(self):

        if not self.LOSS or isinstance(self.LOSS, dict):
            match self.MODEL.CAUSALITY:
                case 'acausal':
                    self.LOSS = LossConfig(**(self.LOSS if self.LOSS else {}))


