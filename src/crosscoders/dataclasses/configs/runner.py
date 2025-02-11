



from dataclasses import MISSING, dataclass, field


from crosscoders.dataclasses.configs.globals import HardwareConfig
from crosscoders.abc.dataclass import DataclassABC

from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import torch



@dataclass(repr=False)
class LossConfig(DataclassABC):
    
    pass
    # L1_COEFFICIENT: float = 8e-5
    L1_COEFFICIENT: float = 1.


@dataclass(repr=False)
class AcausalLossConfig(LossConfig):

    # L1_COEFFICIENT: float = 8e-5
    L1_COEFFICIENT: float = 1.





@dataclass(repr=False)
class ModelConfig(DataclassABC):

    CAUSALITY: Literal['acausal', 'weak', 'strict']
    LOCALITY: Literal['global', 'local', 'skip'] = 'global'

    N_LAYERS: int = 12
    # n_layers_input: int = 1
    # n_layers_predict: int = 11
    D_MODEL: int = 768
    D_CODER: int = 16384

    # hardware
    # dtype: str | torch.dtype = torch.float32
    # device: str | torch.device = 'cuda'

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

    LOSS: Optional[LossConfig] = None

    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)


    def __post_init__(self):

        if self.LOSS != MISSING:
            match self.MODEL.CAUSALITY:
                case 'acausal':
                    self.LOSS = AcausalLossConfig()


