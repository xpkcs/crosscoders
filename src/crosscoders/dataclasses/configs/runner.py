



from dataclasses import dataclass, field


from crosscoders.dataclasses.configs.globals import HardwareConfig
from crosscoders.abc.dataclass import DataclassABC

from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload

import torch





@dataclass(repr=False)
class ModelConfig(DataclassABC):

    CAUSALITY: Literal['acausal', 'weak', 'strict']
    LOCALITY: Literal['global', 'local', 'skip'] = 'global'

    N_LAYERS: int = 12
    # n_layers_input: int = 1
    # n_layers_predict: int = 11
    D_MODEL: int = 768
    D_CODER: int = 2048

    # hardware
    # dtype: str | torch.dtype = torch.float32
    # device: str | torch.device = 'cuda'

    HARDWARE: HardwareConfig = field(default_factory=HardwareConfig)


@dataclass(repr=False)
class OptimizerParameters(DataclassABC):

    lr: float = 1e-3
    betas: Tuple[float,float] = (.9,.999)


@dataclass(repr=False)
class OptimizerConfig(DataclassABC):

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    parameters: OptimizerParameters = field(default_factory=OptimizerParameters)


@dataclass(repr=False)
class AutoencoderLightningModuleConfig(DataclassABC):

    MODEL: ModelConfig

    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)


