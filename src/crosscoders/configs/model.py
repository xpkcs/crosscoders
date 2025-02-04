



from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, overload


from crosscoders.configs import HardwareConfig
from crosscoders.abc import DataclassABC




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

