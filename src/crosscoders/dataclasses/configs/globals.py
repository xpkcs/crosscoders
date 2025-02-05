

from dataclasses import dataclass, asdict, field
from typing import Optional
import torch

from crosscoders.abc.dataclass import DataclassABC



@dataclass(repr=False)
class HardwareConfig(DataclassABC):

    dtype: str | torch.dtype = torch.float32
    device: str | torch.device = 'cuda'


@dataclass(repr=False)
class ExperimentConfig(DataclassABC):

    BATCH_SIZE: int
    MAX_EPOCHS: int = 1
    MAX_TOKENS: Optional[int] = None
    MAX_RECORDS: Optional[int] = None

    # NUM_GPUS_ACTIVATION: float | int = 0.4
    NUM_GPUS: float | int = 1
    NUM_TRAINERS: int = 1

    HARDWARE: Optional[HardwareConfig] = field(default_factory=HardwareConfig)


@dataclass(repr=False)
class GlobalsConfig(DataclassABC):

    PROJECT_ROOT_DIR: str
    CONFIG_FILEPATH: str
    # DATA_DIR: str

    EXPERIMENT: ExperimentConfig = field(default_factory=ExperimentConfig)



