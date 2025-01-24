

from dataclasses import dataclass, asdict, field
from typing import Optional
import torch



@dataclass
class HardwareConfig:

    dtype: str | torch.dtype = torch.float32
    device: str | torch.device = 'cuda'

    def asdict(self):

        return asdict(self)


@dataclass
class ExperimentConfig:

    # mode: Literal['train', 'inference']

    CONFIG_FILEPATH: str
    PROJECT_ROOT_DIR: str

    MAX_RECORDS: Optional[int] = 1000

    NUM_GPUS_ACTIVATION: float | int = 0.4
    NUM_GPUS: float | int = 1
    NUM_TRAINERS: int = 1

    HARDWARE: Optional[HardwareConfig] = field(default_factory=HardwareConfig)


@dataclass
class GlobalsConfig:

    EXPERIMENT: ExperimentConfig

    BATCH_SIZE: int
    MAX_EPOCHS: int
    MAX_TOKENS: Optional[int] = None



