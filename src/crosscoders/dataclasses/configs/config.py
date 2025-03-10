

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING
import torch

from crosscoders.dataclasses.configs.runner import RunnerConfig


__all__ = ['Paths', 'Config']




@dataclass
class Paths:
    # pass

    # PROJECT_ROOT_DIR: str = str(Path(__file__).parent.parent.parent.parent)
    # LOCAL_DATA_DIR: Optional[str] = None
    # PROJECT_ROOT_DIR: str =

    # local_prefix: str = '${hydra:runtime.cwd}'
    __local_prefix: str = Path(__file__).parents[3]
    __s3_prefix   : str = 's3://${..s3_bucket}'
    __prefix      : str = '${ifelse:${..local}, ${._Paths__local_prefix}, ${._Paths__s3_prefix}}'

    data_dir   : str = '${._Paths__prefix}/data'
    config_path: str = MISSING


@dataclass
class Config:

    mode: str = MISSING

    local    : bool = False
    s3_bucket: str = 'crosscoders'
    paths    : Paths = field(default_factory=Paths)

    seed  : int = 314159
    device: str = 'cuda'
    dtype : str = 'float32'

    runner: RunnerConfig = field(default_factory=RunnerConfig)


    def __post_init__(self):

        self.dtype = getattr(torch, self.dtype)
