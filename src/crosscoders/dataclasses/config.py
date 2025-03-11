

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING
import torch

from crosscoders.dataclasses.runner import RunnerConfig


__all__ = ['Paths', 'Config']




@dataclass
class Paths:
    # pass

    # PROJECT_ROOT_DIR: str = str(Path(__file__).parent.parent.parent.parent)
    # LOCAL_DATA_DIR: Optional[str] = None
    # PROJECT_ROOT_DIR: str =

    # local_prefix: str = '${hydra:runtime.cwd}'
    __local_prefix: str = Path(__file__).parents[2]
    __s3_prefix   : str = 's3://${..s3_bucket}'
    __prefix      : str = '${ifelse:${..local}, ${._Paths__local_prefix}, ${._Paths__s3_prefix}}'

    # config_path: str = MISSING
    data_dir   : str = '${._Paths__prefix}/data'
    tokens_dir : str = '${.data_dir}/${..runner.dataset.name}/language_model=${..runner.language_model.name}/slice=${..runner.dataset.slice}/tag=${..runner.dataset.prefix}/'



@dataclass
class Config:

    # stage    : str = MISSING

    local    : bool = False
    s3_bucket: str = 'crosscoders'
    paths    : Paths = field(default_factory=Paths)
    # slice    : str = '${.runner.dataset.slice}'

    seed  : int = 314159
    device: str = 'cuda'
    dtype : str = 'float32'

    runner: RunnerConfig = field(default_factory=RunnerConfig)


    def __post_init__(self):

        self.dtype = getattr(torch, self.dtype)
