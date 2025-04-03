

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

from crosscoders.dataclasses.language_model import LanguageModelConfig
from crosscoders.dataclasses.dataset import DatasetConfig
from crosscoders.dataclasses.runner.base import BatchConfig, ActivationsConfig, RunnerConfig


__all__ = ['Paths', 'Config']




@dataclass
class Paths:
    # pass

    # PROJECT_ROOT_DIR: str = str(Path(__file__).parent.parent.parent.parent)
    # LOCAL_DATA_DIR: Optional[str] = None
    # PROJECT_ROOT_DIR: str =

    # local_prefix: str = '${hydra:runtime.cwd}'
    __local_prefix : str = Path(__file__).parents[2]
    __s3_prefix    : str = 's3://${..globals.s3_bucket}'
    prefix         : str = '${ifelse:${..globals.local}, ${._Paths__local_prefix}, ${._Paths__s3_prefix}}'

    # config_path: str = MISSING
    # data_dir       : str = '${._Paths__prefix}/data'
    data_dir       : str = 'data'
    dataset_dir    : str = '${.data_dir}/${..dataset.name}/language_model=${..language_model.name}/slice=${..dataset.slice}/tag=${..dataset.tag}'
    activations_dir: str = '${.dataset_dir}/activations/'
    zarr_dir       : str = '${.dataset_dir}/zarr/'



@dataclass
class Globals:

    # ray_job  : bool = False

    local    : bool = False
    s3_bucket: str = 'crosscoders'

    seed  : int = 314159
    device: str = 'cuda'
    dtype : str = 'float32'


    # def __post_init__(self):

    #     self.dtype = getattr(torch, self.dtype)



@dataclass
class Config:

    globals: Globals = field(default_factory=Globals)

    paths: Paths = field(default_factory=Paths)

    # batch: BatchConfig = field(default_factory=BatchConfig)

    # language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)

    # activations: ActivationsConfig = field(default_factory=ActivationsConfig)

    # dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # runner: RunnerConfig = field(default_factory=RunnerConfig)

    batch: BatchConfig = MISSING

    language_model: LanguageModelConfig = MISSING

    activations: ActivationsConfig = MISSING

    dataset: DatasetConfig = MISSING

    runner: RunnerConfig = MISSING



