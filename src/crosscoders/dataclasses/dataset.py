

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from crosscoders.dataclasses.datasource import DatasourceConfig





@dataclass
class DatasetConfig:

    # _target_: str = MISSING
    name: str = MISSING
    tag: str = ''

    # batch_size: int = '${batch.batch_size}'
    # n_tokens: int = '${batch.n_tokens}'
    # n_batches: int = '${batch.n_batches}'

    slice: str = '${ifelse:${eq:${runner.stage}, "eval"}, "validation", "train"}'
    activations_dir: str = '${paths.activations_dir}'

    datasource: DatasourceConfig = field(default_factory=DatasourceConfig)



@dataclass
class TinyStoriesDatasetConfig(DatasetConfig):

    name: str = 'tiny-stories-v1'

    org: str = 'roneneldan'
    repo: str = 'TinyStories'

