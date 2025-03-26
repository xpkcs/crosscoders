



from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class DatasetConfig:

    _target_: str = MISSING
    name: str = MISSING
    tag: str = MISSING

    batch_size: int = '${runner.batch_size}'
    max_tokens: int = '${runner.max_tokens}'

    slice: str = '${ifelse:${eq:${runner.stage}, "eval"}, "validation", "train"}'
    activations_dir: str = '${paths.activations_dir}'




@dataclass
class TokensToActivationsDatasetConfig(DatasetConfig):

    _target_: str = 'crosscoders.data.dataset.TokensToActivationsDataset'


@dataclass
class ActivationsDatasetConfig(DatasetConfig):

    _target_: str = 'crosscoders.data.dataset.ActivationsDataset'






@dataclass
class HuggingFaceConfig:

    org: str = MISSING
    repo: str = MISSING


@dataclass
class TinyStoriesDatasetConfig(DatasetConfig, HuggingFaceConfig):

    name: str = 'TinyStories'

    org: str = 'roneneldan'
    repo: str = '${name}'