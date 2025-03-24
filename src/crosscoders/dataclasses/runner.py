

from dataclasses import dataclass, field
from typing import Optional, Tuple
from omegaconf import MISSING, DictConfig, OmegaConf
import numpy as np


from crosscoders.dataclasses.autoencoders.baseline import BaselineModuleConfig


__all__ = ['RunnerConfig']




# @dataclass
# class DimensionsConfig:

#     n_layers: int = '${runner.language_model.n_layers}'
#     d_model : int = '${runner.language_model.d_model}'
#     d_coder : int = '${runner.crosscoder.model.cfg.d_coder}'


@dataclass
class LanguageModelConfig:

    name    : str = 'tiny-stories-33M'
    n_layers: int = 4
    d_model : int = 768


@dataclass
class OptimizerConfig:

    _target_: str = 'torch.optim.Adam'
    lr      : float = 2e-4
    betas   : Tuple[float,float] = (.9,.999)
    fused   : bool = True


@dataclass
class DatasetConfig:

    _target_: str = MISSING
    name: str = MISSING

    batch_size: int = '${runner.batch_size}'
    max_tokens: int = '${runner.max_tokens}'

    slice: str = '${ifelse:${eq:${runner.stage}, "eval"}, "validation", "train"}'
    activations_dir: str = '${paths.activations_dir}'
    tag: Optional[str] = ''


# @dataclass
# class TokensToActivationsDatasetConfig(DatasetConfig):

#     _target_: str = 'crosscoders.data.dataset.TokensToActivationsDataset'


# @dataclass
# class ActivationsDatasetConfig(DatasetConfig):

#     _target_: str = 'crosscoders.data.dataset.ActivationsDataset'


# @dataclass
# class TinyStoriesDatasetConfig(DatasetConfig):

#     name: str = 'roneneldan/TinyStories'
#     tag: str = 'tiny-stories-33M-1B'

# @dataclass
# class TinyStoriesTokensToActivationsDatasetConfig(TinyStoriesDatasetConfig, TokensToActivationsDatasetConfig):
#     ...



@dataclass
class TrainingObjective:

    input_name : str = 'resid_post'
    output_name: str = 'resid_post'


@dataclass
class RunnerConfig:

    _target_: str = MISSING

    stage: str = MISSING

    batch_size : int = MISSING
    max_tokens : int = 1000000000
    max_batches: int = '${ceil:${eval:"${.max_tokens} / ${.batch_size}"}}'


    task: TrainingObjective = field(default_factory=TrainingObjective)

    ray_job: bool = '${globals.ray_job}'

    # dims: DimensionsConfig = field(default_factory=DimensionsConfig)
    # dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # dataset: DatasetConfig = MISSING
    dataset: DatasetConfig = '${dataset}'

    # language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)


    @classmethod
    def from_config(cls, cfg: Optional[DictConfig] = None, **kwargs: dict) -> None:

        if cfg is None:
            cfg = {}
        elif isinstance(cfg, dict):
            cfg = cfg
        elif isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


        return cfg | kwargs


    # def __post_init__(self):

    #     # if issubclass(type(self.dataset), DatasetConfig):
    #     self.dataset.slice = self.stage


@dataclass
class DataRunnerConfig(RunnerConfig):

    stage: str = 'data'

    batch_size : int = 48

    dataset: DatasetConfig = '${dataset}'


# @dataclass
# class TokensToActivationsDataRunnerConfig(DataRunnerConfig):
#     ...

#     # _target_: str = 'crosscoders.data.runner.TokensToActivationsDataRunner'

#     # dataset: TinyStoriesTokensToActivationsDatasetConfig = field(default_factory=TinyStoriesTokensToActivationsDatasetConfig)


#     # @classmethod
#     # def from_config(cls, cfg: Optional[RunnerConfig] = None, **kwargs: dict) -> None:

#     #     cfg = super().from_config(cfg, kwargs)

#     #     if 'dataset' in cfg:    # todo: no hardcode
#     #         cfg['dataset'] = TinyStoriesTokensToActivationsDatasetConfig(**cfg['dataset'])
#     #     if 'language_model' in cfg:
#     #         cfg['language_model'] = LanguageModelConfig(**cfg['language_model'])


#     #     return cls(**cfg, **kwargs)

#     # def __post_init__(self):
#     #     self.dataset.slice = 'train'


@dataclass
class TrainRunnerConfig(RunnerConfig):

    stage: str = 'train'

    batch_size : int = 25000

    # dataset: ActivationsDatasetConfig = field(default_factory=ActivationsDatasetConfig)


    optimizer : OptimizerConfig      = field(default_factory=OptimizerConfig)
    crosscoder: BaselineModuleConfig = field(default_factory=BaselineModuleConfig)


    # @classmethod
    # def from_config(cls, cfg: Optional[RunnerConfig] = None, **kwargs: dict) -> None:

    #     cfg = super().from_config(cfg, kwargs)

    #     for k in ('dataset', 'optimizer')
    #     if 'dataset' in cfg:
    #         cfg['dataset'] = TokensDatasetConfig(**cfg['dataset'])
    #     if 'language_model' in cfg:
    #         cfg['language_model'] = LanguageModelConfig(**cfg['language_model'])



@dataclass
class EvalRunnerConfig(DataRunnerConfig, TrainRunnerConfig):

    stage: str = 'eval'

    batch_size : int = 25000

