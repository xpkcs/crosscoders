

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
from omegaconf import MISSING, DictConfig, OmegaConf
import numpy as np


from crosscoders.dataclasses.autoencoders.baseline import BaselineModuleConfig
from crosscoders.dataclasses.dataset import DatasetConfig


__all__ = ['RunnerConfig']




# @dataclass
# class DimensionsConfig:

#     n_layers: int = '${runner.language_model.n_layers}'
#     d_model : int = '${runner.language_model.d_model}'
#     d_coder : int = '${runner.crosscoder.model.cfg.d_coder}'


@dataclass
class ActivationsConfig:

    layers: Optional[list[int]] = '${range:${language_model.n_layers}}'
    # types: list[str] = field(default_factory=lambda: ['resid_mid', 'ln2.normalized', 'mlp_out', 'resid_post'])
    types: list[str] = field(default_factory=lambda: ['ln2.normalized', 'mlp_out'])


@dataclass
class OptimizerConfig:

    _target_: str = 'torch.optim.Adam'
    lr      : float = 2e-4
    betas   : Tuple[float,float] = (.9,.999)
    fused   : bool = True



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
class TrainingObjectiveConfig:

    input_name : str = MISSING
    output_name: str = MISSING


@dataclass
class ReconstructionTrainingObjective(TrainingObjectiveConfig):

    input_name : str = 'resid_post'
    output_name: str = 'resid_post'


@dataclass
class PredictionTrainingObjective(TrainingObjectiveConfig):

    input_name : str = 'ln2.normalized'
    output_name: str = 'mlp_out'


class JOB_TYPE_ENUM(Enum):

    true: bool = True
    false: bool = False

    ray: str = 'ray'
    glue: str = 'glue'
    # emr: str = 'emr'


@dataclass
class BatchConfig:

    type       : str = MISSING
    batch_size : int = MISSING

    # max_seq_len: int = '${ifelse:${eq:${.type}, "token"}, 1, ${language_model.n_context}}'
    max_seq_len: int = '${language_model.n_context}'

    n_tokens   : int = 1000000000 # 1B
    n_seqs: int = '${ceil:${eval:"${.n_tokens} / ${.max_seq_len}"}}'

    n_batches: int = '${ceil:${eval:"${.n_seqs} / ${.batch_size}"}}'
    # n_records: int = '${ifelse:${eq:${.type}, "token"}, ${.n_tokens}, ${.n_seqs}}'
    n_records: int | None = None




@dataclass
class BackendConfig:

    type: str = MISSING

@dataclass
class RayBackendConfig(BackendConfig):

    type: str = 'ray'

@dataclass
class SparkBackendConfig(BackendConfig):

    type: str = 'spark'


@dataclass
class DataStrategyConfig:

    # type: str = MISSING
    _target_: str = MISSING
    activations_path: str = '${paths._Paths__prefix}/${paths.activations_dir}'

    # def __post_init__(self):
    #     self.type = self.type.lower()

@dataclass
class TokensToActivationsStrategyConfig(DataStrategyConfig):

    type: str = 'TokensToActivations'

# @dataclass
# class ShuffleStrategyConfig:

#     type: str = 'Shuffle'



@dataclass
class RunnerConfig:
    pass
    # _target_: str = MISSING

@dataclass
class DataRunnerConfig(RunnerConfig):
    # _target_: str = 'crosscoders.runner.DataRunner'
    stage: str = 'data'
    strategy: DataStrategyConfig = field(default_factory=DataStrategyConfig)


@dataclass
class TrainStrategyConfig:

    # type: str = MISSING
    _target_: str = MISSING

@dataclass
class FitLoopConfig:

    # type: str = MISSING
    _target_: str = MISSING


@dataclass
class TrainRunnerConfig(RunnerConfig):
    # _target_: str = 'crosscoders.runner.DataRunner'
    stage: str = 'train'
    strategy: TrainStrategyConfig = field(default_factory=TrainStrategyConfig)
    fit_loop: FitLoopConfig = field(default_factory=FitLoopConfig)

    training_objective: TrainingObjectiveConfig = field(default_factory=TrainingObjectiveConfig)

# @dataclass
# class RunnerConfig:

#     stage: str = MISSING

#     # _target_: str = MISSING

#     # batch_size    : int = MISSING
#     # n_tokens    : int = 1000000000
#     # n_tokens_seq: int = 512 # max seq len

#     # batch: BatchConfig = field(default_factory=BatchConfig)

#     training_objective: TrainingObjectiveConfig = field(default_factory=TrainingObjectiveConfig)

#     # job: JOB_TYPE_ENUM = JOB_TYPE_ENUM.false
#     # ray_job: bool = '${globals.ray_job}'
#     # glue_job: bool = '${globals.glue_job}'


#     # dims: DimensionsConfig = field(default_factory=DimensionsConfig)
#     # dataset: DatasetConfig = field(default_factory=DatasetConfig)
#     # dataset: DatasetConfig = MISSING
#     # dataset: DatasetConfig = '${dataset}'

#     # language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)

#     backend: BackendConfig = field(default_factory=BackendConfig)



#     @classmethod
#     def from_config(cls, cfg: Optional[DictConfig] = None, **kwargs: dict) -> None:

#         if cfg is None:
#             cfg = {}
#         elif isinstance(cfg, dict):
#             cfg = cfg
#         elif isinstance(cfg, DictConfig):
#             cfg = OmegaConf.to_container(cfg, resolve=True)
#         else:
#             cfg = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


#         return cfg | kwargs


#     # def __post_init__(self):

#     #     # if issubclass(type(self.dataset), DatasetConfig):
#     #     self.dataset.slice = self.stage


# @dataclass
# class DataRunnerConfig(RunnerConfig):

#     stage: str = 'data'
#     data_strategy: DataStrategyConfig = field(default_factory=DataStrategyConfig)

# @dataclass
# class RayDataRunnerConfig(DataRunnerConfig):
#     ...

# @dataclass
# class SparkDataRunnerConfig(DataRunnerConfig):
#     ...


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


# @dataclass
# class TrainRunnerConfig(RunnerConfig):

#     stage: str = 'train'

#     # batch_size : int = 25000

#     # dataset: ActivationsDatasetConfig = field(default_factory=ActivationsDatasetConfig)


#     optimizer  : OptimizerConfig      = field(default_factory=OptimizerConfig)
#     autoencoder: BaselineModuleConfig = field(default_factory=BaselineModuleConfig)


#     # @classmethod
#     # def from_config(cls, cfg: Optional[RunnerConfig] = None, **kwargs: dict) -> None:

#     #     cfg = super().from_config(cfg, kwargs)

#     #     for k in ('dataset', 'optimizer')
#     #     if 'dataset' in cfg:
#     #         cfg['dataset'] = TokensDatasetConfig(**cfg['dataset'])
#     #     if 'language_model' in cfg:
#     #         cfg['language_model'] = LanguageModelConfig(**cfg['language_model'])



# @dataclass
# class EvalRunnerConfig(DataRunnerConfig, TrainRunnerConfig):

#     stage: str = 'eval'

#     # batch_size : int = 25000

