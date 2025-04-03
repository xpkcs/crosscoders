'''
These runners are separated from the other runner.py file so that when we import
from runner.py we don't need to have torch installed. Importing this file requires
torch b/c of BaselineModuleConfig.
'''


from dataclasses import dataclass, field



from crosscoders.dataclasses.autoencoders.baseline import BaselineModuleConfig
from crosscoders.dataclasses.runner import OptimizerConfig, RunnerConfig


# __all__ = ['DataRunnerConfig', 'RayDataRunnerConfig', 'SparkDataRunnerConfig']






@dataclass
class DataRunnerConfig(RunnerConfig):

    stage: str = 'data'


@dataclass
class RayDataRunnerConfig(DataRunnerConfig):
    ...

@dataclass
class SparkDataRunnerConfig(DataRunnerConfig):
    ...


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

    # batch_size : int = 25000

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

    # batch_size : int = 25000

