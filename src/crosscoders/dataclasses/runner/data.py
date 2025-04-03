'''
These runners are separated from the other runner.py file so that when we import
from runner.py we don't need to have torch installed. Importing this file requires
torch b/c of BaselineModuleConfig.
'''


from dataclasses import dataclass, field


from crosscoders.dataclasses.runner.base import RunnerConfig


__all__ = ['DataRunnerConfig', 'RayDataRunnerConfig', 'SparkDataRunnerConfig']






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

