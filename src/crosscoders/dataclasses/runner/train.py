'''
These runners are separated from the other runner.py file so that when we import
from runner.py we don't need to have torch installed. Importing this file requires
torch b/c of BaselineModuleConfig.
'''


from dataclasses import dataclass, field



from crosscoders.dataclasses.autoencoders.baseline import BaselineModuleConfig
from crosscoders.dataclasses.runner.base import OptimizerConfig, RunnerConfig


__all__ = ['TrainRunnerConfig']





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

