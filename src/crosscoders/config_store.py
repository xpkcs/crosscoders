

# ------------------------- resolvers ------------------------- #

import numpy as np

def register_resolvers(replace=True):

    def eq(x, y):

        return x == y

    def ifelse(condition, on_if, on_else):

        return on_if if condition else on_else

    def ceil(i):

        return int(np.ceil(float(i)))


    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver('eval', eval, replace=replace)
    OmegaConf.register_new_resolver('eq', eq, replace=replace)
    OmegaConf.register_new_resolver('ifelse', ifelse, replace=replace)
    OmegaConf.register_new_resolver('ceil', ceil, replace=replace)
    OmegaConf.register_new_resolver('range', lambda n: list(range(int(n))), replace=replace)


# ------------------------- structured configs ------------------------- #

def init_config_store():

    # from crosscoders.dataclasses.args import ArgsConfig
    from crosscoders.dataclasses.config import Config
    from crosscoders.dataclasses.runner import (
        BatchConfig,
        DataRunnerConfig,
        PredictionTrainingObjective,
        RayDataRunnerConfig,
        ReconstructionTrainingObjective,
        RunnerConfig,
        SparkDataRunnerConfig,
        TrainRunnerConfig,
        EvalRunnerConfig
    )

    from crosscoders.dataclasses.language_model import TinyStories33MLanguageModelConfig
    from crosscoders.dataclasses.dataset import TinyStoriesDatasetConfig
    from crosscoders.dataclasses.datasource import HuggingFaceDatasourceConfig, S3DatasourceConfig

    from crosscoders.dataclasses.autoencoders.baseline import BaselineAutoencoderConfig
    from crosscoders.dataclasses.autoencoders.jumprelu import JumpReLUAutoencoderConfig


    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    # cs.store(name='args', node=ArgsConfig)
    cs.store(name='base_config', node=Config)
    # cs.store(name='base_runner', node=RunnerConfig)

    cs.store(group='runner', name='data', node=DataRunnerConfig)
    cs.store(group='runner', name='train', node=TrainRunnerConfig)
    cs.store(group='runner', name='eval', node=EvalRunnerConfig)
    cs.store(group='runner', name='RayDataRunner', node=RayDataRunnerConfig)
    cs.store(group='runner', name='SparkDataRunner', node=SparkDataRunnerConfig)
    cs.store(group='runner/training_objective', name='reconstruction', node=ReconstructionTrainingObjective)
    cs.store(group='runner/training_objective', name='prediction', node=PredictionTrainingObjective)

    cs.store(group='language_model', name='tiny-stories-33M', node=TinyStories33MLanguageModelConfig)
    cs.store(group='dataset', name='tiny-stories', node=TinyStoriesDatasetConfig)

    cs.store(group='dataset/datasource', name='hf', node=HuggingFaceDatasourceConfig(which='tokens'))
    cs.store(group='dataset/datasource', name='s3', node=S3DatasourceConfig(which='activations'))

    cs.store(group='batch', name='token', node=BatchConfig(type='token', max_seq_len=1))
    cs.store(group='batch', name='sequence', node=BatchConfig(type='sequence'))

    cs.store(group='crosscoder', name='baseline', node=BaselineAutoencoderConfig)
    cs.store(group='crosscoder', name='jumprelu', node=JumpReLUAutoencoderConfig)


__all__ = [
    'register_resolvers',
    'init_config_store',
]