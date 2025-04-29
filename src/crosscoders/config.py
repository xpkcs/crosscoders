

import os
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np

from rich import print as printr

from crosscoders.dataclasses.dataset import TinyStoriesDatasetConfig
from crosscoders.dataclasses.datasource import HuggingFaceDatasourceConfig, LocalDatasourceConfig, S3DatasourceConfig
from crosscoders.dataclasses.language_model import TinyStories33MLanguageModelConfig




# ------------------------- resolvers ------------------------- #
def eq(x, y):

    return x == y

def ifelse(condition, on_if, on_else):

    return on_if if condition else on_else

def ceil(i):

    return int(np.ceil(float(i)))

replace = True
OmegaConf.register_new_resolver('eval', eval, replace=replace)
OmegaConf.register_new_resolver('eq', eq, replace=replace)
OmegaConf.register_new_resolver('ifelse', ifelse, replace=replace)
OmegaConf.register_new_resolver('ceil', ceil, replace=replace)
OmegaConf.register_new_resolver('range', lambda n: list(range(int(n))), replace=replace)





# ------------------------- structured configs ------------------------- #

# from crosscoders.dataclasses.args import ArgsConfig
from crosscoders.dataclasses.config import Config
from crosscoders.dataclasses.runner import BatchConfig, DataRunnerConfig, PredictionTrainingObjective, RayBackendConfig, ReconstructionTrainingObjective, RunnerConfig, TokensToActivationsStrategyConfig, TrainRunnerConfig, FitLoopConfig
from crosscoders.dataclasses.autoencoders.baseline import BaselineAutoencoderConfig
from crosscoders.dataclasses.autoencoders.jumprelu import JumpReLUAutoencoderConfig


cs = ConfigStore.instance()

cs.store(name='base_config', node=Config)
cs.store(name='base_runner', node=RunnerConfig)

cs.store(group='runner', name='data', node=DataRunnerConfig)
cs.store(group='runner', name='train.default', node=TrainRunnerConfig)
# cs.store(group='runner', name='0.0-activations', node=DataRunnerConfig(backend=RayBackendConfig(), data_strategy=TokensToActivationsStrategyConfig()))
# cs.store(group='runner', name='1.0-train', node=TrainRunnerConfig(backend=RayBackendConfig()))
# # cs.store(group='runner', name='data', node=DataRunnerConfig)
# # cs.store(group='runner', name='train', node=TrainRunnerConfig)
# # cs.store(group='runner', name='eval', node=EvalRunnerConfig)
# cs.store(group='crosscoder', name='baseline', node=BaselineAutoencoderConfig)
# cs.store(group='crosscoder', name='jumprelu', node=JumpReLUAutoencoderConfig)
# # cs.store(name='args', node=ArgsConfig)


# # cs.store(group='runner', name='RayDataRunner', node=RayDataRunnerConfig)
# # cs.store(group='runner', name='SparkDataRunner', node=SparkDataRunnerConfig)
cs.store(group='runner/training_objective', name='reconstruction', node=ReconstructionTrainingObjective)
cs.store(group='runner/training_objective', name='prediction', node=PredictionTrainingObjective)

cs.store(group='language_model', name='tiny-stories-33M', node=TinyStories33MLanguageModelConfig)
cs.store(group='dataset', name='tiny-stories', node=TinyStoriesDatasetConfig)


cs.store(group='dataset/datasource', name='local', node=LocalDatasourceConfig)
cs.store(group='dataset/datasource', name='hf', node=HuggingFaceDatasourceConfig)
cs.store(group='dataset/datasource', name='s3', node=S3DatasourceConfig)



cs.store(group='batch', name='token', node=BatchConfig(type='token'))
cs.store(group='batch', name='sequence', node=BatchConfig(type='sequence'))




# cs.store(group='strategy', name='default', node=DefaultStrategyConfig)
# cs.store(group='strategy', name='clt', node=DefaultStrategyConfig('crosscoders.strats.CLTDistributedStrategy'))
cs.store(group='runner/fit_loop', name='default', node=FitLoopConfig('crosscoders.fit_loop.DefaultFitLoop'))
# cs.store(group='fit_loop', name='distributed', node=FitLoopConfig('crosscoders.fit_loop.DistributedFitLoop'))


# ------------------------- globally available var ------------------------- #
def load_omegaconf(config_name: str = os.environ['CONFIG_NAME'], config_path: str = os.environ.get('CONFIG_PATH', '../../src/configs'), overrides: Optional[List[str]] = None):

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    # print(OmegaConf.to_yaml(cfg))


    return cfg


def print_config(cfg, resolve: bool = False, rich: bool = False) -> None:

    cfg_dict = OmegaConf.to_container(cfg, resolve=resolve)
    # cfg_dict['paths']['_Paths__prefix'] = '[magenta]' + str(cfg.paths._Paths__prefix) + '[/]'
    # cfg_dict['paths']['activations_dir'] = '[magenta]' + str(cfg.paths.activations_dir) + '[/]'
    # cfg_dict['batch']['batch_size'] = '[sea_green1]' + str(cfg.batch.batch_size) + '[/]'
    # cfg_dict['batch']['n_tokens'] = '[sea_green1]' + str(cfg.batch.n_tokens) + '[/]'
    # cfg_dict['dataset']['datasource']['which'] = '[red]' + str(cfg.dataset.datasource.which) + '[/]'
    # cfg_dict['runner']['stage'] = '[red]' + str(cfg.runner.stage) + '[/]'
    # cfg_dict['runner']['_target_'] = (lambda i: ''.join([
    #         str(cfg.runner._target_[:i]), '[red]', str(cfg.runner._target_[i:]), '[/]'
    #     ]))(cfg.runner._target_.rfind(".") + 1)


    # cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_yaml = OmegaConf.to_yaml(cfg_dict)


    if rich:

        from rich.console import Console

        console = Console(highlight=False)

        console.print()
        console.print('[bold light_steel_blue3]' + ' '.join(['-' * 25, 'CONFIG', '-' * 25]) + '[/]')
        console.print(cfg_yaml, end='')
        console.print('[bold light_steel_blue3]' + '-' * 61 + '[/]')
        console.print()

    else:
        print()
        print(' '.join(['-' * 25, 'CONFIG', '-' * 25]))
        # print(OmegaConf.to_yaml(cfg, resolve=resolve), end='')
        print(cfg_yaml, end='')
        print('-' * 61)
        print()


def set_config(cfg) -> None:

    global CONFIG

    # print_config(**kwargs)

    CONFIG = cfg
    # torch.manual_seed(CONFIG.seed)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config() -> DictConfig:

    if CONFIG is None:
        raise RuntimeError('global config not set')

    return CONFIG


CONFIG = None
set_config(load_omegaconf())


__all__ = [
    'Config',
    'load_omegaconf',
    'print_config',
    'set_config',
    'get_config',
]