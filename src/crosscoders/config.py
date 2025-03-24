

import os
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
import numpy as np




# ------------------------- resolvers ------------------------- #
def eq(x, y):

    return x == y

def ifelse(condition, on_if, on_else):

    return on_if if condition else on_else

def ceil(i):

    return int(np.ceil(float(i)))

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('eq', eq)
OmegaConf.register_new_resolver('ifelse', ifelse)
OmegaConf.register_new_resolver('ceil', ceil)





# ------------------------- structured configs ------------------------- #

from crosscoders.dataclasses.args import ArgsConfig
from crosscoders.dataclasses.config import Config
from crosscoders.dataclasses.runner import DataRunnerConfig, RunnerConfig, TrainRunnerConfig, EvalRunnerConfig
from crosscoders.dataclasses.autoencoders.baseline import BaselineAutoencoderConfig
from crosscoders.dataclasses.autoencoders.jumprelu import JumpReLUAutoencoderConfig


cs = ConfigStore.instance()

cs.store(name='base_config', node=Config)
cs.store(name='base_runner', node=RunnerConfig)
cs.store(group='runner', name='data', node=DataRunnerConfig)
cs.store(group='runner', name='train', node=TrainRunnerConfig)
cs.store(group='runner', name='eval', node=EvalRunnerConfig)
cs.store(group='crosscoder', name='baseline', node=BaselineAutoencoderConfig)
cs.store(group='crosscoder', name='jumprelu', node=JumpReLUAutoencoderConfig)
cs.store(name='args', node=ArgsConfig)




# ------------------------- globally available var ------------------------- #
def load_omegaconf(config_name: str = os.environ['CONFIG_NAME'], config_path: str = os.environ.get('CONFIG_PATH', '../../src/configs'), overrides: Optional[List[str]] = None):

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    # print(OmegaConf.to_yaml(cfg))


    return cfg


def set_config(cfg: Config, resolve=False) -> None:

    global CONFIG


    print()
    print(' '.join(['-' * 25, 'CONFIG', '-' * 25]))
    print(OmegaConf.to_yaml(cfg, resolve=resolve), end='')
    print('-' * 61)
    print()


    CONFIG = cfg
    # torch.manual_seed(CONFIG.seed)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config():

    if CONFIG is None:
        raise RuntimeError('global config not set')

    return CONFIG


CONFIG = None
set_config(load_omegaconf(), resolve=True)
