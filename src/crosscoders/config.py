

import os

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch




# ------------------------- resolvers ------------------------- #
def ifelse(condition, on_if, on_else):

    return on_if if condition else on_else


OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('ifelse', ifelse)





# ------------------------- structured configs ------------------------- #

from crosscoders.dataclasses.configs.config import Config
from crosscoders.dataclasses.configs.autoencoders.baseline import BaselineAutoencoderConfig
from crosscoders.dataclasses.configs.autoencoders.jumprelu import JumpReLUAutoencoderConfig


cs = ConfigStore.instance()

cs.store(name='base_config', node=Config)
cs.store(group='crosscoder', name='baseline', node=BaselineAutoencoderConfig)
cs.store(group='crosscoder', name='jumprelu', node=JumpReLUAutoencoderConfig)




# ------------------------- globally available var ------------------------- #
def load_omegaconf(config_name: str = 'config', config_path: str = os.environ.get('CONFIG_PATH', '../../src/configs')):

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    # print(OmegaConf.to_yaml(cfg))


    return cfg


def set_config(cfg: Config) -> None:

    global CONFIG

    print()
    print(' '.join(['-' * 25, 'CONFIG', '-' * 25]))
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print('-' * 61)
    print()


    CONFIG = cfg
    torch.manual_seed(CONFIG.seed)
    # torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config():

    if CONFIG is None:
        raise RuntimeError('global config not set')

    return CONFIG


CONFIG = None
set_config(load_omegaconf())
