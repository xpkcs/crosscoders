

import os

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from crosscoders.dataclasses.configs.config import Config


def load_omegaconf(config_name: str = 'config', config_path: str = os.environ.get('CONFIG_PATH', '../../src/configs')):

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    # print(OmegaConf.to_yaml(cfg))


    return cfg





def ifelse(condition, on_if, on_else):

    return on_if if condition else on_else


OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('ifelse', ifelse)






cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)



def set_config(cfg: Config) -> None:

    global CONFIG


    CONFIG = cfg

    print()
    print(' '.join(['-' * 25, 'CONFIG', '-' * 25]))
    print(OmegaConf.to_yaml(CONFIG, resolve=True))
    print('-' * 61)
    print()


def get_config():

    try:
        assert CONFIG is not None
        return CONFIG

    except:
        raise RuntimeError('global config not set')


CONFIG = None
set_config(load_omegaconf())