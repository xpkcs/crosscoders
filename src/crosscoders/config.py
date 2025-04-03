

import os
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf
from crosscoders.dataclasses.config import Config




# ------------------------- globally available vars ------------------------- #
def load_omegaconf(
    config_name: str = os.environ['CONFIG_NAME'],
    config_path: str = os.environ.get('CONFIG_PATH', '../../src/configs'),
    overrides: Optional[List[str]] = None
):

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    # print(OmegaConf.to_yaml(cfg))


    return cfg


def print_config(cfg, resolve: bool = False, rich: bool = True) -> None:

    try:
        assert rich

        from rich.console import Console

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict['paths']['_Paths__prefix'] = '[magenta]' + str(cfg.paths._Paths__prefix) + '[/]'
        cfg_dict['paths']['activations_dir'] = '[magenta]' + str(cfg.paths.activations_dir) + '[/]'
        cfg_dict['batch']['batch_size'] = '[sea_green1]' + str(cfg.batch.batch_size) + '[/]'
        cfg_dict['batch']['n_tokens'] = '[sea_green1]' + str(cfg.batch.n_tokens) + '[/]'
        cfg_dict['dataset']['datasource']['which'] = '[red]' + str(cfg.dataset.datasource.which) + '[/]'
        cfg_dict['runner']['stage'] = '[red]' + str(cfg.runner.stage) + '[/]'
        cfg_dict['runner']['_target_'] = (lambda i: ''.join([
                str(cfg.runner._target_[:i]), '[red]', str(cfg.runner._target_[i:]), '[/]'
            ]))(cfg.runner._target_.rfind(".") + 1)


        cfg_yaml = OmegaConf.to_yaml(cfg_dict)


        console = Console(highlight=False)

        console.print()
        console.print('[bold light_steel_blue3]' + ' '.join(['-' * 25, 'CONFIG', '-' * 25]) + '[/]')
        console.print(cfg_yaml, end='')
        console.print('[bold light_steel_blue3]' + '-' * 61 + '[/]')
        console.print()

    except:
        print()
        print(' '.join(['-' * 25, 'CONFIG', '-' * 25]))
        print(OmegaConf.to_yaml(cfg, resolve=resolve), end='')
        print('-' * 61)
        print()


def set_config(cfg) -> None:

    global __CONFIG__

    # print_config(**kwargs)

    __CONFIG__ = cfg

    import torch

    torch.manual_seed(__CONFIG__.globals.seed)
    torch.set_default_dtype(torch.float32)
    # torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config() -> DictConfig:

    if __CONFIG__ is None:
        raise RuntimeError('global config not set')

    return __CONFIG__


__CONFIG__ = None
# set_config(load_omegaconf())


__all__ = [
    'Config',
    'load_omegaconf',
    'print_config',
    'set_config',
    'get_config',
]