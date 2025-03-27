#!/usr/bin/env python3


import os
from pathlib import Path




# import click
# @click.command()
# @click.option('-s', '--stage',
#               type=click.Choice(['data', 'train', 'tune', 'eval']),
#               default=None,
#               help='Which stage to run',
# )
# @click.option('-rj', '--ray-job',
#               type=bool,
#               default=False,
#               help='Whether to run as a ray job. Affects ray.init.',
# )
# def main(stage, ray_job) -> None:


import hydra
from omegaconf import OmegaConf
from crosscoders.dataclasses.config import Config

import logging
from rich import print as printr
from rich.logging import RichHandler
from rich.text import Text

# logging.basicConfig(
#     format='%(message)s',
#     level='NOTSET',
#     datefmt="[%X]",
#     handlers=[RichHandler(markup=True)],
# )
# log = logging.getLogger('rich')


@hydra.main(
    config_path=os.environ.get('CONFIG_PATH', str((Path(__file__).parent / '../../src/configs').resolve())),
    config_name=os.environ.get('CONFIG_NAME', 'config'),
    version_base=None
)
def main(cfg: Config) -> None:
    '''
    CLI to run the `crosscoders` package.
    '''

    from crosscoders.config import print_config, get_config, set_config

    set_config(cfg)
    CONFIG = get_config()

    printr(f'[bold red]>>>[/] [bold green]STAGE:[/] {cfg.runner.stage}')

    print_config(CONFIG, resolve=True)

    printr(f'[bold red]>>>[/] current directory: {os.getcwd()}')
    printr(f'[bold red]>>>[/] hydra   directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}')

    os.chdir(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    OmegaConf.save(cfg, 'config.yml')
    OmegaConf.save(cfg, 'config-resolved.yml', resolve=True)


    from scripts import data, train#, eval
    match cfg.runner.stage:
        case 'data':
            data.main(cfg)

        case 'train':
            train.main(cfg)

        case 'eval':
            ...




if __name__ == '__main__':
    main()
