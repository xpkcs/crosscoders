#!/usr/bin/env python3



import os
from pathlib import Path

import hydra
import click
from omegaconf import OmegaConf


# from crosscoders.config import load_omegaconf


from crosscoders.config import CONFIG
from crosscoders.dataclasses.config import Config



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

@hydra.main(
    config_path=os.environ.get('CONFIG_PATH', '../../src/configs'),
    config_name=os.environ.get('CONFIG_NAME', 'config'),
    version_base=None
)
def main(cfg: Config) -> None:
    '''
    CLI to run the `crosscoders` package.
    '''

    # from crosscoders.config import set_config
    # set_config(cfg)


    print(f'> STAGE: {cfg.runner.stage}')

    # from crosscoders.config import load_omegaconf
    # cfg = load_omegaconf(cfg.runner.stage) # , overrides=[f'++stage=blah'])

    # print(OmegaConf.to_yaml(cfg, resolve=True))


    # print(f"Working directory : {os.getcwd()}")
    # print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")


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
