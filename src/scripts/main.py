#!/usr/bin/env python3



import os
from pathlib import Path

# import ray
import hydra
import click
from omegaconf import OmegaConf
import ray.runtime_env

# from crosscoders.config import load_omegaconf
# from crosscoders.dataclasses.config import Config

# import crosscoders as xc
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

    if cfg.globals.ray_job:

        import ray, ray.data, ray.runtime_env


        # ray.data._internal.datasource.parquet_datasource.NUM_CPUS_FOR_META_FETCH_TASK = 4
        # ray.data.datasource.parquet_meta_provider.RETRY_MAX_ATTEMPTS_FOR_META_FETCH_TASK = 256
        # ray.data.datasource.parquet_meta_provider.RETRY_MAX_BACKOFF_S_FOR_META_FETCH_TASK = 256


        ray.init(
            runtime_env=ray.runtime_env.RuntimeEnv(
                # env_vars={
                #     'CONFIG_PATH': CONFIG.CONFIG_FILEPATH,
                #     # 'RAY_DEBUG': '1'
                # },
                # py_executable_args=["-Xfrozen_modules=off"]
            )
        )


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
