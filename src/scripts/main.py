#!/usr/bin/env python3



from pathlib import Path

# import ray
import hydra
import click
import ray.runtime_env

# from crosscoders.config import load_omegaconf
# from crosscoders.dataclasses.config import Config




@click.command()
@click.option('-s', '--stage',
              type=click.Choice(['data', 'train', 'tune', 'eval']),
              default=None,
              help='Which stage to run',
)
@click.option('-rj', '--ray-job',
              type=bool,
              default=False,
              help='Whether to run as a ray job. Affects ray.init.',
)
def main(stage, ray_job) -> None:
    '''
    CLI to run the `crosscoders` package.
    '''


    if ray_job:

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




    print(f'> STAGE: {stage}')

    from crosscoders.config import load_omegaconf
    cfg = load_omegaconf(stage) # , overrides=[f'++stage=blah'])


    from scripts import data, train   # , eval
    match stage:
        case 'data':
            data.main(cfg)

        case 'train':
            train.main(cfg)

        case 'eval':
            ...




if __name__ == '__main__':
    main()
