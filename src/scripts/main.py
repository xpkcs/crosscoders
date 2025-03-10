#!/usr/bin/env python3



from pathlib import Path

# import ray
import hydra

from crosscoders.dataclasses.configs.config import Config

# from crosscoders.constants import CONSTANTS, get_constants
# from ray.runtime_env import RuntimeEnv


# @dataclass
# class ExperimentConfig:

#     batch_size: int
#     max_epochs: int
#     max_tokens: int

#     CONFIG_FILEPATH: str
#     PROJECT_ROOT_DIR: str

#     NUM_GPUS_ACTIVATION: float | int = 0.2
#     NUM_GPUS: float | int = 1
#     NUM_TRAINERS: int = 1


# cs = ConfigStore.instance()
# cs.store(name='config', node=Config)

# def ifelse(condition, on_if, on_else):

#     return on_if if condition else on_else

# OmegaConf.register_new_resolver('ifelse', ifelse)
# OmegaConf.register_new_resolver('eval', eval)



# from crosscoders.dataclasses.configs.config import Config

@hydra.main(
    version_base = None,
    config_path  = str(Path(__file__).parent.parent / 'configs'),
    config_name  = 'config'
)
def main(cfg: Config) -> None:
# @click.command()
# # @click.option('-c', '--config',
# #               type=click.Path(resolve_path=True, path_type=Path),
# #               default=Path(__file__).resolve().parent / 'experiments' / 'train.yml',
# #               help='File path to experiment config.')
# @click.argument('mode',
#                 type=click.Choice(['data', 'train', 'inference']),
#                 default='inference')
# def main(**kwargs: dict) -> None:
    '''
    CLI to run the `crosscoders` package.
    '''

    # import crosscoders as xc
    # from crosscoders import CONSTANTS
    # print(CONSTANTS)

    # cfg = get_config(kwargs['config'])

    # print(cfg)

    # xc.configs.utils.update_dataclass(CONSTANTS, cfg)

    # ray.data._internal.datasource.parquet_datasource.NUM_CPUS_FOR_META_FETCH_TASK = 4
    # ray.data.datasource.parquet_meta_provider.RETRY_MAX_ATTEMPTS_FOR_META_FETCH_TASK = 256
    # ray.data.datasource.parquet_meta_provider.RETRY_MAX_BACKOFF_S_FOR_META_FETCH_TASK = 256


    # ray.init(
    #     runtime_env=RuntimeEnv(
    #         env_vars={
    #             'CONFIG_FILEPATH': CONSTANTS.CONFIG_FILEPATH,
    #             # 'RAY_DEBUG': '1'
    #         },
    #         # py_executable_args=["-Xfrozen_modules=off"]
    #     )
    # )

    # print('Input: Manual Overrides')
    # print('-' * 50)
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # print('-' * 50)

    # set_constants(OmegaConf.to_container(cfg, resolve=True))

    from crosscoders.autoencoders.runner import Runner
    from crosscoders.constants import set_constants

    set_constants(cfg)

    # cfg = hydra.utils.instantiate(manual_overrides)

    # print('Experiment Config')
    # print('-' * 50)
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # print('-' * 50)


    # xc = BaselineAutoencoder(BaselineModelConfig())

    # optimizer = hydra.utils.instantiate(cfg.optimizer, params=list(xc.parameters()))



    # hydra.utils.instantiate(cfg)
    runner = Runner(cfg)


    # from scripts import data, train   # , inference
    # match cfg['mode']:
    #     case 'data':
    #         data.main()

    #     case 'train':
    #         train.main()

    #     case 'inference':
    #         ...




if __name__ == '__main__':
    main()
    main()
