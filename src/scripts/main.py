#!/usr/bin/env python3



import click


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



@click.command()
# @click.option('-c', '--config',
#               type=click.Path(resolve_path=True, path_type=Path),
#               default=Path(__file__).resolve().parent / 'experiments' / 'train.yml',
#               help='File path to experiment config.')
@click.argument('mode',
                type=click.Choice(['data', 'train', 'inference']),
                default='inference')
def main(**kwargs: dict) -> None:
    '''
    CLI to run the `crosscoders` package.
    '''

    # import crosscoders as xc
    # from crosscoders import CONSTANTS
    # print(CONSTANTS)

    # cfg = get_config(kwargs['config'])

    # print(cfg)

    # xc.configs.utils.update_dataclass(CONSTANTS, cfg)


    from scripts import data, train   # , inference
    match kwargs['mode']:
        case 'data':
            data.main()

        case 'train':
            train.main()

        case 'inference':
            ...




if __name__ == '__main__':
    main()
