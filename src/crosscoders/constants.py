



import os


from dotenv import load_dotenv;     load_dotenv()

from crosscoders.configs.utils import from_dict, update_dataclass




def resolve_path(path):

    return os.path.abspath(os.path.expanduser(path))


def load_constants():

    global CONSTANTS

    # load global config object
    from crosscoders.configs.globals import GlobalsConfig
    from crosscoders.configs.utils import get_config


    try:
        cfg = get_config(resolve_path(os.environ['CONFIG_FILEPATH']))
        cfg['experiment']['EXPERIMENT'] = cfg['experiment'].get('EXPERIMENT', {})
        cfg['experiment']['EXPERIMENT'] |= {
            'PROJECT_ROOT_DIR': resolve_path(os.environ['PROJECT_ROOT_DIR']),
            'CONFIG_FILEPATH': resolve_path(os.environ['CONFIG_FILEPATH'])
        }

    except KeyError as e:
        raise TypeError(f'Missing required env var: {e.args[0]}')

    # CONSTANTS = ExperimentConfig(**cfg['experiment'])
    # CONSTANTS = GlobalsConfig(**cfg['experiment'])
    # print(CONSTANTS)
    # update_dataclass(CONSTANTS, cfg['experiment'])
    # print(CONSTANTS)


    CONSTANTS = from_dict(GlobalsConfig, cfg['experiment'])

    print()
    print(' '.join(['-'* 25, 'CONSTANTS', '-' * 25]))
    print(CONSTANTS)
    print('-' * 61)
    print()



# CONSTANTS = None
load_constants()

