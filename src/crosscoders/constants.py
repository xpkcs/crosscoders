



import os


from dotenv import load_dotenv;     load_dotenv()

from crosscoders.configs.runner import AutoencoderLightningModuleConfig
from crosscoders.utils import from_dict, update_dataclass



# corresponds to the non-dataclass fields of the
# `GlobalsConfig` dataclass at `src/crosscoders/configs/globals.py`.
REQUIRED_ENV_VARS = [
    'PROJECT_ROOT_DIR',
    'CONFIG_FILEPATH',
]


def resolve_path(path):

    return os.path.abspath(os.path.expanduser(path))


def load_constants():

    global CONSTANTS

    # load global config object
    from crosscoders.configs.globals import GlobalsConfig
    from crosscoders.utils import get_config


    try:
        cfg = get_config(resolve_path(os.environ['CONFIG_FILEPATH']))
        cfg['GLOBALS'] |= {e: resolve_path(os.environ[e]) for e in REQUIRED_ENV_VARS}

    except KeyError as e:
        raise TypeError(f'Missing required env var: {e.args[0]}')


    CONSTANTS = from_dict(GlobalsConfig, cfg.get('GLOBALS', {}))
    # RUNNER_CFG = from_dict(AutoencoderLightningModuleConfig, cfg.get('RUNNER', {}))

    print()
    print(' '.join(['-'* 25, 'CONSTANTS', '-' * 25]))
    print(CONSTANTS)
    print('-' * 61)
    print()



# CONSTANTS = None
load_constants()

