

# env vars
from dotenv import load_dotenv;                                                 load_dotenv()
from crosscoders.utils import check_required_env_vars;                          check_required_env_vars()


# set config store
from crosscoders.config_store import register_resolvers, init_config_store

register_resolvers()
init_config_store()


# set global config
from crosscoders.config import load_omegaconf, print_config, get_config, set_config

set_config(load_omegaconf())
# print_config(get_config(), resolve=True)




from crosscoders import abc, autoencoders, dataclasses


# misc



# ------------------------- #




__all__ = [
    'load_omegaconf', 'print_config', 'get_config', 'set_config',
    'abc', 'autoencoders', 'dataclasses',
]
