

from dotenv import load_dotenv;     load_dotenv()


import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

from crosscoders.config import print_config, get_config, set_config
from crosscoders import abc, autoencoders, dataclasses


print_config(get_config(), resolve=True)



__all__ = [
    'print_config', 'get_config', 'set_config',
    'abc', 'autoencoders', 'dataclasses',
]
