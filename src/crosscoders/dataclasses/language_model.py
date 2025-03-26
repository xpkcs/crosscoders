

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING






@dataclass
class LanguageModelConfig:

    name    : str = ''
    n_layers: int = 0
    d_model : int = 0


@dataclass
class TinyStories33MLanguageModelConfig(LanguageModelConfig):

    name    : str = 'tiny-stories-33M'
    n_layers: int = 4
    d_model : int = 768
