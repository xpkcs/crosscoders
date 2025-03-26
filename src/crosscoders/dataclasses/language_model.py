

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING






@dataclass
class LanguageModelConfig:

    name    : str = MISSING
    n_layers: int = MISSING
    d_model : int = MISSING


@dataclass
class TinyStories33MLanguageModelConfig(LanguageModelConfig):

    name    : str = 'tiny-stories-33M'
    n_layers: int = 4
    d_model : int = 768
