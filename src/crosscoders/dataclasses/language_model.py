

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING






@dataclass
class LanguageModelConfig:

    name     : str = MISSING
    d_model  : int = MISSING
    n_layers : int = MISSING
    n_context: int = MISSING # context window len / max seq len


@dataclass
class TinyStories33MLanguageModelConfig(LanguageModelConfig):

    name     : str = 'tiny-stories-33M'
    d_model  : int = 768
    n_layers : int = 4
    n_context: int = 512

