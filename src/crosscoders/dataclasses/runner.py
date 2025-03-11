

from dataclasses import dataclass, field
from typing import Tuple

from crosscoders.dataclasses.autoencoders.baseline import BaselineModuleConfig


__all__ = ['RunnerConfig']




# @dataclass
# class DimensionsConfig:

#     n_layers: int = '${runner.language_model.n_layers}'
#     d_model : int = '${runner.language_model.d_model}'
#     d_coder : int = '${runner.crosscoder.model.cfg.d_coder}'


@dataclass
class LanguageModelConfig:

    name    : str = 'tiny-stories-33M'
    n_layers: int = 4
    d_model : int = 768


@dataclass
class OptimizerConfig:

    _target_: str = 'torch.optim.Adam'
    lr      : float = 2e-4
    betas   : Tuple[float,float] = (.9,.999)
    fused   : bool = True


@dataclass
class RunnerConfig:

    batch_size : int = 25000
    max_tokens : int = 1000000000
    max_batches: int = '${eval:"${runner.max_tokens} // ${runner.batch_size}"}'

    input_name : str = 'resid_post'
    output_name: str = 'resid_post'

    # dims: DimensionsConfig = field(default_factory=DimensionsConfig)

    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)

    optimizer : OptimizerConfig      = field(default_factory=OptimizerConfig)
    crosscoder: BaselineModuleConfig = field(default_factory=BaselineModuleConfig)
