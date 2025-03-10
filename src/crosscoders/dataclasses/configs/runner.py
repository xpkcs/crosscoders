



from dataclasses import dataclass, field
from typing import Tuple

from crosscoders.dataclasses.configs.crosscoders import \
    BaselineCrosscoderConfig


@dataclass
class LanguageModelConfig:
    name: str = 'tiny-stories-33M'
    n_layers: int = 4
    d_model: int = 768


@dataclass
class OptimizerConfig:
    _target_: str = 'torch.optim.Adam'
    lr: float = 2e-4
    betas: Tuple[float,float] = (.9,.999)
    fused: bool = True


@dataclass
class RunnerConfig:

    _target_: str = 'crosscoders.autoencoders.runner.Runner'

    batch_size: int = 25000
    max_tokens: int = 1000000000
    max_batches: int = '${eval:"${runner.max_tokens} // ${runner.batch_size}"}'

    input_name: str = 'resid_post'
    output_name: str = 'resid_post'

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    crosscoder: BaselineCrosscoderConfig = field(default_factory=BaselineCrosscoderConfig)
