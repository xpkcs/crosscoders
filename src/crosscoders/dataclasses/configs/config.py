


# from omegaconf import OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

from omegaconf import MISSING
import torch

from crosscoders.dataclasses.configs.runner import RunnerConfig


@dataclass
class Paths:
    # pass
    # PROJECT_ROOT_DIR: str =
    # local_prefix: str = '${hydra:runtime.cwd}'
    local_prefix: str = Path(__file__).parents[3]
    s3_prefix: str = 's3://${..s3_bucket}'
    prefix: str = '${ifelse:${..local}, ${.local_prefix}, ${.s3_prefix}}'

    data_dir: str = '${.prefix}/data'
    config_path: str = MISSING


    # def get_path(self, path_type: str, local: Optional[bool] = None) -> str:

    #     try:
    #         local = '${..LOCAL}' if local is None else local
    #         prefix = self.local_prefix if local else self.s3_prefix
    #         suffix = getattr(self, path_type)

    #         path = f'{prefix}/{suffix}'

    #     except:
    #         raise ValueError(f'Unknown path type {path_type}')

    #     else:
    #         return path


# @dataclass
# class Hyperparameters:
#     lambda_s: float = 10

# @dataclass
# class HyperparametersJumpReLU(Hyperparameters):
#     eps: float = 2
#     c: float = 4
#     lambda_p: float = 3e-6

# @dataclass
# class Optimizer:
#     _target_: str = 'torch.optim.Adam'
#     lr: float = 2e-4
#     betas: Tuple[float,float] = (.9,.999)
#     fused: bool = True

# @dataclass
# class LanguageModel:
#     name: str = 'tiny-stories-33M'
#     n_layers: int = 4
#     d_model: int = 768


# # @dataclass
# # class InitWdec:
# #     _target_ = torch.nn.init.kaiming_uniform_

# # @dataclass
# # class InitWenc:
# #     _target_ = None

# # @dataclass
# # class Initbdec:
# #     _target_ = None

# # @dataclass
# # class Initbenc:
# #     _target_ = None


# # @dataclass
# # class ActivationFunction:
# #     _target_: str = torch.nn.functional.relu


# @dataclass
# class ParameterInitializationFunction:

#     W_dec: str = 'kaiming_uniform_'
#     W_enc: str = 'transpose'
#     b_dec: str = 'pass'
#     b_enc: str = 'pass'


# @dataclass
# class Crosscoder:
#     _target_ = None
#     d_coder: int = 24576

#     init: ParameterInitializationFunction = field(default_factory=ParameterInitializationFunction)



#     # W_dec_init: InitWdec = field(default_factory=InitWdec)
#     # W_enc_init: InitWenc = field(default_factory=InitWenc)
#     # b_dec_init: Initbdec = field(default_factory=Initbdec)
#     # b_enc_init: Initbenc = field(default_factory=Initbenc)

#     # activation_function: ActivationFunction = field(default_factory=ActivationFunction)
#     activation_function: str = 'relu'


# @dataclass
# class Runner:

#     batch_size: int = 25000
#     max_tokens: int = 1000000000
#     max_batches: int = '${eval:"${runner.max_tokens} // ${runner.batch_size}"}'

#     optimizer: Optimizer = field(default_factory=Optimizer)

#     language_model: LanguageModel = field(default_factory=LanguageModel)
#     crosscoder: Crosscoder = field(default_factory=Crosscoder)



@dataclass
class Config:

    mode: str = MISSING


    seed: int = 314159
    local: bool = False

    input_name: str = 'ln2.normalized'
    output_name: str = 'mlp_out'

    # PROJECT_ROOT_DIR: str = str(Path(__file__).parent.parent.parent.parent)
    # LOCAL_DATA_DIR: Optional[str] = None

    s3_bucket: str = 'crosscoders'

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'float32'

    paths: Paths = field(default_factory=Paths)
    # PATHS: Paths = Paths()

    # def __post_init__(self):

    #     if self.LOCAL_DATA_DIR is None:
    #         self.LOCAL_DATA_DIR = f'{self.PROJECT_ROOT_DIR}/data'


    # hps: Hyperparameters = field(default_factory=Hyperparameters)

    runner: RunnerConfig = field(default_factory=RunnerConfig)
