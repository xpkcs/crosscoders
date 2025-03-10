

from dataclasses import MISSING, dataclass, field

from crosscoders.abc.dataclass import DataclassABC as DataclassABC
from crosscoders.dataclasses.configs.globals import \
    HardwareConfig as HardwareConfig

# @dataclass
# class InitWdec:
#     _target_ = torch.nn.init.kaiming_uniform_

# @dataclass
# class InitWenc:
#     _target_ = None

# @dataclass
# class Initbdec:
#     _target_ = None

# @dataclass
# class Initbenc:
#     _target_ = None


# @dataclass
# class ActivationFunction:
#     _target_: str = torch.nn.functional.relu


@dataclass
class ParameterInitializationFunctionConfig:

    W_dec: str = 'kaiming_uniform_'
    W_enc: str = 'transpose'
    b_dec: str = 'pass'
    b_enc: str = 'pass'


# @dataclass
# class CrosscoderConfig:
#     _target_ = None
#     d_coder: int = 24576

#     init: ParameterInitializationFunctionConfig = field(default_factory=ParameterInitializationFunctionConfig)



#     # W_dec_init: InitWdec = field(default_factory=InitWdec)
#     # W_enc_init: InitWenc = field(default_factory=InitWenc)
#     # b_dec_init: Initbdec = field(default_factory=Initbdec)
#     # b_enc_init: Initbenc = field(default_factory=Initbenc)

#     # activation_function: ActivationFunction = field(default_factory=ActivationFunction)
#     activation_function: str = 'relu'



@dataclass
class CrosscoderConfig:
    _target_: str = ''

    d_coder: int = 24576
    init: ParameterInitializationFunctionConfig = field(default_factory=ParameterInitializationFunctionConfig)


@dataclass
class Hyperparameters:

    lambda_s: float = 10
    x_scalar: float = 1.
    y_scalar: float = 1.


@dataclass
class LossMetrics:

    loss: float
    error: float
    l1: float
    l0: float

    explained_variance: float


# @dataclass
# class HyperparametersJumpReLU(Hyperparameters):
#     eps: float = 2
#     c: float = 4
#     lambda_p: float = 3e-6





from hydra.core.config_store import ConfigStore

from crosscoders.dataclasses.configs.autoencoders.baseline import \
    BaselineCrosscoderConfig
from crosscoders.dataclasses.configs.autoencoders.jumprelu import \
    JumpReLUCrosscoderConfig

cs = ConfigStore.instance()
cs.store(group='crosscoder', name='baseline', node=BaselineCrosscoderConfig)
cs.store(group='crosscoder', name='jumprelu', node=JumpReLUCrosscoderConfig)
