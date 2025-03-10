

from dataclasses import dataclass
from omegaconf import MISSING




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
class HyperparametersConfig:

    lambda_s: float = 1.
    x_scalar: float = 1.
    y_scalar: float = 1.


@dataclass
class ParameterInitializationFunctionConfig:

    W_dec: str = 'kaiming_uniform_'
    W_enc: str = 'transpose'
    b_dec: str = 'pass'
    b_enc: str = 'pass'


@dataclass
class AutoencoderInitConfig:

    n_layers: int = '${runner.language_model.n_layers}'
    d_model: int = '${runner.language_model.d_model}'
    d_coder: int = 24576

    param_init: ParameterInitializationFunctionConfig = MISSING


@dataclass
class AutoencoderConfig:

    _target_: str = MISSING
    _convert_: str = 'object'

    cfg: AutoencoderInitConfig = MISSING


@dataclass
class DeadNeuronMetrics:

    all_tokens: float
    one_token : float
    no_token  : float


@dataclass
class LossMetrics:

    explained_variance: float
    dead_neuron: DeadNeuronMetrics

    loss : float
    error: float
    l1   : float
    l0   : float


@dataclass
class ModuleConfig:

    hps: HyperparametersConfig
    model: AutoencoderConfig
    # loss_metrics: LossMetrics




from crosscoders.dataclasses.configs.autoencoders.baseline import *
from crosscoders.dataclasses.configs.autoencoders.jumprelu import *

from crosscoders.dataclasses.configs.autoencoders.baseline import __all__ as __baseline_all__
from crosscoders.dataclasses.configs.autoencoders.jumprelu import __all__ as __jumprelu_all__


__all__ = []
__all__ += [
    object
    for _ in (__baseline_all__, __jumprelu_all__)
    for object in _
]
