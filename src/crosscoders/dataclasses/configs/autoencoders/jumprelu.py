

from dataclasses import dataclass, field

from crosscoders.dataclasses.configs.autoencoders import (
    AutoencoderConfig,
    AutoencoderInitConfig,
    HyperparametersConfig,
    LossMetrics,
    ModuleConfig,
    ParameterInitializationFunctionConfig
)


__all__ = ['JumpReLUModuleConfig']




@dataclass
class JumpReLUHyperparametersConfig(HyperparametersConfig):

    lambda_s: float = 10.
    eps: float = 2.
    c: float = 4.
    lambda_p: float = 3e-6


@dataclass
class JumpReLUParameterInitializationFunctionConfig(ParameterInitializationFunctionConfig):

    pass


@dataclass
class JumpReLUAutoencoderInitConfig(AutoencoderInitConfig):

    param_init: JumpReLUParameterInitializationFunctionConfig = field(default_factory=JumpReLUParameterInitializationFunctionConfig)


@dataclass
class JumpReLUAutoencoderConfig(AutoencoderConfig):

    _target_: str = 'crosscoders.autoencoders.jumprelu.JumpReLUAutoencoder'

    cfg: JumpReLUAutoencoderInitConfig = field(default_factory=JumpReLUAutoencoderInitConfig)


@dataclass
class JumpReLULossMetrics(LossMetrics):

    lp: float


@dataclass
class JumpReLUModuleConfig(ModuleConfig):

    hps: JumpReLUHyperparametersConfig = field(default_factory=JumpReLUHyperparametersConfig)
    model: JumpReLUAutoencoderConfig = field(default_factory=JumpReLUAutoencoderConfig)
    # loss_metrics: JumpReLULossMetrics = JumpReLULossMetrics
