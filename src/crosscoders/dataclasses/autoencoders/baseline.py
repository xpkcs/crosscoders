

from dataclasses import dataclass, field

from crosscoders.dataclasses.autoencoders import (
    AutoencoderConfig,
    AutoencoderInitConfig,
    HyperparametersConfig,
    LossMetrics,
    ModuleConfig,
    ParameterInitializationFunctionConfig
)


__all__ = ['BaselineModuleConfig']




@dataclass
class BaselineHyperparametersConfig(HyperparametersConfig):

    lambda_s: float = 2.


@dataclass
class BaselineParameterInitializationFunctionConfig(ParameterInitializationFunctionConfig):

    pass


@dataclass
class BaselineAutoencoderInitConfig(AutoencoderInitConfig):

    param_init: BaselineParameterInitializationFunctionConfig = field(default_factory=BaselineParameterInitializationFunctionConfig)


@dataclass
class BaselineAutoencoderConfig(AutoencoderConfig):

    _target_: str = 'crosscoders.autoencoders.BaselineAutoencoder'

    cfg: BaselineAutoencoderInitConfig = field(default_factory=BaselineAutoencoderInitConfig)


@dataclass
class BaselineLossMetrics(LossMetrics):

    pass


@dataclass
class BaselineModuleConfig(ModuleConfig):

    hps: BaselineHyperparametersConfig = field(default_factory=BaselineHyperparametersConfig)
    model: BaselineAutoencoderConfig = field(default_factory=BaselineAutoencoderConfig)
    # loss_metrics: BaselineLossMetrics = MISSING