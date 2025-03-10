


from dataclasses import dataclass

from crosscoders.dataclasses.configs.autoencoders import (CrosscoderConfig,
                                                          Hyperparameters,
                                                          LossMetrics)


@dataclass
class BaselineHyperparameters(Hyperparameters):
    pass


@dataclass
class BaselineCrosscoderConfig(CrosscoderConfig):

    _target_: str = 'crosscoders.autoencoders.baseline.BaselineAutoencoder'


@dataclass
class BaselineLossMetrics(LossMetrics):
    pass
