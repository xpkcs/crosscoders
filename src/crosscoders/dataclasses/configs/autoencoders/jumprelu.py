


from dataclasses import dataclass

from crosscoders.dataclasses.configs.autoencoders import (CrosscoderConfig,
                                                          Hyperparameters,
                                                          LossMetrics)


@dataclass
class JumpReLUHyperparameters(Hyperparameters):

    eps: float
    c: float
    lambda_p: float


@dataclass
class JumpReLUCrosscoderConfig(CrosscoderConfig):

    _target_: str = 'crosscoders.autoencoders.jumprelu.JumpReLUAutoencoder'


@dataclass
class JumpReLULossMetrics(LossMetrics):

    lp: float
