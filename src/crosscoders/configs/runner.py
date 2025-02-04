



from dataclasses import dataclass, field


from crosscoders.configs import ModelConfig, OptimizerConfig
from crosscoders.abc import DataclassABC
from crosscoders.utils import dataclass_repr




@dataclass(repr=False)
class AutoencoderLightningModuleConfig(DataclassABC):

    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)

    MODEL: ModelConfig = field(default_factory=ModelConfig)

