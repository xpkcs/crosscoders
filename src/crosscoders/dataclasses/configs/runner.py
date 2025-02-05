



from dataclasses import dataclass, field


from crosscoders.dataclasses.configs.model import ModelConfig
from crosscoders.dataclasses.configs.optimizer import OptimizerConfig
from crosscoders.abc.dataclass import DataclassABC




@dataclass(repr=False)
class AutoencoderLightningModuleConfig(DataclassABC):

    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)

    MODEL: ModelConfig = field(default_factory=ModelConfig)

