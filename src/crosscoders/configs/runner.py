



from dataclasses import dataclass, field


from crosscoders.configs import ModelConfig, OptimizerConfig




@dataclass
class AutoencoderLightningModuleConfig:

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    model: ModelConfig = field(default_factory=ModelConfig)
