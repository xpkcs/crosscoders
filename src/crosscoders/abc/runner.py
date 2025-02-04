



from abc import abstractmethod
import lightning as pl


from crosscoders.abc import AutoencoderABC, LossABC
from crosscoders.configs import AutoencoderLightningModuleConfig




class AutoencoderLightningModuleABC(pl.LightningModule):

    model: AutoencoderABC
    loss: LossABC


    def __init__(self, cfg: AutoencoderLightningModuleConfig):

        super().__init__()

        self.cfg = cfg


    # @property
    # @abstractmethod
    # def model(self):
    #     ...


    # @property
    # @abstractmethod
    # def loss(self):
    #     ...


    # def configure_model(self):

    #     match self.cfg.model.causality:

    #         case 'acausal':
    #             self.model = AcausalAutoencoder(self.cfg.model)

    #         case _:
    #             raise NotImplementedError()


    def configure_optimizers(self):

        return self.cfg.OPTIMIZER.optimizer(
            self.model.parameters(),
            **self.cfg.OPTIMIZER.parameters.asdict()
        )


    def forward(self, batch):

        return self.model(batch)


    # def training_step(self, batch):

    #     return self.loss(batch, self(batch))
