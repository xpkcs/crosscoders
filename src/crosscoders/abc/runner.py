



import lightning as pl


from crosscoders.configs import AutoencoderLightningModuleConfig




class AutoencoderLightningModuleABC(pl.LightningModule):

    def __init__(self, cfg: AutoencoderLightningModuleConfig):

        super().__init__()

        self.cfg = cfg


    # def configure_model(self):

    #     match self.cfg.model.causality:

    #         case 'acausal':
    #             self.model = AcausalAutoencoder(self.cfg.model)

    #         case _:
    #             raise NotImplementedError()


    def configure_optimizers(self):

        return self.cfg.optimizer.optimizer(
            self.model.parameters(),
            **self.cfg.optimizer.parameters._asdict()
        )


    def forward(self, batch):

        return self.model(batch)


    def training_step(self, batch):

        return self.loss(batch, self(batch))
