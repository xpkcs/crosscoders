



from functools import cached_property
from typing import Dict, Optional
import torch
from crosscoders.abc import AutoencoderLightningModuleABC
from crosscoders.autoencoders.acausal.loss import AcausalLoss
from crosscoders.configs import AutoencoderLightningModuleConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoder
# from crosscoders.constants import MAX_TOKENS
from crosscoders import CONSTANTS




class AcausalAutoencoderLightningModule(AutoencoderLightningModuleABC):

    def __init__(self, cfg: AutoencoderLightningModuleConfig) -> None:

        super().__init__(cfg)

        self.model = AcausalAutoencoder(self.cfg.MODEL)

        self.loss = AcausalLoss()

        self.n_tokens_processed: int = 0
        self.n_seqs_processed: int = 0


    # @cached_property
    # def model(self) -> AcausalAutoencoder:

    #     return AcausalAutoencoder(self.cfg.MODEL)
    
    
    # @cached_property
    # def loss(self) -> AcausalLoss:

    #     return AcausalLoss()
    



    # def on_before_batch_transfer(self, batch, dataloader_idx):

    #     batch = batch['resid_post']

    #     return batch


    def on_train_batch_start(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[int]:

        if type(CONSTANTS.EXPERIMENT.MAX_TOKENS) is int and self.n_tokens_processed > CONSTANTS.EXPERIMENT.MAX_TOKENS:
            return -1


    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        # batch_size * seq_len
        # self.n_tokens_processed += int(batch['resid_post'].shape[0] * batch['resid_post'].shape[1])
        # self.n_seqs_processed += batch['resid_post'].shape[0]
        self.n_tokens_processed += batch['resid_post'].shape[0]

        # raise NotImplementedError({k: type(v) for k,v in batch.items()})

        loss_metrics = self.loss(batch['resid_post'], self(batch['resid_post']), W_dec=self.model.W_dec, x_enc=self.model.x_enc)

        self.log('loss', loss_metrics.loss, on_step=True, prog_bar=True)
        self.log('error', loss_metrics.reconstruction_error, on_step=True, prog_bar=True)
        self.log('l1', loss_metrics.regularization_penalty_l1, on_step=True, prog_bar=True)
        # self.log('l0', loss_metrics.regularization_penalty_l0, on_step=True, prog_bar=True)
        self.log('n_tokens_processed', self.n_tokens_processed, on_step=True, prog_bar=True)
        # self.log('n_seqs_processed', self.n_seqs_processed, on_step=True, prog_bar=True)


        return loss_metrics.loss
