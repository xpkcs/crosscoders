



from typing import Dict
import torch
from crosscoders.abc import LossABC, AutoencoderLightningModuleABC
from crosscoders.autoencoders.acausal.loss import AcausalLoss
from crosscoders.configs import AutoencoderLightningModuleConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoder
# from crosscoders.constants import MAX_TOKENS
from crosscoders import CONSTANTS




class AcausalAutoencoderLightningModule(AutoencoderLightningModuleABC):

    def __init__(self, cfg: AutoencoderLightningModuleConfig):

        super().__init__(cfg)

        self.model = AcausalAutoencoder(self.cfg.model)

        # self.loss = None  # TODO
        self.loss: LossABC = AcausalLoss()

        self.n_tokens_processed: int = 0
        self.n_seqs_processed: int = 0


    # def on_before_batch_transfer(self, batch, dataloader_idx):

    #     batch = batch['resid_post']

    #     return batch


    def on_train_batch_start(self, batch: Dict, batch_idx: int):

        if type(CONSTANTS.EXPERIMENT.MAX_TOKENS) == int and self.n_tokens_processed > CONSTANTS.EXPERIMENT.MAX_TOKENS:
            return -1


    def training_step(self, batch: Dict[str, torch.Tensor]):

        # batch_size * seq_len
        self.n_tokens_processed += int(batch['resid_post'].shape[0] * batch['resid_post'].shape[1])
        self.n_seqs_processed += batch['resid_post'].shape[0]

        # raise NotImplementedError({k: type(v) for k,v in batch.items()})

        reconstruction_error, regularization_penalty_l1, regularization_penalty_l0 = self.loss(batch['resid_post'], self(batch['resid_post']))
        loss = reconstruction_error + regularization_penalty_l1

        self.log('loss', loss, on_step=True, prog_bar=True)
        self.log('n_tokens_processed', self.n_tokens_processed, on_step=True, prog_bar=True)
        self.log('n_seqs_processed', self.n_seqs_processed, on_step=True, prog_bar=True)


        return loss
