



import datetime
from functools import cached_property
from typing import Dict, Optional
import einops
import torch
from crosscoders.abc import AutoencoderRunnerABC
from crosscoders.autoencoders.acausal.loss import AcausalLoss
from crosscoders.dataclasses.configs.runner import RunnerConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoder
# from crosscoders.constants import MAX_TOKENS
from crosscoders import CONSTANTS
from crosscoders.dataclasses.metrics.loss import LossMetrics
from torch.utils.tensorboard import SummaryWriter




class AcausalAutoencoderRunner(AutoencoderRunnerABC, AcausalLoss):

    def __init__(self, cfg: RunnerConfig) -> None:

        super().__init__(cfg)

        self.model: AcausalAutoencoder = AcausalAutoencoder(self.cfg.MODEL)

        self.optimizer = self.configure_optimizers()

        self.writer = SummaryWriter(f'{CONSTANTS.PROJECT_ROOT_DIR}/log/{datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H:%M:%S")}')
        self.num_tokens_processed: int = 0


    def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:

        x = batch['resid_post']
        self.num_tokens_processed += x.shape[0]
        
        outputs = self.model(x)
        loss = self.loss(outputs, x, W_dec=self.model.W_dec, x_enc=self.model.x_enc)
        loss.loss.backward()
        
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.writer.add_scalar(f'total_grad_norm', total_grad_norm, self.num_tokens_processed)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.writer.add_scalar(f'loss/loss', loss.loss.item(), self.num_tokens_processed)
        self.writer.add_scalar(f'loss/error', loss.error.item(), self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l1', loss.l1.item(), self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l0', loss.l0.item(), self.num_tokens_processed)

        return loss
