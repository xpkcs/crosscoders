



import datetime
from functools import cached_property
import os
import tempfile
from typing import Dict, Optional
import einops
import ray, ray.train
import torch
from crosscoders.abc import AutoencoderRunnerABC
from crosscoders.autoencoders.acausal.loss import AcausalLoss
from crosscoders.dataclasses.configs.runner import RunnerConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoder
# from crosscoders.constants import MAX_TOKENS
from crosscoders import CONSTANTS
from crosscoders.dataclasses.metrics.loss import LossMetrics
from torch.utils.tensorboard import SummaryWriter




class AcausalAutoencoderRunner(AutoencoderRunnerABC):

    def __init__(self, cfg: RunnerConfig) -> None:

        super().__init__(cfg)

        self.model: AcausalAutoencoder = AcausalAutoencoder(self.cfg.MODEL)

        self.loss: AcausalLoss = AcausalLoss(self.cfg.LOSS)

        self.optimizer = self.configure_optimizers()

        self.writer = SummaryWriter(f'{CONSTANTS.PROJECT_ROOT_DIR}/log/{datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H:%M:%S")}')
        self.num_tokens_processed: int = 0


    def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:

        x = batch['resid_post']
        self.num_tokens_processed += x.shape[0]
        
        x_hat = self.model(x)
        loss, metrics = self.loss(x, x_hat, self.model.x_enc, self.model.W_dec)
        loss.backward()
        
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.writer.add_scalar(f'eval/total_grad_norm', total_grad_norm, self.num_tokens_processed)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.writer.add_scalar(f'loss/loss', metrics.loss, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/error', metrics.error, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l1', metrics.l1, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l0', metrics.l0, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/explained_variance', metrics.explained_variance, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/all_tokens', metrics.dead_neurons.all_tokens, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/one_token', metrics.dead_neurons.one_token, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/no_token', metrics.dead_neurons.no_token, self.num_tokens_processed)

        return metrics


    def fit(self, dl):
        
        for batch_idx, batch in enumerate(dl):

            metrics = self.training_step(batch)
            metrics_dict = {
                'loss': metrics.loss,
                'error': metrics.error,
                'l1': metrics.l1,
                'l0': metrics.l0,
                'explained_variance': metrics.explained_variance,
                'dead_neurons/all_tokens': metrics.dead_neurons.all_tokens,
                'dead_neurons/one_token': metrics.dead_neurons.one_token,
                'dead_neurons/no_token': metrics.dead_neurons.no_token,
                'n_tokens': self.num_tokens_processed,
            }

            # if batch_idx % 10 == 0:
            ray.train.report(
                metrics_dict,
            )


        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                self.model.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics_dict,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )


        return metrics