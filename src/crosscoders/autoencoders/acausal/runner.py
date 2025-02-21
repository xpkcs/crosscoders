



import datetime
from functools import cached_property
import os
import tempfile
from typing import Dict, Optional
import einops
import ray, ray.train, ray.tune
import torch
from crosscoders.abc import AutoencoderRunnerABC
from crosscoders.autoencoders.acausal.loss import AcausalLoss, AcausalLossJumpReLU
from crosscoders.dataclasses.configs.runner import RunnerConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoder
# from crosscoders.constants import MAX_TOKENS
from crosscoders import CONSTANTS
from crosscoders.dataclasses.metrics.loss import LossMetrics
from torch.utils.tensorboard import SummaryWriter
from ray import train, tune
from ray.train import Checkpoint


def get_final_decay_scheduler(optimizer, total_steps, decay_start_fraction=0.8):

    # cite: claude
    def lr_lambda(current_step):
        max_batches = CONSTANTS.EXPERIMENT.MAX_TOKENS // CONSTANTS.EXPERIMENT.BATCH_SIZE + 1
        decay_start = int(max_batches * decay_start_fraction)

        if current_step < decay_start:
            return 1.0
        else:
            decay_progress = (current_step - decay_start) / (total_steps - decay_start)
            return 1.0 - decay_progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AcausalAutoencoderRunner(AutoencoderRunnerABC):

    def __init__(self, cfg: RunnerConfig) -> None:

        super().__init__(cfg)

        self.model: AcausalAutoencoder = AcausalAutoencoder(self.cfg.MODEL)

        self.loss: AcausalLoss = AcausalLossJumpReLU(self.cfg.LOSS)

        self.optimizer = self.configure_optimizers()
        # self.scheduler = self.configure_schedulers()

        self.writer = SummaryWriter(f'{CONSTANTS.PROJECT_ROOT_DIR}/log/{datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H:%M:%S")}')
        self.num_tokens_processed: int = 0


    def configure_schedulers(self):

        return get_final_decay_scheduler(self.optimizer, CONSTANTS.EXPERIMENT.MAX_BATCHES)


    def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:

        x = batch[self.cfg.INPUT_NAME]
        self.num_tokens_processed += x.shape[0]

        y = batch[self.cfg.OUTPUT_NAME]
        y_hat = self.model(x)
        # TODO:
        # loss, metrics = self.loss(x, x_hat, self.model.x_enc, self.model.W_dec)
        loss, metrics = self.loss(y, y_hat, self.model.x_enc, self.model.W_dec, self.model.t)
        loss.backward()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.writer.add_scalar(f'eval/total_grad_norm', total_grad_norm, self.num_tokens_processed)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.writer.add_scalar(f'loss/loss', metrics.loss, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/error', metrics.error, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l1', metrics.l1, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/lp', metrics.lp, self.num_tokens_processed)
        self.writer.add_scalar(f'loss/l0', metrics.l0, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/explained_variance', metrics.explained_variance, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/all_tokens', metrics.dead_neurons.all_tokens, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/one_token', metrics.dead_neurons.one_token, self.num_tokens_processed)
        self.writer.add_scalar(f'eval/dead_neurons/no_token', metrics.dead_neurons.no_token, self.num_tokens_processed)

        return metrics


    def fit(self, dl, **kwargs):

        for batch_idx, batch in enumerate(dl):

            batch[self.cfg.INPUT_NAME] = kwargs.get('X_SCALAR', 1.) * batch[self.cfg.INPUT_NAME]

            metrics = self.training_step(batch)
            metrics_dict = {
                'loss': metrics.loss,
                'error': metrics.error,
                'l1': metrics.l1,
                'lp': metrics.lp,
                'l0': metrics.l0,
                'explained_variance': metrics.explained_variance,
                'dead_neurons/all_tokens': metrics.dead_neurons.all_tokens,
                'dead_neurons/one_token': metrics.dead_neurons.one_token,
                'dead_neurons/no_token': metrics.dead_neurons.no_token,
                'n_tokens': self.num_tokens_processed,
            }


            ray.train.report(
                metrics_dict,
            )

            # self.scheduler.step()


        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                self.model.state_dict(),
                os.path.join(temp_checkpoint_dir, 'model.pt')
            )
            train.report(
                metrics_dict,
                checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
            )


        return metrics