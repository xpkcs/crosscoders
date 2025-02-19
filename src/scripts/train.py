



import os
import tempfile
import datasets
import lightning as pl
import ray
import ray.train, ray.train.torch
import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.runtime_env import RuntimeEnv

import numpy as np
import torch


# from crosscoders.data.preprocessing import TokenToLatents
from crosscoders.autoencoders.acausal.loss import AcausalLoss
from crosscoders.autoencoders.acausal.model import AcausalAutoencoder
from crosscoders.autoencoders.acausal.runner import AcausalAutoencoderRunner
from crosscoders.dataclasses.configs.runner import LossConfig, ModelConfig

from crosscoders import CONSTANTS
from crosscoders.dataclasses.configs.runner import RunnerConfig
from crosscoders.data.dataset import TinyStoriesRayDataset
from crosscoders.utils import from_dict, get_config, update_dataclass
from torch.utils.tensorboard import SummaryWriter

import os

import tempfile


import torch.amp, torch.optim
import datetime, numpy as np
import logging

logger = logging.getLogger()


def get_x_scalar(ds, runner_cfg):

    def get_batch_expected_norm(batch):

        return {
            'resid_post_norm_sum': [np.linalg.norm(batch['resid_post'], ord=2, axis=-1).sum()],
            'resid_post_norm_count': [batch['resid_post'].shape[0] * batch['resid_post'].shape[1]],
        }

    stats = ds.map_batches(get_batch_expected_norm, zero_copy_batch=True).sum()


    scaled_model_dim = np.sqrt(runner_cfg.MODEL.D_MODEL)
    x_mean_l2 = stats['sum(resid_post_norm_sum)'] / stats['sum(resid_post_norm_count)']

    X_SCALAR = scaled_model_dim / x_mean_l2

    return scaled_model_dim, x_mean_l2, X_SCALAR


def scale_x(batch, X_SCALAR):

    batch['resid_post'] = X_SCALAR * batch['resid_post']

    return batch




def train_loop_per_worker(ray_cfg, **kwargs):

    ray_tune = 'train_ds' in kwargs


    if ray_tune:
        train_ds = kwargs['train_ds']
    else:
        train_ds = ray.train.get_dataset_shard('train')




    # runner config
    runner_cfg = from_dict(
        RunnerConfig,
        get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
    )

    update_dataclass(runner_cfg, ray_cfg)


    logger.info(runner_cfg)
    print(runner_cfg)

    # if 'scale' in ray_cfg and ray_cfg['scale']:
    # if ray_cfg.get('scale', False):
    #     scaled_model_dim, x_mean_l2, X_SCALAR = get_x_scalar(train_ds, runner_cfg)
    #     train_ds = train_ds.map_batches(scale_x, X_SCALAR)

    # dataloader
    train_dl = train_ds.iter_torch_batches(
        batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
        # local_shuffle_buffer_size=16
    )





    # runner
    runner = AcausalAutoencoderRunner(runner_cfg)

    if not ray_tune:
        runner.model = ray.train.torch.prepare_model(runner.model)

    logger.info(str({k: kwargs.get(k) for k in ('X_SCALAR',) if k in kwargs and kwargs.get(k) is not None}))
    metrics = runner.fit(train_dl, **{k: kwargs.get(k) for k in ('X_SCALAR',) if k in kwargs})

    print(metrics)



    # metrics_dict = {
    #     'loss': metrics.loss,
    #     'error': metrics.error,
    #     'l1': metrics.l1,
    #     'l0': metrics.l0,
    #     'explained_variance': metrics.explained_variance,
    #     'dead_neurons/all_tokens': metrics.dead_neurons.all_tokens,
    #     'dead_neurons/one_token': metrics.dead_neurons.one_token,
    #     'dead_neurons/no_token': metrics.dead_neurons.no_token,
    #     'n_tokens': runner.num_tokens_processed,
    # }
    # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:

    #     torch.save(
    #         {'epoch': 0, 'model': runner.model.state_dict()},
    #         os.path.join(temp_checkpoint_dir, "model.pt")
    #     )
    #     ray.train.report(
    #         metrics_dict,
    #         checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
    #     )

    # metrics_dict['should_checkpoint'] = True

    # return metrics_dict



def main():
    
    train_ds = TinyStoriesRayDataset().load('activations')

    print(train_ds)

    # train_dl = train_ds \
    #     .iter_torch_batches(
    #         batch_size=20,
    #         collate_fn=lambda _: {k: torch.as_tensor(np.stack(v)) for k, v in _.items()}
    #     )
    # for batch_idx, batch in enumerate(train_dl):
    #     if batch_idx % 10 == 0:
    #         print(batch_idx)

    #     print(batch.keys())


    trainer = TorchTrainer(
        train_loop_per_worker,
        # train_loop_config=ExperimentConfig(),
        scaling_config=ray.train.ScalingConfig(
            num_workers=CONSTANTS.EXPERIMENT.NUM_TRAINERS,
            use_gpu=True,
            resources_per_worker={'CPU': 2, 'GPU': 1}
        ),
        # run_config = RunConfig(
        #     checkpoint_config=CheckpointConfig(num_to_keep=1),
        #     storage_path="s3://..."
        # )
        datasets={'train': train_ds}
    )
    result: ray.train.Result = trainer.fit()


    return result