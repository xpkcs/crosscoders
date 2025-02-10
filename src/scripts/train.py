



import os
import tempfile
import datasets
import lightning as pl
import ray
import ray.train
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
from crosscoders.utils import from_dict, get_config
from torch.utils.tensorboard import SummaryWriter

import os

import tempfile


import torch.amp, torch.optim
import datetime, numpy as np





def train_loop_per_worker():


    train_dl = ray.train.get_dataset_shard('train').iter_torch_batches(
        batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
        # local_shuffle_buffer_size=16
    )



    cfg = from_dict(
        RunnerConfig,
        get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
    )

    runner = AcausalAutoencoderRunner(cfg)
    loss = runner.fit(train_dl)


    metrics = {
        'loss': loss.loss.item(),
        'error': loss.error.item(),
        'l1': loss.l1.item(),
        'l0': loss.l0.item(),
    }
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        torch.save(
            runner.model.state_dict(),
            os.path.join(temp_checkpoint_dir, "model.pt")
        )
        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        )




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