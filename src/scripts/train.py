



import datasets
import lightning as pl
import ray, ray.train, ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.runtime_env import RuntimeEnv
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os
import numpy as np
import torch


from crosscoders.data.preprocessing import TokenToLatents
from crosscoders.configs import ExperimentConfig, ModelConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoderLightningModule

from crosscoders import CONSTANTS
from crosscoders.configs import AutoencoderLightningModuleConfig






def train_loop_per_worker():

    # cfg =

    # Fetch the Dataset shards
    train_dl = ray.train.get_dataset_shard('train').iter_torch_batches(
        batch_size=CONSTANTS.BATCH_SIZE,
        collate_fn=lambda _: {k: torch.as_tensor(np.stack(v)) for k, v in _.items()}
    )
    # valid_dl = ray.train.get_dataset_shard('valid').iter_torch_batches(batch_size=cfg.BATCH_SIZE)

    # train_dl = train_ds.iter_torch_batches(batch_size=cfg.BATCH_SIZE)
    # val_dl = val_ds.iter_torch_batches(batch_size=cfg.BATCH_SIZE)


    model = AcausalAutoencoderLightningModule(
        AutoencoderLightningModuleConfig(
            model=ModelConfig('acausal')
        )
    )


    trainer = pl.Trainer(
        # max_epochs=10,
        max_epochs=CONSTANTS.MAX_EPOCHS,
        devices='auto',
        accelerator='gpu',
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[
            ray.train.lightning.RayTrainReportCallback(),
            # EarlyStopping(monitor='n_tokens_processed', stopping_threshold=100)
        ],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=False,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_dl)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)




def main():

    ray.init(
        runtime_env=RuntimeEnv(
            env_vars={'CONFIG_FILEPATH': CONSTANTS.EXPERIMENT.CONFIG_FILEPATH}
        )
    )

    hf_dataset_name = 'roneneldan/TinyStories'
    hf_dataset = datasets.load_dataset(hf_dataset_name)
    train_ds = ray.data.from_huggingface(hf_dataset['train'], concurrency=2).limit(10)

    train_ds = train_ds.map_batches(TokenToLatents, batch_size=5, concurrency=1, num_gpus=CONSTANTS.EXPERIMENT.NUM_GPUS_ACTIVATION)


    trainer = TorchTrainer(
        train_loop_per_worker,
        # train_loop_config=ExperimentConfig(),
        scaling_config=ray.train.ScalingConfig(
            num_workers=CONSTANTS.EXPERIMENT.NUM_TRAINERS,
            use_gpu=True,
            resources_per_worker={'GPU': 0.5}
            # resources_per_worker={'GPU': (NUM_GPUS - ACTIVATION_GPU) / NUM_TRAINERS}
        ),
        # run_config = RunConfig(
        #     checkpoint_config=CheckpointConfig(num_to_keep=1),
        #     storage_path="s3://..."
        # )
        datasets={'train': train_ds}
    )
    result: ray.train.Result = trainer.fit()


    return result




if __name__ == '__main__':
    main()
