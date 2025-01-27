



import datasets
import lightning as pl
import ray, ray.train, ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.runtime_env import RuntimeEnv

import numpy as np
import torch


from crosscoders.data.preprocessing import TokenToLatents
from crosscoders.configs import ExperimentConfig, ModelConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoderLightningModule

from crosscoders import CONSTANTS
from crosscoders.configs import AutoencoderLightningModuleConfig






def train_loop_per_worker():

    # dataloader
    train_dl = ray.train.get_dataset_shard('train').iter_torch_batches(
        batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
        collate_fn=lambda _: {k: torch.as_tensor(np.stack(v)) for k, v in _.items()}
    )
    # valid_dl = ray.train.get_dataset_shard('valid').iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)

    # train_dl = train_ds.iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)
    # val_dl = val_ds.iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)


    model = AcausalAutoencoderLightningModule(
        AutoencoderLightningModuleConfig(
            model=ModelConfig('acausal')
        )
    )


    trainer = pl.Trainer(
        # max_epochs=10,
        max_epochs=CONSTANTS.EXPERIMENT.MAX_EPOCHS,
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
            env_vars={
                'CONFIG_FILEPATH': CONSTANTS.CONFIG_FILEPATH,
                # 'RAY_DEBUG': '1'
            },
            # py_executable_args=["-Xfrozen_modules=off"]
        )
    )


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
            resources_per_worker={'GPU': 0.5}
            # resources_per_worker={'GPU': round((CONSTANTS.EXPERIMENT.NUM_GPUS - CONSTANTS.EXPERIMENT.NUM_GPUS_ACTIVATION) / CONSTANTS.EXPERIMENT.NUM_TRAINERS, 2)}
            # resources_per_worker={'GPU': round((CONSTANTS.EXPERIMENT.NUM_GPUS - CONSTANTS.EXPERIMENT.NUM_GPUS_ACTIVATION - 0.01) / CONSTANTS.EXPERIMENT.NUM_TRAINERS, 2)}
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
