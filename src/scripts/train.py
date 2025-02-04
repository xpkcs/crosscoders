



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
from crosscoders.configs import ModelConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoderLightningModule

from crosscoders import CONSTANTS
from crosscoders.configs import AutoencoderLightningModuleConfig
from crosscoders.data.dataset import TinyStoriesRayDataset
from crosscoders.utils import from_dict, get_config




# def collate_fn(batch):
#     batch_ = {}
#     for k, v in batch.items():
#         b, l, d = v.shape[0], max(_.shape[0] for _ in v), v[0].shape[1]
#         batch_[k] = torch.as_tensor(np.stack([np.pad(_, ((0, b), (0, l), (0, d))) for _ in v]))

#     return batch_


def train_loop_per_worker():

    # dataloader
    train_dl = ray.train.get_dataset_shard('train').iter_torch_batches(
        batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
        # collate_fn=lambda _: {k: torch.as_tensor(np.stack(v)) for k, v in _.items()},
        # collate_fn=lambda _: {k: torch.as_tensor(np.stack([np.pad(_, ()) for _ in v])) for k, v in _.items()},
        # local_shuffle_buffer_size=16
    )
    # valid_dl = ray.train.get_dataset_shard('valid').iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)

    # train_dl = train_ds.iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)
    # val_dl = val_ds.iter_torch_batches(batch_size=cfg.EXPERIMENT.BATCH_SIZE)



    model = AcausalAutoencoderLightningModule(
        from_dict(
            AutoencoderLightningModuleConfig, 
            get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
        )
    )

    # model = AcausalAutoencoderLightningModule(
    #     AutoencoderLightningModuleConfig(
    #         model=ModelConfig('acausal', D_CODER=16384)
    #     )
    # )


    trainer = pl.Trainer(
        # max_epochs=10,
        max_epochs=CONSTANTS.EXPERIMENT.MAX_EPOCHS,
        devices='auto',
        accelerator='auto',
        # strategy=ray.train.lightning.RayDDPStrategy(),
        strategy=ray.train.lightning.RayDeepSpeedStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[
            ray.train.lightning.RayTrainReportCallback(),
            # EarlyStopping(monitor='n_tokens_processed', stopping_threshold=100)
        ],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=False,
        gradient_clip_val=0.5,
        log_every_n_steps=10,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_dl)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)




def main():

    # hf_dataset_name = 'roneneldan/TinyStories'
    # hf_dataset = datasets.load_dataset(hf_dataset_name)
    # train_ds = ray.data.from_huggingface(hf_dataset['train'], concurrency=1)

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




# if __name__ == '__main__':
#     main()
