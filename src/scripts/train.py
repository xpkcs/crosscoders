



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
from crosscoders.dataclasses.configs.runner import ModelConfig
from crosscoders.autoencoders.acausal import AcausalAutoencoderLightningModule

from crosscoders import CONSTANTS
from crosscoders.dataclasses.configs.runner import AutoencoderLightningModuleConfig
from crosscoders.data.dataset import TinyStoriesRayDataset
from crosscoders.utils import from_dict, get_config


import os

import tempfile


import torch.amp, torch.optim
import datetime, numpy as np





# def collate_fn(batch):
#     batch_ = {}
#     for k, v in batch.items():
#         b, l, d = v.shape[0], max(_.shape[0] for _ in v), v[0].shape[1]
#         batch_[k] = torch.as_tensor(np.stack([np.pad(_, ((0, b), (0, l), (0, d))) for _ in v]))

#     return batch_


def train_loop():
    

    train_ds = TinyStoriesRayDataset().load('activations')

    train_dl = train_ds.iter_torch_batches(
        # prefetch_batches=10,
        batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
        device='cuda'
    )



    cfg = from_dict(
        AutoencoderLightningModuleConfig,
        get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
    )

    model = AcausalAutoencoder(cfg.MODEL)
    model.to('cuda')

    criterion = AcausalLoss()

    # optimizer = cfg.OPTIMIZER.optimizer(
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg.OPTIMIZER.parameters.asdict()
    )

    scaler = torch.amp.GradScaler("cuda", enabled=False)


    for epoch in range(CONSTANTS.EXPERIMENT.NUM_EPOCHS):

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=5, warmup=10, active=25, repeat=0),
    #         # schedule=torch.profiler.schedule(wait=5, warmup=10, active=25, repeat=1),
    #         # schedule=torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H:%M:%S")}'),
    #         record_shapes=True,
    #         profile_memory=True,
    #         # with_stack=True,
    #         # with_modules=True,

    # ) as prof:

        model.train()
        for batch_idx, batch in enumerate(train_dl):
            
            # prof.step()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                outputs = model(batch['resid_post'])
                loss = criterion(outputs, batch['resid_post'], W_dec=model.W_dec, x_enc=model.x_enc)

            scaler.scale(loss.loss).backward()
            grad_norms = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
            print(np.mean(grad_norms), np.std(grad_norms), np.min(grad_norms), np.max(grad_norms))

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()


            if batch_idx % 1 == 0:
                metrics = {
                    'loss': loss.loss.item(),
                    'error': loss.error.item(),
                    'l1': loss.l1.item(),
                    'l0': loss.l0.item(),
                }
                print(batch_idx, metrics)





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



    # model = AcausalAutoencoderLightningModule(
    #     from_dict(
    #         AutoencoderLightningModuleConfig,
    #         get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
    #     )
    # )

    # model = AcausalAutoencoderLightningModule(
    #     AutoencoderLightningModuleConfig(
    #         model=ModelConfig('acausal', D_CODER=16384)
    #     )
    # )

    cfg = from_dict(
        AutoencoderLightningModuleConfig,
        get_config(CONSTANTS.CONFIG_FILEPATH).get('RUNNER', {})
    )

    model = AcausalAutoencoder(cfg.MODEL)

    criterion = AcausalLoss()

    optimizer = cfg.OPTIMIZER.optimizer(
        model.parameters(),
        **cfg.OPTIMIZER.parameters.asdict()
    )

    print(os.getcwd())


    # for epoch in range(1):

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./crosscoders'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:

        model.train()
        for batch_idx, batch in enumerate(train_dl):
            prof.step()
            # This is done by `prepare_data_loader`!
            # images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(batch['resid_post'])
            loss = criterion(outputs, batch['resid_post'], W_dec=model.W_dec, x_enc=model.x_enc)
            optimizer.zero_grad()
            loss.loss.backward()
            optimizer.step()


            if batch_idx % 1 == 0:
                metrics = {
                    'loss': loss.loss.item(),
                    'error': loss.error.item(),
                    'l1': loss.l1.item(),
                    'l0': loss.l0.item(),
                }
                ray.train.report(
                    metrics
                )




            # metrics = {
            #     'loss': loss.loss.item(),
            #     'error': loss.error.item(),
            #     'l1': loss.l1.item(),
            #     'l0': loss.l0.item(),
            # }
            # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(temp_checkpoint_dir, "model.pt")
            #     )
            #     ray.train.report(
            #         metrics,
            #         checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            #     )
















    # trainer = pl.Trainer(
    #     # max_epochs=10,
    #     max_epochs=CONSTANTS.EXPERIMENT.MAX_EPOCHS,
    #     devices='auto',
    #     accelerator='auto',
    #     # strategy=ray.train.lightning.RayDDPStrategy(),
    #     strategy=ray.train.lightning.RayDeepSpeedStrategy(),
    #     plugins=[ray.train.lightning.RayLightningEnvironment()],
    #     callbacks=[
    #         ray.train.lightning.RayTrainReportCallback(),
    #         # EarlyStopping(monitor='n_tokens_processed', stopping_threshold=100)
    #     ],
    #     enable_checkpointing=False,
    #     gradient_clip_val=0.5,
    #     log_every_n_steps=10,
    #     # accumulate_grad_batches=1,
    # )

    # trainer = ray.train.lightning.prepare_trainer(trainer)

    # trainer.fit(model, train_dataloaders=train_dl)
    # # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)




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
