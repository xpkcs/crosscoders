


from crosscoders.data.dataset import Dataset


class DefaultTrainStrategy:

    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self, fit_loop):

        ds = Dataset(self.cfg.dataset, 'activations')
        print(ds.schema())



        # train_dl = None
        # val_dl = None
        # model = None
        # loss_fn = None
        # optimizer = None
        # schedulers = None

        # # return fit_loop(train_dl, val_dl, model, loss_fn, optimizer, schedulers, cfg)
        # return fit_loop(model)

from ray.train import ScalingConfig, RunConfig, Result
from ray.train.torch import TorchTrainer
from omegaconf import OmegaConf

class RayTrainStrategy:

    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self, fit_loop):


        def train_loop_per_worker(cfg):

            print('hi')
            # return fit_loop()

        # self.cfg.dataset.slice = 'train'
        train_ds = Dataset(self.cfg.dataset, 'activations')._load()
        # self.cfg.dataset.slice = 'eval'
        # val_ds = Dataset(self.cfg.dataset, 'activations')



        trainer = TorchTrainer(
            train_loop_per_worker,
            # train_loop_config=OmegaConf.to_container(cfg),
            scaling_config=ScalingConfig(
                use_gpu=True,
                num_workers=1,
                resources_per_worker={'CPU': 1, 'GPU': 1}
            ),
            run_config = RunConfig(
                # checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),
                # storage_path='s3://crosscoders/ray/tiny-stories-33M'
                storage_path=f'{self.cfg.paths._Paths__local_prefix}/ray_results/test'
            ),
            datasets={'train': train_ds}
        )
        result: Result = trainer.fit()


