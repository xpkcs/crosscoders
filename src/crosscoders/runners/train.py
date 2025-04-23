



from typing import Dict

import torch
from crosscoders.dataclasses.autoencoders import LossMetrics
from crosscoders.utils import dataclass_to_dict, flatten_dict





# class RayReportingStrategy(ReportingStrategy):
#     def __init__(self):
#         self.is_initialized = False

#     def initialize(self, config: Dict[str, Any]) -> None:
#         self.is_initialized = True
#         # Ray-specific initialization if needed

#     def report_metrics(self, metrics: Dict[str, Any], checkpoint_dir: Optional[str] = None) -> None:
#         if checkpoint_dir:
#             checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
#             ray.train.report(metrics, checkpoint=checkpoint)
#         else:
#             ray.train.report(metrics)

#     def finalize(self) -> None:
#         # Clean up any resources
#         pass



class Trainer:

    # def __init__(self, config, trainer, data_loader_factory, model_factory, backend, reporting_strategy):
    def __init__(self, config, backend):

        self.config = config
        # self.trainer = trainer
        # self.data_loader_factory = data_loader_factory
        # self.model_factory = model_factory
        self.backend = backend
        # self.reporting_strategy = reporting_strategy

        # self.train_loader = None
        # self.val_loader = None



        # self.model = hydra.utils.instantiate(cfg.runner.crosscoder).model
        # self.optimizer = hydra.utils.instantiate(cfg.runner.optimizer, params=list(self.model.parameters()))
        # self.scheduler = self.configure_schedulers()

        self.num_tokens_processed: int = 0
        self.batch_idx: int = 0

        self.model = None
        self.optimizer = None

    def initialize(self, model_config):
        self.model = self.model_factory(**model_config)
        self.optimizer = self.optimizer_factory(
            self.model.parameters(),
            **model_config.get("optimizer_params", {})
        )
        # self.backend = self.backend.initialize()


    # def configure_schedulers(self):

    #     return {
    #         'lr'      : get_scheduler_lr(self.optimizer),
    #         'lambda_s': get_scheduler_lambda_s(self.cfg.runner.crosscoder.hps.lambda_s)
    #     }


    # def configure_optimizers(self):

    #     return self.cfg.OPTIMIZER.optimizer(
    #         self.model.parameters(),
    #         **self.cfg.OPTIMIZER.parameters.asdict()
    #     )


    # def forward(self, batch):

    #     return self.model(batch)


    # def train_batch(self, batch):

    #     ...

    # def post_train_batch(self):
    #     ...


    # def train_epoch(self):
    #     ...

    # def post_train_epoch(self):
    #     ...

    # def evaluate(self):
    #     ...


    def fit(self, dl, **kwargs):

        for epoch_idx in range(EPOCHS):

            for batch_idx, batch in enumerate(dl):

                metrics, report = self.train_batch(batch)

                ray.train.report(report)


            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(temp_checkpoint_dir, 'model.pt')
                )
                ray.train.report(
                    report,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )


        return report












# class CrosscoderRunner:


#     def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:

#         x     = self.cfg.runner.crosscoder.hps.x_scalar * batch[self.cfg.runner.input_name]
#         y     = self.cfg.runner.crosscoder.hps.y_scalar * batch[self.cfg.runner.output_name]
#         y_hat = self.model(x)

#         loss, metrics = self.model.loss(y, y_hat, lambda_s=self.scheduler['lambda_s'].get_lambda_s())
#         loss.backward()

#         total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

#         self.optimizer.step()
#         self.optimizer.zero_grad()

#         self.num_tokens_processed += batch[self.cfg.runner.input_name].shape[0]

#         report = (
#             {'training_iteration': self.num_tokens_processed} |
#             flatten_dict(dataclass_to_dict(metrics)) |
#             flatten_dict(
#                 {
#                     'lr'      : self.scheduler['lr'].get_last_lr()[0],
#                     'lambda_s': self.scheduler['lambda_s'].get_lambda_s()
#                 }
#             )
#         )

#         self.scheduler['lr'].step()
#         self.scheduler['lambda_s'].step()


#         return metrics, report




# class CrossLayerTranscoderRunner(AutoencoderRunnerABC):


#     def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:
#         ...