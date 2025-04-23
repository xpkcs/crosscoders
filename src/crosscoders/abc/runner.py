



# from abc import abstractmethod
# import os
# import tempfile

# import hydra
# import ray
# import torch
# # import lightning as pl


# # from crosscoders.abc import AutoencoderABC, LossABC
# from crosscoders.abc.base import BaseABC
# from crosscoders.autoencoders.schedulers import get_scheduler_lambda_s, get_scheduler_lr
# from crosscoders.dataclasses.config import Config
# # from crosscoders.dataclasses.configs.runner import RunnerConfig



# from abc import abstractmethod

# from crosscoders.dataclasses.runner import JOB_TYPE_ENUM




# class Runner:

#     def __init__(self, cfg) -> None:

#         # for k, v in cfg.items():
#         #     setattr(self, k, v)

#         self.cfg = cfg

#         self.logger = self._setup_logger()


#     def _setup_logger(self):

#         import logging

#         logger = logging.getLogger(self.__class__.__name__)

#         return logger


#     def run(self, *args, **kwargs) -> None:

#         self.logger.info(f"Starting runner")

#         try:
#             self._pre_run(*args, **kwargs)
#             result = self._run(*args, **kwargs)
#             self._post_run(result, *args, **kwargs)
#             self.logger.info(f"Finished runner")
#             return result

#         except Exception as e:
#             self.logger.error(f"Error in runner: {e}")
#             raise


#     @abstractmethod
#     def _pre_run(self, *args, **kwargs):
#         ...

#     @abstractmethod
#     def _run(self, *args, **kwargs):
#         ...

#     @abstractmethod
#     def _post_run(self, *args, **kwargs):
#         ...


# class DataProcessingRunner(Runner):
#     """Runner for data processing with pluggable backends."""

#     def __init__(self, cfg, processing_backend):
#         super().__init__(cfg)
#         self.backend = processing_backend

#     def _pre_run(self, *args, **kwargs):
#         """Initialize the processing backend."""
#         self.logger.info(f"Initializing {self.backend.__class__.__name__}")
#         self.backend.initialize(self.cfg)

#     def _execute(self, *args, **kwargs):
#         """Process data using the selected backend."""
#         self.logger.info(f"Processing data with {self.backend.__class__.__name__}")
#         return self.backend.execute()

#     def _post_run(self, result, *args, **kwargs):
#         """Clean up resources."""
#         self.logger.info("Shutting down processing backend")
#         self.backend.shutdown()



# class AutoencoderRunnerABC(BaseABC):


#     cfg: Config

#     def __init__(self, cfg: Config) -> None:

#         super().__init__(cfg)


#         self.model = hydra.utils.instantiate(cfg.runner.crosscoder).model
#         self.optimizer = hydra.utils.instantiate(cfg.runner.optimizer, params=list(self.model.parameters()))
#         self.scheduler = self.configure_schedulers()

#         self.num_tokens_processed: int = 0
#         self.batch_idx: int = 0


#     def configure_schedulers(self):

#         return {
#             'lr'      : get_scheduler_lr(self.optimizer),
#             'lambda_s': get_scheduler_lambda_s(self.cfg.runner.crosscoder.hps.lambda_s)
#         }


#     def configure_optimizers(self):

#         return self.cfg.OPTIMIZER.optimizer(
#             self.model.parameters(),
#             **self.cfg.OPTIMIZER.parameters.asdict()
#         )


#     def forward(self, batch):

#         return self.model(batch)


#     def training_step(self, batch):

#         ...


#     def fit(self, dl, **kwargs):

#         for batch_idx, batch in enumerate(dl):

#             metrics, report = self.training_step(batch)

#             ray.train.report(report)


#         with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#             torch.save(
#                 self.model.state_dict(),
#                 os.path.join(temp_checkpoint_dir, 'model.pt')
#             )
#             ray.train.report(
#                 report,
#                 checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
#             )


#         return report
