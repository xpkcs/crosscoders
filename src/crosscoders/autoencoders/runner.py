



import gc
import os
import tempfile
from typing import Dict

import hydra
import ray
import ray.train
import ray.tune
import torch

# from crosscoders import CONSTANTS
from crosscoders.abc import AutoencoderRunnerABC
from crosscoders.autoencoders.schedulers import (get_scheduler_lambda_s,
                                                 get_scheduler_lr)
from crosscoders.config import get_config
from crosscoders.dataclasses.autoencoders import LossMetrics
from crosscoders.dataclasses.config import Config
# from crosscoders.dataclasses.configs.runner import RunnerConfig
# from crosscoders.dataclasses.metrics.loss import LossMetrics
from crosscoders.utils import dataclass_to_dict, flatten_dict

CONSTANTS = get_config()


# @dataclass(repr=False)
# class RunnerConfig(DataclassABC):

#     # model: Literal['baseline', 'jumprelu'] = 'baseline'
#     recipe: str = 'baseline'
#     model: Any = None

#     OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)

#     INPUT_NAME: str = 'resid_post'
#     OUTPUT_NAME: str = 'resid_post'

#     X_SCALAR: float = 1.
#     Y_SCALAR: float = 1.


#     def __post_init__(self):

#         match self.recipe:
#             case 'baseline':
#                 self.model = BaselineModelConfig(lambda_s=2)
#             case 'jumprelu':
#                 self.model = JumpReLUModelConfig(lambda_s=10)


class Runner(AutoencoderRunnerABC):


    cfg: Config

    def __init__(self, cfg: Config) -> None:

        super().__init__(cfg)


        self.model = hydra.utils.instantiate(cfg.runner.crosscoder).model
        self.optimizer = hydra.utils.instantiate(cfg.runner.optimizer, params=list(self.model.parameters()))
        self.scheduler = self.configure_schedulers()

        self.num_tokens_processed: int = 0
        self.batch_idx: int = 0


    def configure_schedulers(self):

        return {
            'lr'      : get_scheduler_lr(self.optimizer),
            'lambda_s': get_scheduler_lambda_s(self.cfg.runner.crosscoder.hps.lambda_s)
        }


    def training_step(self, batch: Dict[str, torch.Tensor]) -> LossMetrics:

        x     = self.cfg.runner.crosscoder.hps.x_scalar * batch[self.cfg.runner.input_name]
        y     = self.cfg.runner.crosscoder.hps.y_scalar * batch[self.cfg.runner.output_name]
        y_hat = self.model(x)

        loss, metrics = self.model.loss(y, y_hat, lambda_s=self.scheduler['lambda_s'].get_lambda_s())
        loss.backward()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.num_tokens_processed += batch[self.cfg.runner.input_name].shape[0]

        report = (
            {'training_iteration': self.num_tokens_processed} |
            flatten_dict(dataclass_to_dict(metrics)) |
            flatten_dict(
                {
                    'lr'      : self.scheduler['lr'].get_last_lr()[0],
                    'lambda_s': self.scheduler['lambda_s'].get_lambda_s()
                }
            )
        )

        self.scheduler['lr'].step()
        self.scheduler['lambda_s'].step()


        return metrics, report


    def fit(self, dl, **kwargs):

        for batch_idx, batch in enumerate(dl):

            metrics, report = self.training_step(batch)

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


    def cleanup(self):

        """Free GPU memory after training"""
        # 1. Move model to CPU
        if hasattr(self, 'model'):
            self.model.cpu()

        # 2. Clear optimizer state
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad(set_to_none=True)
            del self.optimizer
            self.optimizer = None

        # 3. Delete any cached data, tensors, or intermediate results
        for attr in dir(self):
            if isinstance(getattr(self, attr), torch.Tensor):
                delattr(self, attr)

        # 4. Run garbage collection
        gc.collect()

        # 5. Empty CUDA cache
        torch.cuda.empty_cache()

        # 6. Optional: Ensure all CUDA ops are finished
        torch.cuda.synchronize()