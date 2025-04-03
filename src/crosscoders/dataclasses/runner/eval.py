'''
These runners are separated from the other runner.py file so that when we import
from runner.py we don't need to have torch installed. Importing this file requires
torch b/c of BaselineModuleConfig.
'''


from dataclasses import dataclass, field

from crosscoders.dataclasses.runner.data import DataRunnerConfig
from crosscoders.dataclasses.runner.train import TrainRunnerConfig


__all__ = ['EvalRunnerConfig']







@dataclass
class EvalRunnerConfig(DataRunnerConfig, TrainRunnerConfig):

    stage: str = 'eval'

    # batch_size : int = 25000
