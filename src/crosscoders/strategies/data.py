



# from crosscoders.abc.runner import RunnerABC
from crosscoders.data.dataset import Dataset
from crosscoders.dataclasses.runner import DataStrategyConfig, RayBackendConfig
# from crosscoders.runners.data import RayDataRunner

from crosscoders.config import Config, get_config

CONFIG: Config = get_config()

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional



class DataStrategy(ABC):

    # def __init__(self, cfg) -> None:
    #     self.cfg: DataStrategyConfig = cfg


    @abstractmethod
    def execute(self) -> Optional[Any]:
        ...


class TokensToActivationsStrategy(DataStrategy):

    def __init__(self, activations_path):
        self.activations_path = activations_path

    def execute(self) -> None:

        print(f'saving activations @ {self.activations_path}', flush=True)


        Dataset(CONFIG.dataset).write_parquet(
            self.activations_path,
            **{
                'compression': 'zstd',
                # 'min_rows_per_file': 8192,
                'ray_remote_args': {
                    'num_cpus': 1
                }
            } # | kwargs
        )



