



# from crosscoders.abc.runner import RunnerABC
from crosscoders.data.dataset import Dataset
from crosscoders.dataclasses.runner import DataStrategyConfig, RayBackendConfig
# from crosscoders.runners.data import RayDataRunner

from crosscoders.config import Config, get_config

CONFIG: Config = get_config()

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional




class Backend(ABC):
    """Interface for different data processing backends."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the processing environment."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...


class RayBackend(Backend):


    def __init__(self, cfg) -> None:
        self.cfg: RayBackendConfig = cfg



    def initialize(self) -> None:
        pass

        # import ray, ray.data, ray.runtime_env

        # if self.cfg['job'] == JOB_TYPE_ENUM.ray:



        #     # ray.data._internal.datasource.parquet_datasource.NUM_CPUS_FOR_META_FETCH_TASK = 4
        #     # ray.data.datasource.parquet_meta_provider.RETRY_MAX_ATTEMPTS_FOR_META_FETCH_TASK = 256
        #     # ray.data.datasource.parquet_meta_provider.RETRY_MAX_BACKOFF_S_FOR_META_FETCH_TASK = 256

        #     runtime_env_kwargs = {} | kwargs

        #     ray.init(
        #         runtime_env=ray.runtime_env.RuntimeEnv(
        #             # env_vars={
        #             #     'CONFIG_PATH': CONFIG.CONFIG_FILEPATH,
        #             #     # 'RAY_DEBUG': '1'
        #             # },
        #             # py_executable_args=["-Xfrozen_modules=off"]
        #         )
        #     )


    def shutdown(self) -> None:
        pass

        # import ray

        # ray.shutdown()


class DataStrategy(ABC):

    def __init__(self, cfg) -> None:
        self.cfg: DataStrategyConfig = cfg


    @abstractmethod
    def execute(self) -> Optional[Any]:
        ...


class TokensToActivationsStrategy(DataStrategy):

    def execute(self) -> None:

        print(f'saving activations @ {self.cfg.activations_path}', flush=True)


        Dataset(CONFIG.dataset).write_parquet(
            self.cfg.activations_path,
            **{
                'compression': 'zstd',
                # 'min_rows_per_file': 8192,
                'ray_remote_args': {
                    'num_cpus': 1
                }
            } # | kwargs
        )



