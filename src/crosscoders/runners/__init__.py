

from abc import abstractmethod

from crosscoders.dataclasses.runner import JOB_TYPE_ENUM




class Runner:

    def __init__(self, **kwargs: dict) -> None:

        self.cfg = kwargs

        # for k, v in kwargs.items():
        #     setattr(self, k, v)


    @abstractmethod
    def run(self) -> None:
        ...


class RayRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


        if self.cfg['job'] == JOB_TYPE_ENUM.ray:

            import ray, ray.data, ray.runtime_env


            # ray.data._internal.datasource.parquet_datasource.NUM_CPUS_FOR_META_FETCH_TASK = 4
            # ray.data.datasource.parquet_meta_provider.RETRY_MAX_ATTEMPTS_FOR_META_FETCH_TASK = 256
            # ray.data.datasource.parquet_meta_provider.RETRY_MAX_BACKOFF_S_FOR_META_FETCH_TASK = 256


            ray.init(
                runtime_env=ray.runtime_env.RuntimeEnv(
                    # env_vars={
                    #     'CONFIG_PATH': CONFIG.CONFIG_FILEPATH,
                    #     # 'RAY_DEBUG': '1'
                    # },
                    # py_executable_args=["-Xfrozen_modules=off"]
                )
            )