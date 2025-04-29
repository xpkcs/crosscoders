from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig
from hydra.utils import instantiate


class Runner(ABC):

    def __init__(self, cfg) -> None:

        self.name = self.__class__.__name__
        self.cfg = cfg

        self.logger = self._setup_logger()

    def _setup_logger(self) -> None:

        import logging

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        return logger

    def run(self, *args, **kwargs) -> Any:

        try:
            self.logger.info(f"Starting runner")
            self._pre_run(*args, **kwargs)
            result = self._run(*args, **kwargs)
            self._post_run(result, *args, **kwargs)
            self.logger.info(f"Finished runner")

            return result

        except Exception as e:
            self.logger.error(f"Error in runner: {e}")
            raise

    def _pre_run(self) -> None:
        pass

    @abstractmethod
    def _run(self) -> Any: ...

    def _post_run(self, result: Any) -> None:
        pass


class DataRunner(Runner):
    """Runner for data processing with swappable underlying backend framework (e.g., Ray, Spark) and processing strategy."""

    def __init__(self, cfg):

        super().__init__(cfg)
        # self.backend = backend
        self.strategy = instantiate(cfg.runner.strategy)

    # def _pre_run(self) -> None:

        # self.logger.info(
        #     f"Initializing processing backend {self.backend.__class__.__name__}"
        # )
        # self.backend.initialize()

    def _run(self) -> Any:
        """Process data"""

        # self.logger.info(
        #     f"Processing data with backend {self.backend.__class__.__name__}"
        # )
        self.logger.info(f"Executing strategy {self.strategy.__class__.__name__}")

        return self.strategy.execute()

    def _post_run(self, result) -> None:
        """Clean up resources"""

        self.logger.info("Shutting down processing backend")
        # self.backend.shutdown()


class TrainRunner(Runner):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)
        self.strategy = instantiate(cfg.runner.strategy, cfg)
        self.fit_loop = instantiate(cfg.runner.fit_loop)

    # def _pre_run(self) -> None:
    #     self.model_factory = None
    #     self.optimizer_factory = None
    #     self.scheduler_factory = None
    #     self.loss_fn = None

    def _run(self) -> Any:

        return self.strategy.execute(
            fit_loop=self.fit_loop,
            # model_factory=self.model_factory,
            # optimizer_factory=self.optimizer_factory,
            # scheduler_factory=self.scheduler_factory,
            # loss_fn=self.loss_fn,
        )
