

from abc import abstractmethod
from typing import Any




class Runner:

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
            self.logger.info(f'Starting runner')
            self._pre_run(*args, **kwargs)
            result = self._run(*args, **kwargs)
            self._post_run(result, *args, **kwargs)
            self.logger.info(f'Finished runner')

            return result

        except Exception as e:
            self.logger.error(f'Error in runner: {e}')
            raise


    @abstractmethod
    def _pre_run(self, *args, **kwargs) -> Any | None:
        ...

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any | None:
        ...

    @abstractmethod
    def _post_run(self, *args, **kwargs) -> Any | None:
        ...


class DataProcessingRunner(Runner):
    '''Runner for data processing with swappable underlying backend framework (e.g., Ray, Spark) and processing strategy.'''

    def __init__(self, cfg, backend, strategy):

        super().__init__(cfg)
        self.backend = backend
        self.strategy = strategy

    def _pre_run(self) -> None:

        self.logger.info(f'Initializing processing backend {self.backend.__class__.__name__}')
        self.backend.initialize()

    def _run(self) -> Any:
        '''Process data'''

        self.logger.info(f'Processing data with backend {self.backend.__class__.__name__}')
        self.logger.info(f'Executing strategy {self.strategy.__class__.__name__}')

        return self.strategy.execute()

    def _post_run(self, result) -> None:
        '''Clean up resources'''

        self.logger.info('Shutting down processing backend')
        self.backend.shutdown()


class TrainingRunner(Runner):
    '''Runner for model training'''

    def __init__(self, cfg, trainer) -> None: #, trainer, data_loader_factory) -> None:

        super().__init__(cfg)
        self.trainer = trainer
        # self.data_loader_factory = data_loader_factory

        # self.train_loader = None
        # self.val_loader = None


    def _pre_run(self) -> None:
        '''Set up training environment'''

        self.logger.info('Initializing training environment')
        # self.trainer.initialize(self.cfg.get('model_cfg', {}))

        # # Create data loaders
        # train_data, val_data = data
        # self.train_loader = self.data_loader_factory(
        #     train_data,
        #     **self.cfg.get('train_loader_params', {})
        # )
        # self.val_loader = self.data_loader_factory(
        #     val_data,
        #     **self.cfg.get('val_loader_params', {})
        # )

    def _run(self) -> Any:
        '''Run training loop'''

        # num_epochs = self.cfg.get('num_epochs', 10)
        # results = []

        # for epoch in range(num_epochs):
        #     self.logger.info(f'Starting epoch {epoch+1}/{num_epochs}')

        #     # Train
        #     train_metrics = self.trainer.train_epoch(self.train_loader)

        #     # Evaluate
        #     val_metrics = self.trainer.evaluate(self.val_loader)

        #     results.append({
        #         'epoch': epoch + 1,
        #         'train_metrics': train_metrics,
        #         'val_metrics': val_metrics
        #     })

        # return results






