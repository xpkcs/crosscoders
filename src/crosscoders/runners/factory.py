# factories.py
from typing import Dict, Any, Type
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from crosscoders.runners.engine import RayBackend
from crosscoders.runners.strategy import TokensToActivationsStrategy
from crosscoders.runners.runner import DataProcessingRunner, TrainRunner
from crosscoders.runners.trainer import Trainer




class MapFactory:

    _map = {}


    @classmethod
    def create(cls, cfg):

        try:
            object_class = cls._map[cfg.type]

        except ConfigAttributeError as e:
            raise e

        except KeyError:
            raise ValueError(f'unsupported backend type: {cfg.type}')

        else:
            return object_class(cfg)


class BackendFactory(MapFactory):

    _map = {
        'ray': RayBackend
    }


class DataStrategyFactory(MapFactory):

    _map = {
        'TokensToActivations': TokensToActivationsStrategy
        # 'Shuffle': None,
        # 'Stats': None,
    }



class AutoencoderFactory(MapFactory):

    _map = {
        'crosscoder': None,
        'crosslayertranscoder': None,
    }


class RunnerFactory:
    """Factory for creating runners."""

    @classmethod
    def create_data_processing_runner(cls, cfg) -> DataProcessingRunner:
        """Create data processing runner with strategy from cfg."""

        backend = BackendFactory.create(cfg.backend)
        strategy = DataStrategyFactory.create(cfg.data_strategy)


        return DataProcessingRunner(cfg, backend, strategy)


    @classmethod
    def create_training_runner(cls, cfg) -> TrainingRunner:

        # data_loader_factory = get_data_loader_factory(cfg)
        # optimizer_factory = get_optimizer_factory(cfg.training.optimizer)

        backend = BackendFactory.create(cfg.backend)
        # reporting_strategy = ReportingStrategyFactory.create_strategy(cfg)

        # trainer = Trainer(cfg, data_loader_factory, model_factory, backend, reporting_strategy)
        trainer = Trainer(cfg, AutoencoderFactory)

        return TrainingRunner(cfg, trainer, backend)