

from abc import abstractmethod
from typing import Literal
import boto3
import datasets
import hydra
import ray
import ray.data
import numpy as np
import torch

from omegaconf import MISSING, OmegaConf

from crosscoders.config import get_config
from crosscoders.data.preprocessing import TokenToActivations
from crosscoders.dataclasses.config import Config
from crosscoders.dataclasses.dataset import DatasetConfig

import numpy as np


CONFIG: Config = get_config()




def get_s3_keys(bucket_name, key_prefix):

    # TODO: logging

    bucket = boto3.resource('s3').Bucket(bucket_name)

    return [
        f's3://{obj.bucket_name}/{obj.key}'
        for obj in bucket.objects.filter(Prefix=key_prefix, Marker=key_prefix, Delimiter='/')
    ]




class Datasource:

    @abstractmethod
    def _load():
        ...


class S3Datasource(Datasource):

    @abstractmethod
    def _load(bucket_name, key_prefix, **kwargs):

        keys = get_s3_keys(bucket_name=bucket_name, key_prefix=key_prefix)
        # self.rng.shuffle(keys)    # does this matter for ray?

        return ray.data.read_parquet_bulk(
            keys,
            ray_remote_args={'num_cpus': 1},
            shuffle=ray.data.FileShuffleConfig(seed=CONFIG.globals.seed)
        )



class HuggingFaceDatasource(Datasource):

    _target_: str = 'crosscoders.data.dataset.HuggingFaceDatasource._load'

    # org: str = MISSING
    # repo: str = MISSING
    # slice: str = MISSING


    @abstractmethod
    # def _load(org, repo, slice, **kwargs):
    def _load(**kwargs):

        device = torch.get_default_device()
        torch.set_default_device('cpu')

        hf_dataset = datasets.load_dataset(f"{kwargs['org']}/{kwargs['repo']}", streaming=True)
        ds = ray.data.from_huggingface(hf_dataset[kwargs['slice']])

        torch.set_default_device(device)


        return ds


class Dataset:

    def __init__(self, cfg: DatasetConfig):

        self.cfg: DatasetConfig = cfg


    # def _load_tokens(self):

    #     ds = hydra.utils.call(self.cfg.datasource)

    #     return ds.map_batches(
    #         TokenToActivations,
    #         batch_size=self.cfg.batch_size,
    #         concurrency=1,
    #         num_gpus=1,
    #         num_cpus=1
    #     )


    # def _load_activations(self):

    #     ds = S3Datasource._load(
    #         bucket_name=CONFIG.globals.s3_bucket,
    #         key_prefix=CONFIG.paths.activations_dir.replace(f's3://{CONFIG.globals.s3_bucket}', '')
    #     )

    #     return ds


    def load(self, which: Literal['tokens', 'activations'] = 'tokens') -> ray.data.Dataset:

        # ds = hydra.utils.call(self.cfg.datasource)
        ds = self.cfg.datasource


        # ds = getattr(self, f'_load_{which}')()
        match which:
            case 'tokens':
                ds = ds.map_batches(
                    TokenToActivations,
                    batch_size=self.cfg.batch_size,
                    concurrency=1,
                    num_gpus=1,
                    num_cpus=1
                )

            case 'activations':
                pass


        if self.cfg.max_tokens:
            ds = ds.limit(self.cfg.max_tokens)


        return ds


    def save(self, ds: ray.data.Dataset) -> None:

        print(f'saving activations @ {CONFIG.paths._Paths__prefix}/{CONFIG.paths.activations_dir}', flush=True)

        ds.write_parquet(
            f'{CONFIG.paths._Paths__prefix}/{CONFIG.paths.activations_dir}',
            compression='zstd',
            # min_rows_per_file=8192,
            ray_remote_args={
                'num_cpus': 1
            },
        )
