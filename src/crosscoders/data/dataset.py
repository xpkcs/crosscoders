

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

    @abstractmethod
    # def _load(org, repo, slice, **kwargs):
    def _load(**kwargs):

        device = torch.get_default_device()
        torch.set_default_device('cpu')

        hf_dataset = datasets.load_dataset(f"{kwargs['org']}/{kwargs['repo']}", streaming=True, split=kwargs['slice'])
        ds = ray.data.from_huggingface(hf_dataset)

        torch.set_default_device(device)


        return ds


class Dataset:

    def __init__(self, cfg: DatasetConfig):

        self.cfg: DatasetConfig = cfg


    def load(self, which: Literal['tokens', 'activations'] = 'tokens', shuffle_level='sequences') -> ray.data.Dataset:

        # ds = self.cfg.datasource
        ds = hydra.utils.call(self.cfg.datasource)

        # ds = ds.limit(CONFIG.batch.n_records)


        match which:
            case 'tokens':
                ds = ds.map_batches(
                    TokenToActivations,
                    batch_size=CONFIG.batch.batch_size,
                    concurrency=1,
                    num_gpus=1,
                    num_cpus=1,
                    memory=10*1024*1024*1024,
                    zero_copy_batch=True
                )

                # ds = ds.limit(CONFIG.batch.n_records * len(CONFIG.activations.types) * len(CONFIG.activations.layers))


            case 'activations':
                pass




        return ds


    def save(self, ds: ray.data.Dataset) -> None:

        print(f'saving activations @ {CONFIG.paths._Paths__prefix}/{CONFIG.paths.activations_dir}', flush=True)
        # print(f'saving activations @ /home/ec2-user/crosscoders/{CONFIG.paths.activations_dir}', flush=True)

        ds.write_parquet(
            f'{CONFIG.paths._Paths__prefix}/{CONFIG.paths.activations_dir}',
            compression='zstd',
            # min_rows_per_file=8192,
            ray_remote_args={
                'num_cpus': 1
            },
        )


        # ds.write_parquet(
        #     f'{CONFIG.paths._Paths__prefix}/{CONFIG.paths.activations_dir}',
        #     compression='zstd',
        #     # min_rows_per_file=8192,
        #     min_rows_per_file=256,   # seqs
        #     concurrency=3,
        #     ray_remote_args={
        #         'num_cpus': 2,
        #         'memory': 4 * 1024 * 1024 * 1024
        #     },
        # )
