

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
from crosscoders.data.actors import ChunkIndexer, Indexer
from crosscoders.data.preprocessing import CountTokensBatch, IndexBatch, TokenToActivations
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

        print(f'reading from {bucket_name} @ {key_prefix}')

        keys = get_s3_keys(bucket_name=bucket_name, key_prefix=key_prefix)
        # self.rng.shuffle(keys)    # does this matter for ray?

        return ray.data.read_parquet_bulk(
            keys[:100],
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
        # self.indexer_chunk = ChunkIndexer.remote(100000)


    def load(self, which: Literal['tokens', 'activations'] = 'tokens', scheduling_strategy='DEFAULT') -> ray.data.Dataset:

        # ds = self.cfg.datasource
        ds = hydra.utils.call(self.cfg.datasource)

        # ds = ds.limit(CONFIG.batch.n_records)


        n_tokens = 0

        match which:
            case 'tokens':

                # ray.get(self.indexer_seq.reset.remote())
                # ray.get(self.indexer_token.reset.remote())

                self.indexer_seq = Indexer.remote()
                self.indexer_token = Indexer.remote()

                ds = ds.map_batches(IndexBatch, fn_args=(self.indexer_seq,), batch_size=10000, concurrency=16, num_cpus=1)


                # n_tokens = ds.map_batches(
                #     CountTokensBatch,
                #     batch_size=CONFIG.batch.batch_size,
                #     concurrency=(1,8),
                #     # num_gpus=1,
                #     num_cpus=4,
                #     # resources = {'gpu_node': 1},
                #     # memory=28*1024*1024*1024,
                #     zero_copy_batch=True,
                #     scheduling_strategy=scheduling_strategy,
                #     # runtime_env={
                #     #     'env_vars': {'CUDA_VISIBLE_DEVICES': ''}
                #     # }
                # ).sum('n_tokens')

                ds = ds.map_batches(
                    TokenToActivations,
                    fn_args=(self.indexer_token,),
                    batch_size=CONFIG.batch.batch_size,
                    concurrency=4,
                    # num_cpus=4,
                    num_gpus=1,
                    num_cpus=1,
                    # resources = {'gpu_node': 1},
                    # memory=28*1024*1024*1024,
                    zero_copy_batch=True,
                    scheduling_strategy=scheduling_strategy,
                    # runtime_env={
                    #     'env_vars': {'CUDA_VISIBLE_DEVICES': ''}
                    # }
                )

                # ds = ds.limit(CONFIG.batch.n_records * len(CONFIG.activations.types) * len(CONFIG.activations.layers))


            case 'activations':
                pass




        return ds, n_tokens


    def save(self, ds: ray.data.Dataset) -> None:

        print(f'saving activations @ {CONFIG.paths.activations_dir}', flush=True)
        # print(f'saving activations @ /home/ec2-user/crosscoders/{CONFIG.paths.activations_dir}', flush=True)

        ds.write_parquet(
            CONFIG.paths.activations_dir,
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
