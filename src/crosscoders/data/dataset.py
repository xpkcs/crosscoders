

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

import numpy as np


# CONFIG = get_config()




def get_s3_keys(bucket_name, key_prefix):

    # TODO: logging

    bucket = boto3.resource('s3').Bucket(bucket_name)

    return [
        f's3://{obj.bucket_name}/{obj.key}'
        for obj in bucket.objects.filter(Prefix=key_prefix, Marker=key_prefix, Delimiter='/')
    ]


class Dataset:

    def __init__(self, **kwargs):
        self.cfg = kwargs

    # def __init__(self, name: str = 'roneneldan/TinyStories', tag: str = '', slice: str = 'train', activations_dir: str = '') -> None:

    #     self.name = name
    #     self.slice = slice
    #     self.activations_dir = activations_dir
    #     # self.rng = np.random.default_rng(seed=CONFIG.seed)


    # @staticmethod
    # def instantiate(cfg):

    #     print('> instantiating dataset:')
    #     print(OmegaConf.to_yaml(cfg, resolve=True), end='\n\n')
    #     ds = hydra.utils.instantiate(cfg)


    #     return ds


    # def load(self, which: Literal['tokens', 'activations'] = 'tokens') -> ray.data.Dataset:

    #     match which:
    #         case 'tokens':
    #             device = torch.get_default_device()
    #             torch.set_default_device('cpu')

    #             hf_dataset = datasets.load_dataset(self.name, streaming=True)
    #             ds = ray.data.from_huggingface(hf_dataset[self.slice])

    #             # if CONFIG.runner.max_records:
    #             #     ds = ds.limit(CONFIG.runner.max_records)

    #             ds = ds.map_batches(
    #                 TokenToActivations,
    #                 batch_size=CONFIG.runner.batch_size,
    #                 # concurrency=(1, 2),
    #                 # num_gpus=0.5,
    #                 concurrency=1,
    #                 num_gpus=1,
    #                 num_cpus=1
    #             )

    #             torch.set_default_device(device)


    #         case 'activations':
    #             keys = get_s3_keys(CONFIG.s3_bucket, self.prefix)
    #             self.rng.shuffle(keys)

    #             ds = ray.data.read_parquet_bulk(
    #                 keys,
    #                 ray_remote_args={'num_cpus': 1},
    #                 shuffle=ray.data.FileShuffleConfig(seed=CONFIG.seed)
    #             )


    #     if CONFIG.runner.max_tokens:
    #         ds = ds.limit(CONFIG.runner.max_tokens)


    #     return ds




class TokensToActivationsDataset(Dataset):

    def load(self) -> ray.data.Dataset:

        device = torch.get_default_device()
        torch.set_default_device('cpu')

        hf_ds = datasets.load_dataset(self.cfg['name'], streaming=True)
        ds = ray.data.from_huggingface(hf_ds[self.cfg['slice']])

        ds = ds.map_batches(
            TokenToActivations,
            batch_size=self.cfg['batch_size'],
            # concurrency=(1, 2),
            concurrency=1,
            num_gpus=1,
            num_cpus=2
        )

        torch.set_default_device(device)


        if self.cfg['max_tokens']:
            ds = ds.limit(self.cfg['max_tokens'])


        return ds


    def save(self, ds: ray.data.Dataset) -> None:

        print(f'saving activations @ {self.cfg['activations_dir']}', flush=True)

        ds.write_parquet(
            self.cfg['activations_dir'],
            compression='zstd',
            # min_rows_per_file=8192,
            ray_remote_args={
                'num_cpus': 1
            },
        )



class ActivationsDataset(Dataset):

    def load(self) -> ray.data.Dataset:

        path = self.activations_dir.split('/')
        keys = get_s3_keys(bucket_name=path[2], key_prefix='/'.join(path[3:]))
        self.rng.shuffle(keys)

        ds = ray.data.read_parquet_bulk(
            keys,
            ray_remote_args={'num_cpus': 1},
            # shuffle=ray.data.FileShuffleConfig(seed=CONFIG.seed)
        )


        # if CONFIG.runner.max_tokens:
        #     ds = ds.limit(CONFIG.runner.max_tokens)


        return ds