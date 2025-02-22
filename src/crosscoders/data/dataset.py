

from typing import Literal
import boto3
import datasets
import ray
import ray.data
import numpy as np

from crosscoders.constants import CONSTANTS
from crosscoders.data.preprocessing import TokenToActivations

import numpy as np


def get_s3_keys(bucket_name, key_prefix):

    bucket = boto3.resource('s3').Bucket(bucket_name)

    return [
        f's3://{obj.bucket_name}/{obj.key}'
        for obj in bucket.objects.filter(Prefix=key_prefix, Marker=key_prefix, Delimiter='/')
    ]


class TinyStoriesRayDataset:

    def __init__(self, hf_dataset_name: str = 'roneneldan/TinyStories', slice: str = 'train', s3_prefix: str = '', bucket_name: str = 'crosscoders') -> None:

        self.hf_dataset_name = hf_dataset_name
        self.slice = slice
        self.s3_prefix = f'input/{self.hf_dataset_name}/train/{s3_prefix}'
        self.bucket_name = bucket_name
        self.rng = np.random.default_rng(seed=314159)


    def load(self, which: Literal['tokens', 'activations'] = 'tokens') -> ray.data.Dataset:

        match which:
            case 'tokens':
                hf_dataset = datasets.load_dataset(self.hf_dataset_name, streaming=True)
                ds = ray.data.from_huggingface(hf_dataset[self.slice], concurrency=1)

                if CONSTANTS.EXPERIMENT.MAX_RECORDS:
                    ds = ds.limit(CONSTANTS.EXPERIMENT.MAX_RECORDS)

                ds = ds.map_batches(
                    TokenToActivations,
                    batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
                    # concurrency=(1, 2),
                    # num_gpus=0.5,
                    concurrency=1,
                    num_gpus=1,
                    num_cpus=1
                )


            case 'activations':
                keys = get_s3_keys(self.bucket_name, self.s3_prefix)
                self.rng.shuffle(keys)

                ds = ray.data.read_parquet_bulk(
                    keys,
                    # concurrency=1,
                    ray_remote_args={'num_cpus': 4},
                    shuffle=ray.data.FileShuffleConfig(seed=314159)
                )


        if CONSTANTS.EXPERIMENT.MAX_TOKENS:
            ds = ds.limit(CONSTANTS.EXPERIMENT.MAX_TOKENS)


        return ds


    def save(self, ds: ray.data.Dataset, local=False) -> None:

        if local:
            path = f'local://{CONSTANTS.DATA_DIR}/input/{self.hf_dataset_name}/train'
        else:
            path = f's3://{self.bucket_name}/{self.s3_prefix}'

        ds.write_parquet(
            path,
            compression='LZ4',
            concurrency=6,
            ray_remote_args={
                'num_cpus': 1
            },
        )
