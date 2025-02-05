

from typing import Literal
import boto3
import datasets
import ray
import ray.data

from crosscoders.constants import CONSTANTS
from crosscoders.data.preprocessing import TokenToLatents



def get_s3_keys(bucket_name, key_prefix):

    bucket = boto3.resource('s3').Bucket(bucket_name)

    return [
        f's3://{obj.bucket_name}/{obj.key}'
        for obj in bucket.objects.filter(Prefix=key_prefix, Marker=key_prefix)
    ]


class TinyStoriesRayDataset:

    def __init__(self, hf_dataset_name: str = 'roneneldan/TinyStories', slice: str = 'train', bucket_name: str = 'crosscoders') -> None:

        self.hf_dataset_name = hf_dataset_name
        self.slice = slice
        self.s3_prefix = f'input/{self.hf_dataset_name}/train/'
        self.bucket_name = bucket_name


    def load(self, which: Literal['tokens', 'activations'] = 'tokens') -> ray.data.Dataset:

        match which:
            case 'tokens':
                hf_dataset = datasets.load_dataset(self.hf_dataset_name, streaming=True)
                ds = ray.data.from_huggingface(hf_dataset[self.slice], concurrency=1)

            case 'activations':
                ds = ray.data.read_parquet(get_s3_keys(self.bucket_name, self.s3_prefix))


        if CONSTANTS.EXPERIMENT.MAX_RECORDS:
            ds = ds.limit(CONSTANTS.EXPERIMENT.MAX_RECORDS)

        match which:
            case 'tokens':
                ds = ds.map_batches(
                    TokenToLatents,
                    batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
                    concurrency=1,
                    num_gpus=1,
                    num_cpus=1
                )


        return ds


    def save(self, ds: ray.data.Dataset) -> None:

        ds.write_parquet(
            # f'local://{CONSTANTS.DATA_DIR}/input/{self.hf_dataset_name}/train',
            f's3://{self.bucket_name}/{self.s3_prefix}',
            compression='LZ4',
            concurrency=6,
            ray_remote_args={
                'num_cpus': 1
            },
        )
