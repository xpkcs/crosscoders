

import datasets
import ray

from crosscoders.constants import CONSTANTS
from crosscoders.data.preprocessing import TokenToLatents




class TinyStoriesRayDataset:

    def __init__(self, hf_dataset_name: str = 'roneneldan/TinyStories', slice: str = 'train'):

        self.hf_dataset_name = hf_dataset_name
        self.slice = slice


    def load(self):

        hf_dataset = datasets.load_dataset(self.hf_dataset_name)
        ds = ray.data.from_huggingface(hf_dataset[self.slice], concurrency=1)

        if CONSTANTS.EXPERIMENT.MAX_RECORDS:
            ds = ds.limit(CONSTANTS.EXPERIMENT.MAX_RECORDS)


        ds = ds.map_batches(
            TokenToLatents,
            batch_size=CONSTANTS.EXPERIMENT.BATCH_SIZE,
            concurrency=1,
            num_gpus=CONSTANTS.EXPERIMENT.NUM_GPUS_ACTIVATION - 0.01
        )


        return ds


    def save(self, ds):

        ds.write_parquet(f'local://{CONSTANTS.PROJECT_ROOT_DIR}/data/input/{self.hf_dataset_name}/train', compression='LZ4')
