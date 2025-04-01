

from pprint import pformat, pprint
from crosscoders.data.dataset import Dataset
from crosscoders.config import *
from crosscoders.data.preprocessing import SequenceMetadataBatch
from crosscoders.runners import Runner
from crosscoders.utils import instantiate
from crosscoders.io import ZarrIO

CONFIG: Config = get_config()





zarr_dir = f'{CONFIG.paths._Paths__s3_prefix}/{CONFIG.paths.activations_dir}'
zarr_dir = '/home/ec2-user/crosscoders/zarr_dir'
batch_size = 5



import ray
import zarr
import numpy as np
from collections import defaultdict


@ray.remote(num_cpus=1, memory=2 * 1024 * 1024 * 1024)
class PartitionWriter:

    def __init__(self, zarr_dir, d_model):

        self.root = zarr.group(zarr_dir, overwrite=False)
        self.array = self.root
        # self.offset = self.array.shape[0]
        self.offset = 0
        self.offset = {(l, at): 0 for l in CONFIG.activations.layers for at in CONFIG.activations.types}

        self.d_model = d_model


    # def resize(self, n_tokens):

    #     self.offset += n_tokens

    #     for at in CONFIG.activations.types:
    #         self.array[f'activation_type={at}/raw'].resize((self.offset, self.d_model))


    def append_batch(self, batch):
        """
        rows: a list of dictionaries, each with keys 'activations', 'layer', 'activation_type'
        Returns a list of meta updates, each as [start_index, n_tokens]
        """
        meta_updates = []

        # for l in CONFIG.activations.layers:
        #     for at in CONFIG.activations.types:

        # for row in rows:
        #     n_tokens = row['activations'].shape[0]

        #     self.array[f'layer={row["layer"]}/activation_type={row["activation_type"]}/raw'].resize((self.offset[(row['layer'], row['activation_type'])] + n_tokens, self.d_model))
        #     self.array[f'layer={row["layer"]}/activation_type={row["activation_type"]}/raw'][self.offset[(row['layer'], row['activation_type'])]:self.offset[(row['layer'], row['activation_type'])] + n_tokens] = row['activations']

        #     meta_updates += [[self.offset[(row['layer'], row['activation_type'])], n_tokens]]
        #     self.offset[(row['layer'], row['activation_type'])] += n_tokens
        return meta_updates


@ray.remote(num_cpus=0.1, memory=0.5 * 1024 * 1024 * 1024)
class MetaWriter:

    def __init__(self, zarr_dir):

        self.root = zarr.group(zarr_dir, overwrite=False)
        self.meta = self.root['meta/sequences']
        self.offset = self.meta.shape[0]


    def append_batch(self, meta_updates):
        """
        meta_updates: a list of [start_index, n_tokens] entries.
        """
        n_new = len(meta_updates)
        self.meta.resize((self.offset + n_new, 2))
        self.meta[self.offset:self.offset + n_new] = meta_updates
        self.offset += n_new



# -----------------------------------------------------------------------------
# Assume CONFIG is available and zarr_dir is the path or URL to your Zarr store.
# Create one partition writer per (layer, activation_type)
partition_writers = {}
# for layer in CONFIG.activations.layers:
#     for activation_type in CONFIG.activations.types:
#         partition_writers[(layer, activation_type)] = PartitionWriter.remote(
#             zarr_dir, layer, activation_type, CONFIG.language_model.d_model
#         )
partition_writers = PartitionWriter.remote(zarr_dir, CONFIG.language_model.d_model)
# partition_writers = {}
# for layer in CONFIG.activations.layers:
#     partition_writers[layer] = PartitionWriter.remote(
#         zarr_dir, layer, CONFIG.language_model.d_model
#     )
    # partition_writers[layer] = PartitionWriter.remote(
    #     zarr_dir, layer, CONFIG.language_model.d_model
    # )

# Create the meta writer actor.
meta_writer = MetaWriter.remote(zarr_dir)




def main(cfg):


    ds = Dataset(CONFIG.dataset)

    ray_ds = ds.load('tokens')
    # ray_ds = ray_ds.map_batches(SequenceMetadataBatch, concurrency=1)
    # ds.save(ray_ds)


    root, zarrays = ZarrIO.init(zarr_dir)
    print(pformat(zarrays))
    print(pformat(root.tree()))


    # Control the number of concurrent batch processing tasks
    max_concurrent = 1  # With 1 CPUs per task

    pending_tasks = []
    completed_tasks = 0
    total_batches = 0


    for batch_idx, batch in enumerate(ray_ds.iter_batches(batch_size=batch_size, batch_format='default')):



        pass
        # if batch_idx == 10:
        #     break

        # print({k: len(v) for k, v in batch.items()})

        # pending_tasks += [partition_writers.append_batch.remote(batch)]


        # if len(pending_tasks) >= max_concurrent:
        #     done, pending_tasks = ray.wait(pending_tasks, num_returns=max_concurrent)
        #     seq_metadata = ray.get(done)

        # print(pformat(seq_metadata))



        # break