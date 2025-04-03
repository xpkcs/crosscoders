

from pprint import pformat, pprint
from crosscoders.data.dataset import Dataset
from crosscoders.config import *
# from crosscoders.data.preprocessing import SequenceMetadataBatch
# from crosscoders.runners import Runner
# from crosscoders.dataclasses.config import Config
# from crosscoders.utils import instantiate
from crosscoders.io import ZarrIO

CONFIG: Config = get_config()





# zarr_dir = f'{CONFIG.paths._Paths__s3_prefix}/{CONFIG.paths.activations_dir}'
# zarr_dir = '/home/ec2-user/crosscoders/zarr_dir'
batch_size = 10



import ray
import zarr
import numpy as np
from collections import defaultdict







@ray.remote(num_cpus=2, memory=7 * 1024 * 1024 * 1024)
class PartitionWriter:

    def __init__(self, zarr_dir, layer, activation_type, d_model):

        self.root = zarr.group(zarr_dir, overwrite=False)
        self.array = self.root[f'layer={layer}/activation_type={activation_type}/raw']
        # self.offset = self.array.shape[0]
        self.offset = 0
        # self.offset = {(l, at): 0 for l in CONFIG.activations.layers for at in CONFIG.activations.types}

        self.d_model = d_model


    # def resize(self, n_tokens):

    #     self.offset += n_tokens

    #     for at in CONFIG.activations.types:
    #         self.array[f'activation_type={at}/raw'].resize((self.offset, self.d_model))


    def append_batch(self, rows):
        """
        rows: a list of dictionaries, each with keys 'activations', 'layer', 'activation_type'
        Returns a list of meta updates, each as [start_index, n_tokens]
        """
        meta_updates = []

        # for l in CONFIG.activations.layers:
        #     for at in CONFIG.activations.types:

        for row in rows:
            n_tokens = row.shape[0]

            self.array.resize((self.offset + n_tokens, self.d_model))
            self.array[self.offset:self.offset + n_tokens] = row

            meta_updates += [[self.offset, n_tokens]]
            self.offset += n_tokens
        return meta_updates


@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
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


root, zarrays = ZarrIO.init(f'{CONFIG.paths.prefix}/{CONFIG.paths.zarr_dir}')
print(f'{CONFIG.paths.prefix}/{CONFIG.paths.zarr_dir}')
print(pformat(zarrays))
print(pformat(root.tree()))


# -----------------------------------------------------------------------------
# Assume CONFIG is available and zarr_dir is the path or URL to your Zarr store.
# Create one partition writer per (layer, activation_type)
partition_writers = {}
for layer in CONFIG.activations.layers:
    for activation_type in CONFIG.activations.types:
        partition_writers[(layer, activation_type)] = PartitionWriter.remote(
            f'{CONFIG.paths.prefix}/{CONFIG.paths.zarr_dir}', layer, activation_type, CONFIG.language_model.d_model
        )
# partition_writers = PartitionWriter.remote(zarr_dir, CONFIG.language_model.d_model)
# partition_writers = {}
# for layer in CONFIG.activations.layers:
#     partition_writers[layer] = PartitionWriter.remote(
#         zarr_dir, layer, CONFIG.language_model.d_model
#     )
    # partition_writers[layer] = PartitionWriter.remote(
    #     zarr_dir, layer, CONFIG.language_model.d_model
    # )

# Create the meta writer actor.
meta_writer = MetaWriter.remote(f'{CONFIG.paths.prefix}/{CONFIG.paths.zarr_dir}')


@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
def __call__(batch):


    activations = defaultdict(list)
    # tokens = defaultdict(lambda: {
    #     'activation_type': [],
    #     'activations': []
    # })

    for i in range(batch['tokens'].shape[0]):
        # row = {k: v[i] for k, v in batch.items()}

        for l in CONFIG.activations.layers:
            for at in CONFIG.activations.types:

                activations[(l, at)] += [batch[at][i][:(batch['tokens'][i] != batch['tokens'][i][0]).sum() + 1,l,:]]
                # activations[l] += [{
                # #     'seq_id': row['seq_id'].item(),
                # #     'pos_id': row['pos_id'].item(),
                #     # 'layer': l,
                #     'activation_type': at,
                #     'activations': row[at][:(row['tokens'] != row['tokens'][0]).sum() + 1,l,:]
                # }]
                # tokens[l]['activation_type'] += [at]
                # tokens[l]['activations'] += [row[at][:,l,:]]

    # print(tokens)
    print('done building remote args')

    # # done, ready = ray.wait([
    # #     partition_writers[key].append_batch.remote(rows)
    # #     for key, rows in activations.items()
    # # ], num_returns=1)
    # # print(done)

    seq_metadata = ray.get([
        partition_writers[key].append_batch.remote(rows)
        for key, rows in activations.items()
    ])

    # # seq_metadata = ray.get(done[0])
    # print(pformat(seq_metadata))

    # # ready += [meta_writer.append_batch.remote(seq_metadata)]
    ray.get(meta_writer.append_batch.remote(seq_metadata[0]))

    # print('done with batch')




def main(cfg):


    ds = Dataset(CONFIG.dataset)


    # 1. Create a placement group for GPU resources
    pg = ray.util.placement_group(
        bundles=[{'GPU': 1, 'CPU': 8, 'memory': 20 * 1024 * 1024 * 1024}],
        strategy='STRICT_PACK'  # Ensures GPU and CPU are on same node
    )

    # 2. Wait for placement group to be ready
    ray.get(pg.ready())
    print('Placement group is ready')

    # Add a small delay to ensure scheduler registration
    import time;        time.sleep(5)



    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    ray_ds = ds.load('tokens', PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0))
    # ray_ds = ray_ds.map_batches(SequenceMetadataBatch, concurrency=1)
    # ds.save(ray_ds)


    # Control the number of concurrent batch processing tasks
    max_concurrent = 8  # With 1 CPUs per task
    wait_for_tasks = 1

    pending_tasks = []
    completed_tasks = 0
    total_tasks = 0


    for batch_idx, batch in enumerate(ray_ds.iter_batches(batch_size=batch_size, batch_format='default')):

        pending_tasks += [__call__.remote(batch)]


        if len(pending_tasks) >= max_concurrent:
            done, pending_tasks = ray.wait(pending_tasks, num_returns=wait_for_tasks)
            ray.get(done)

            completed_tasks += wait_for_tasks


            print(f'finished {completed_tasks} # of tasks')


    if pending_tasks:
        done = ray.get(pending_tasks)

        completed_tasks += len(done)

        print(f'finished {completed_tasks} # of tasks')