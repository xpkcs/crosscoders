

import os
# import fsspec
import numpy as np
# import s3fs
import zarr

from crosscoders.config import *

CONFIG: Config = get_config()




class ZarrIO:

    @staticmethod
    def reset(zarr_dir):

        if os.path.exists(zarr_dir):
            import shutil
            shutil.rmtree(zarr_dir)

        os.makedirs(zarr_dir)


    @staticmethod
    def get_root(zarr_dir):
        # Setup a Zarr store and group once (this can be done in a "driver" script)
        store = zarr.storage.LocalStore(zarr_dir)
        return zarr.group(store=store)


    @staticmethod
    def init(zarr_dir):

        # store = zarr.storage.LocalStore(zarr_dir)
        # store = zarr.storage.FsspecStore(fsspec.filesystem('s3', asynchronous=True), path='crosscoders/data/tiny-stories-v1/language_model=tiny-stories-33M/slice=train/tag=tiny-stories-33M-1B/')
        # root = zarr.group(store, overwrite=True)

        root = zarr.group(zarr_dir, overwrite=True)

        zarrays = {
            'activations': {},
            'meta/sequences': None
        }

        for layer in CONFIG.activations.layers:
            layer_zgroup = root.create_group(f'layer={layer}')
            for activation_type in CONFIG.activations.types:
                activation_type_zgroup = layer_zgroup.create_group(f'activation_type={activation_type}')

                zarrays['activations'][(layer, activation_type)] = activation_type_zgroup.create_array(
                    f'raw',
                    shape=(0, CONFIG.language_model.d_model),
                    chunks=(10000, CONFIG.language_model.d_model),
                    dtype=np.float32,
                )

        zarrays['meta/sequences'] = root.create_array(
            f'meta/sequences',
            shape=(0, 2),  # (start_idx, length)
            dtype=np.int32,
            chunks=(1000, 2)
        )


        return root, zarrays


    @staticmethod
    def add_row(zarrays, row):

        for activation_type, activations in row.items():

            if activation_type in ('tokens', 'sequence_id'): continue
            activation_type = activation_type[activation_type.find('.') + 1:]   # assuming lm name prefix'd

            for layer in range(CONFIG.language_model.n_layers):

                n_tokens = zarrays['activations'][(layer, activation_type)].shape[0]


                zarrays['activations'][(layer, activation_type)].resize((n_tokens + activations.shape[0], CONFIG.language_model.d_model))
                zarrays['activations'][(layer, activation_type)][-activations.shape[0]:] = activations[:,layer,:]

                # zarrays['sequences'][(layer, activation_type)].resize((zarrays['sequences'][(layer, activation_type)].shape[0] + 1, 2))
                # zarrays['sequences'][(layer, activation_type)][-1] = [n_tokens, activations.shape[0]]

        zarrays['meta/sequences'].resize((zarrays['meta/sequences'].shape[0] + 1, 2))
        zarrays['meta/sequences'][-1] = [n_tokens, activations.shape[0]]


    # @staticmethod
    # def add_batch(batch, zarr_dir):



    @staticmethod
    def get_activations(
        root,
        sequence_idxs: list[int],
        activation_types: list[str] = CONFIG.activations.types
    ):

        # with open(os.path.join(zarr_dir, 'metadata.json'), 'r') as infl:
        #     metadata = json.load(infl)



        batch = {}

        for activation_type in activation_types:

            batch[activation_type] = []


            for seq_id in sequence_idxs:

                if not (0 <= seq_id < CONFIG.batch.n_seqs):
                    continue

                token_start_idx, n_tokens = root['meta/sequences'][seq_id]


                activations = []
                for layer in CONFIG.activations.layers:
                    activations += [root[f'layer={layer}'][f'activation_type={activation_type}']['activations'][token_start_idx:token_start_idx + n_tokens]]


                batch[activation_type] += [np.stack(activations, axis=1)]

            batch[activation_type] = np.concatenate(batch[activation_type])


        return batch
