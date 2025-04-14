



from functools import partial
from typing import Any, Dict, Iterable
import numpy as np
import ray
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from crosscoders.config import Config, get_config
from ray.experimental.tqdm_ray import tqdm
# from tqdm import tqdm
# from tqdm.std import tqdm as tqdm_type


CONFIG: Config = get_config()




def IndexBatch(batch, indexer):

    n_rows = batch['text'].shape[0]

    start_idx = ray.get(indexer.reserve.remote(n_rows))

    # return {'idx': [start_idx + i for i in range(n_rows)]}
    batch['seq_id'] = [start_idx + i for i in range(n_rows)]

    return batch



# class SequenceMetadataBatch:

#     def __init__(self):
#         torch.set_grad_enabled(False)

#         # device = torch.get_default_device()
#         # torch.set_default_device('cpu')
#         # self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device=CONFIG.globals.device).eval()
#         self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device='cpu', move_to_device=False).eval()
#         # torch.set_default_device(device)
#         self.model.to(CONFIG.globals.device)


#         self.pad_token = self.model.to_single_token(self.model.tokenizer.bos_token)

#         self.batch_out: Dict[str, np.ndarray] = {}
#         self.batch_offset_seq = 0


#     def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

#         tokens = self.model.to_tokens(batch['text'])

#         batch_size = tokens.shape[0]
#         max_seq_len = min(tokens.shape[1], CONFIG.batch.max_seq_len)

#         tokens = tokens[:,:max_seq_len]


#         mask = tokens != self.pad_token
#         mask[:, 0] = True

#         self.batch_offset_seq += batch_size


#         return {
#             'seq_id': np.arange(batch_size, dtype=np.int64) + self.batch_offset_seq - batch_size,
#             'seq_len': mask.sum(1).to(torch.int16).cpu().numpy()
#         }

import zarr

def PartitionWriter(batch, zarr_dir, start_idx, end_idx):

    root = zarr.open(zarr_dir)


    n_tokens = batch['tokens'].shape[0]

    for l in CONFIG.activations.layers:
        for at_idx, at in enumerate(CONFIG.activations.types):
            array = root[f'layer={l}/activation_type={at}/raw']

            n_tokens_curr, *dims = array.shape

            if end_idx > n_tokens_curr:
                array.resize((end_idx, *dims))

            array[start_idx:start_idx + n_tokens] = batch['activations'][:,l,at_idx,:]

    # return {'n_tokens': n_tokens}



class CountTokensBatch:

    def __init__(self):

        torch.set_grad_enabled(False)

        # device = torch.get_default_device()
        # torch.set_default_device('cpu')
        self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device='cpu').eval()
        # torch.set_default_device(device)


        self.pad_token = self.model.to_single_token(self.model.tokenizer.bos_token)


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # tokens = self.model.to_tokens(batch['text'])

        mask = self.model.to_tokens(batch['text'])[:,:CONFIG.batch.max_seq_len] != self.pad_token
        mask[:, 0] = True

        return {'n_tokens': [mask.sum().item()]}



# # @ray.remote(n_cpus=4, n_gpus=1, resources={'gpu_node': 1})
# class TokenMetaBatch:

#     def __init__(self):

#         torch.set_grad_enabled(False)

#         self.device = CONFIG.globals.device

#         # device = torch.get_default_device()
#         # torch.set_default_device('cpu')
#         self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device=self.device).eval()
#         # torch.set_default_device(device)



#         self.pad_token = self.model.to_single_token(self.model.tokenizer.bos_token)


#     def __call__(self, batch: Dict[str, np.ndarray], indexer) -> Dict[str, np.ndarray]:

#         tokens = self.model.to_tokens(batch['text'])

#         self.batch_size = tokens.shape[0]
#         self.max_seq_len = min(tokens.shape[1], CONFIG.batch.max_seq_len)

#         tokens = tokens[:,:self.max_seq_len]

#         mask = tokens != self.pad_token
#         mask[:, 0] = True

#         n_tokens = mask.sum().item()



#         batch_out = {}

#         start_idx = ray.get(indexer.reserve.remote(n_tokens))
#         batch_out['idx'] = start_idx + np.arange(n_tokens)
#         # batch_out['chunk_start_idx'] = [chunk_start_idx] * n_tokens




#         batch_out['seq_id'] = torch.repeat_interleave(torch.as_tensor(batch['seq_id'][:,None], device=tokens.device), self.max_seq_len, 1)
#         batch_out['token_pos'] = (mask.cumsum(1) - 1)


# @ray.remote(n_cpus=4, n_gpus=1, resources={'gpu_node': 1})
class TokenToActivations:

    def __init__(self,
        # layers: Iterable[int],
        # model_names: Iterable[str] = ('gpt2-small', 'gpt-neo-125M'),
        # model_names: Iterable[str] = ('tiny-stories-33M',),
        # latent_names: Iterable[str] = ('resid_mid', 'ln2.normalized', 'mlp_out', 'resid_post')
    ):
        torch.set_grad_enabled(False)

        self.device = CONFIG.globals.device

        self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device=self.device).eval()
        self.hooks = [
            (
                get_act_name(at, l) if '.' not in at else
                get_act_name(at.split('.')[1], l, at.split('.')[0]),
                partial(self.store_activation_hook, activation_type=at_idx, layer_idx=l)
            )
            for l in range(self.model.cfg.n_layers)
            for at_idx, at in enumerate(CONFIG.activations.types)
        ]


        self.pad_token = self.model.to_single_token(self.model.tokenizer.bos_token)
        self.n_activation_vectors = len(CONFIG.activations.layers) * len(CONFIG.activations.types)

        self.buffer = torch.empty(
            (CONFIG.batch.batch_size, CONFIG.batch.max_seq_len, self.model.cfg.n_layers, len(CONFIG.activations.types), self.model.cfg.d_model),
            dtype=torch.float32,
            device=torch.device(self.device)
        )

        self.batch_out: Dict[str, np.ndarray] = {}


    def __call__(self, batch: Dict[str, np.ndarray], indexer) -> Dict[str, np.ndarray]:

        tokens = self.model.to_tokens(batch['text'])

        self.batch_size = tokens.shape[0]
        self.max_seq_len = min(tokens.shape[1], CONFIG.batch.max_seq_len)

        tokens = tokens[:,:self.max_seq_len]


        with torch.inference_mode():
            _ = self.model.run_with_hooks(
                tokens,
                fwd_hooks=self.hooks
            )
            # _ = self.model(tokens)


        batch_out = {}


        mask = tokens != self.pad_token
        mask[:, 0] = True

        # mask = batch_out['tokens'] != self.pad_token
        # mask[:, 0] = True
        # mask = mask.cpu().numpy()
        n_tokens = mask.sum().item()


        # chunk_start_idx, chunk_end_idx = ray.get(indexer.reserve.remote(n_tokens))
        start_idx = ray.get(indexer.reserve.remote(n_tokens))
        # batch_out['idx'] = start_idx + torch.arange(n_tokens)
        batch_out['idx'] = start_idx + np.arange(n_tokens)
        # batch_out['chunk_start_idx'] = [chunk_start_idx] * n_tokens



        mask_np = mask.cpu()

        batch_out['seq_id'] = batch['seq_id'][:,None].repeat(self.max_seq_len, 1)[mask_np]
        batch_out['token_pos'] = (mask_np.cumsum(1) - 1)[mask_np]
        # batch_out['seq_id'] = torch.as_tensor(batch['seq_id'][:,None], device=tokens.device).repeat(1, self.max_seq_len)
        # batch_out['token_pos'] = mask.cumsum(1) - 1
        # batch_out['seq_id'] = torch.repeat_interleave(torch.as_tensor(batch['seq_id'][:,None], device=tokens.device), self.max_seq_len, 1)
        # batch_out['token_pos'] = (mask.cumsum(1) - 1)


        batch_out['token'] = tokens

        batch_out['activation'] = self.buffer[:self.batch_size,:self.max_seq_len,...]

        for k, v in batch_out.items():
            if isinstance(v, torch.Tensor):
                match v.ndim:
                    case 1:
                        batch_out[k] = v.cpu().numpy()
                    case _:
                        batch_out[k] = v[mask].cpu().numpy()

        # self.buffer.zero_()

        # raise NotImplementedError({k: v.shape for k,v in batch_out.items()})


        return batch_out


    def store_activation_hook(self, activation, hook, activation_type, layer_idx):

        self.buffer[:self.batch_size,:self.max_seq_len,layer_idx,activation_type,:] = activation