



from functools import partial
from typing import Any, Dict, Iterable
import numpy as np
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from crosscoders.config import Config, get_config
from ray.experimental.tqdm_ray import tqdm
# from tqdm import tqdm
# from tqdm.std import tqdm as tqdm_type


CONFIG: Config = get_config()






class TokenToActivations:

    def __init__(self):

        torch.set_grad_enabled(False)

        device = torch.get_default_device()
        torch.set_default_device('cpu')
        self.model = HookedTransformer.from_pretrained(CONFIG.language_model.name, device=CONFIG.globals.device).eval()
        torch.set_default_device(device)

        self.hooks = [
            (
                get_act_name(at, l) if '.' not in at else
                get_act_name(at.split('.')[1], l, at.split('.')[0]),
                partial(self.store_activation_hook, activation_type=at, layer_idx=l)
            )
            for l in range(self.model.cfg.n_layers)
            for at in CONFIG.activations.types
        ]

        self.bos_token = self.model.to_single_token(self.model.tokenizer.bos_token)

        self.buffers: Dict[str, np.ndarray] = {
                # 'tokens': torch.empty(
                #     (CONFIG.batch.batch_size, CONFIG.batch.max_seq_len),
                #     dtype=torch.int32
                # ),
                'activations': torch.empty(
                    (CONFIG.batch.batch_size, CONFIG.batch.max_seq_len, self.model.cfg.n_layers, len(CONFIG.activations.types), self.model.cfg.d_model),
                    dtype=torch.float32
                )
            }

        self.batch_out: Dict[str, np.ndarray] = {}


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        tokens = self.model.to_tokens(batch['text'])

        self.batch_size = tokens.shape[0]
        self.max_seq_len = min(tokens.shape[1], CONFIG.batch.max_seq_len)

        tokens = tokens[:,:self.max_seq_len]
        tokens.to(torch.int32)


        # mask = tokens != self.bos_token
        # mask[:, 0] = True


        # mask = (tokens != self.bos_token)
        # mask[:, 0] = True
        # last_idx = mask.sum(1)

        # assert (last_idx != 0).all(), 'found empty sequence in last_idx'

        self.batch_out = {
            'tokens': tokens,
        }

        # compose tensors for desired latent_names, add to latents dict
        for i, at in enumerate(CONFIG.activations.types):
            self.batch_out[at] = self.buffers['activations'][:self.batch_size,:self.max_seq_len,:,i,:]


        with torch.inference_mode():
            _ = self.model.run_with_hooks(
                tokens,
                fwd_hooks=self.hooks
            )

        # # if records as seqs
        # batch_out = {}
        # for k in self.batch_out:
        #     _ = self.batch_out[k].detach()
        #     batch_out[k] = [_[i , :idx.item() + 1].cpu().numpy() for i, idx in enumerate(last_idx)]
        # self.batch_out = {}

        mask = tokens != self.bos_token
        mask[:, 0] = True

        batch_out = {}
        for k, v in self.batch_out.items():
            batch_out[k] = v[mask].cpu().numpy()  # if records as seqs
            # batch_out[k] = v.cpu().numpy()
        self.batch_out = {}

        return batch_out


    def store_activation_hook(self, activation, hook, activation_type, layer_idx):

        self.batch_out[activation_type][...,layer_idx,:] = activation
