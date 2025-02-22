



from functools import partial
from typing import Dict, Iterable
import numpy as np
import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from crosscoders.dataclasses.configs.globals import HardwareConfig






class TokenToActivations:

    # def __init__(self, model_names: Iterable[str] = ('gpt2-small', 'gpt-neo-125M')):
    def __init__(self, model_names: Iterable[str] = ('tiny-stories-33M',)):

        self.models = {}
        for mn in model_names:
            self.models[mn]['model'] = HookedTransformer.from_pretrained(mn)
            self.models[mn]['hooks'] = [
                (
                    get_act_name(ln, layer_idx) if '.' not in ln else get_act_name(ln.split('.')[1], layer_idx, ln.split('.')[0]),
                    partial(self.store_activation_hook, model_name=mn, latent_name=ln, layer_idx=layer_idx)
                )
                for layer_idx in range(self.models[mn]['model'].cfg.n_layers)
                for ln in self.latent_names
            ]

        # self.latent_names = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
        # self.latent_names = ('resid_post',)
        # self.latent_names = ('resid_mid', 'mlp_out')
        self.latent_names = ('ln2.normalized', 'mlp_out', 'resid_post')

        torch.set_grad_enabled(False)


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        tokenizer_model = next(iter(self.models.values()))

        self.out = {
            'tokens': tokenizer_model.to_tokens(batch['text'].tolist())
        }


        for mn, model_info in self.models.items():

            # compose tensors for desired latent_names, add to latents dict
            for ln in self.latent_names:
                self.out[f'{mn}.{ln}'] = torch.empty(
                    (*self.out['tokens'].shape, model_info['model'].cfg.n_layers, model_info['model'].cfg.d_model),
                    **HardwareConfig().asdict()
                )

            with torch.inference_mode():
                _ = model_info['model'].run_with_hooks(
                    self.out['tokens'],
                    fwd_hooks=model_info['hooks']
                )


        bos_token = tokenizer_model.to_single_token(tokenizer_model.tokenizer.bos_token)
        col_indices = torch.arange(self.out['tokens'].shape[1]).unsqueeze(0).expand_as(self.out['tokens']).to(self.out['tokens'].device)
        mask = (col_indices != 0) & (self.out['tokens'] != bos_token)

        for k, v in self.out.items():
            self.out[k] = v[mask]


        # convert tensors to cpu/numpy to be serialized for ray comms
        for k in self.out:
            if isinstance(self.out[k], torch.Tensor):
                # TODO: replace this with specified types per key
                if k == 'tokens':
                    self.out[k] = self.out[k].cpu().numpy().astype(np.int32)
                else:
                    self.out[k] = self.out[k].cpu().numpy().astype(np.float32)


        out = self.out
        del self.out


        return out


    # def _delete_tensors(self):

    #     del self.latents


    # def _init_tensors(self, batch_size: int, seq_len: int):

    #     self.latents = {}

    #     # compose tensors for desired latent_names, add to latents dict
    #     for ln in self.latent_names:
    #         self.latents[ln] = torch.empty(
    #             (self.model.cfg.n_layers, batch_size, seq_len, self.model.cfg.d_model),
    #             **HardwareConfig().asdict()
    #         )


    def store_activation_hook(self, activation, hook, model_name, latent_name, layer_idx):

        # layer_idx, latent_name = (lambda _: [int(_[1]), _[2][5:]])(hook.name.split('.'))

        self.out[f'{model_name}.{latent_name}'][...,layer_idx,:] = activation.detach()