



from functools import partial
from typing import Dict, Iterable
import numpy as np
import torch

from transformer_lens import HookedTransformer

from crosscoders.dataclasses.configs.globals import HardwareConfig






class TokenToActivations:

    def __init__(self, model_names: Iterable[str] = ('gpt2-small', 'gpt-neo-125M')):

        self.models = {
            mn: HookedTransformer.from_pretrained(mn)
            for mn in model_names
        }

        # self.latent_names = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
        self.latent_names = ('resid_post',)

        torch.set_grad_enabled(False)


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        tokenizer_model = next(iter(self.models.values()))

        self.out = {
            'tokens': tokenizer_model.to_tokens(batch['text'].tolist())
        }


        for mn, model in self.models.items():


            # compose tensors for desired latent_names, add to latents dict
            for ln in self.latent_names:
                self.out[f'{mn}.{ln}'] = torch.empty(
                    (*self.out['tokens'].shape, model.cfg.n_layers, model.cfg.d_model),
                    **HardwareConfig().asdict()
                )

            with torch.inference_mode():
                _ = model.run_with_hooks(
                    self.out['tokens'],
                    fwd_hooks=[
                        (f'blocks.{layer_idx}.hook_{ln}', partial(self.store_activation_hook, model_name=mn, latent_name=ln, layer_idx=layer_idx))
                        for ln in self.latent_names
                        for layer_idx in range(model.cfg.n_layers)
                    ]
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