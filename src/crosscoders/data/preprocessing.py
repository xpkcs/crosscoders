



from typing import Dict
import numpy as np
import torch

from transformer_lens import HookedTransformer

from crosscoders.configs.globals import HardwareConfig






class TokenToLatents:

    def __init__(self, model: str = 'gpt2-small'):

        self.model = HookedTransformer.from_pretrained(model)

        self.layer_hooks = [
            (f"blocks.{layer_idx}.hook_resid_post", self.store_resid_post_activation)
            for layer_idx in range(self.model.cfg.n_layers)
        ]

        # self.latent_names = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
        self.latent_names = ('resid_post',)


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # tokenize, add tokens to batch dict
        batch['tokens'] = self.model.to_tokens(batch['text'].tolist())

        # batch_size, seq_len = batch['tokens'].shape

        self._init_tensors(*batch['tokens'].shape)


        # get latents/activations

        # _ = self.model(batch['tokens'])
        _ = self.model.run_with_hooks(
            batch['tokens'],
            fwd_hooks=self.layer_hooks
        )

        batch |= {k: v.permute(1, 2, 0, 3).cpu().numpy().astype(np.float32) for k, v in self.latents.items()}

        self._delete_tensors()


        # with torch.inference_mode():

        #     # TODO: can make way more efficient
        #     logits, cache = self.model.run_with_cache(batch['tokens'])    # maybe replace with run_with_hooks?


        del batch['text']
        del batch['tokens']




        # convert tensors to cpu/numpy to be serialized for ray comms
        for k in batch:
            # if k in self.latent_names:
            #     batch[k] = batch[k].permute(1, 2, 0, 3).cpu().numpy().astype(np.float32)
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cpu().numpy().astype(np.float32)


        return batch


    def _delete_tensors(self):

        del self.latents


    def _init_tensors(self, batch_size: int, seq_len: int):

        self.latents = {}

        # compose tensors for desired latent_names, add to latents dict
        for ln in self.latent_names:
            self.latents[ln] = torch.empty(
                (self.model.cfg.n_layers, batch_size, seq_len, self.model.cfg.d_model),
                **HardwareConfig().asdict()
            )


    def store_resid_post_activation(self, tensor, hook):
        """
        Stores the resid_post activation in a dict.

        `hook.name` is something like "blocks.0.hook_resid_post".
        """
        # global batch

        layer_idx, latent_name = (lambda _: [int(_[1]), _[2][5:]])(hook.name.split('.'))

        self.latents[latent_name][layer_idx,...] = tensor.detach()