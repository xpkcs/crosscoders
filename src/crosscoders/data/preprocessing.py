



from typing import Dict
import numpy as np
import torch

from transformer_lens import HookedTransformer


from crosscoders.configs import HardwareConfig




class TokenToLatents:

    def __init__(self, model: str | torch.nn.Module = 'gpt2-small'):

        if type(model) == str:
            self.model = HookedTransformer.from_pretrained(model)
        else:
            self.model = model

        # self.latent_names = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
        self.latent_names = ('resid_post',)


    def __call__(self, batch: Dict[str, np.ndarray]):

        # tokenize, add tokens to batch dict
        batch['tokens'] = self.model.to_tokens(batch['text'].tolist())
        batch_size, seq_len = batch['tokens'].shape


        # get latents/activations
        with torch.inference_mode():

            # TODO: can make way more efficient
            logits, cache = self.model.run_with_cache(batch['tokens'])    # maybe replace with run_with_hooks?


        # compose tensors for desired latent_names, add to batch dict
        for ln in self.latent_names:
            batch[ln] = torch.empty((self.model.cfg.n_layers, batch_size, seq_len, self.model.cfg.d_model), **HardwareConfig().asdict())

        for k,v in cache.items():
            if k.startswith('block') and k.endswith(self.latent_names):

                layer_idx, latent_name = (lambda _: [int(_[1]), _[2][5:]])(k.split('.'))

                batch[latent_name][layer_idx,...] = v


        del batch['text']
        # del batch['tokens']


        # convert tensors to cpu/numpy for ray
        for k in batch:
            if k in self.latent_names:
                batch[k] = batch[k].permute(1, 2, 0, 3).cpu().numpy()
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cpu().numpy()


        return batch