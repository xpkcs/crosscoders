



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

    def __init__(self,
        # layers: Iterable[int],
        # model_names: Iterable[str] = ('gpt2-small', 'gpt-neo-125M'),
        model_names: Iterable[str] = ('tiny-stories-33M',),
        latent_names: Iterable[str] = ('resid_mid', 'ln2.normalized', 'mlp_out', 'resid_post')
    ):
        torch.set_grad_enabled(False)

        self.latent_names = latent_names

        device = torch.get_default_device()
        torch.set_default_device('cpu')
        self.models = {}
        for mn in model_names:
            self.models[mn] = {}
            self.models[mn]['model'] = HookedTransformer.from_pretrained(mn, device=CONFIG.globals.device).eval()
            self.models[mn]['hooks'] = [
                (
                    get_act_name(ln, layer_idx) if '.' not in ln else get_act_name(ln.split('.')[1], layer_idx, ln.split('.')[0]),
                    partial(self.store_activation_hook, model_name=mn, latent_name=ln, layer_idx=layer_idx)
                )
                for layer_idx in range(self.models[mn]['model'].cfg.n_layers)
                for ln in self.latent_names
            ]
        torch.set_default_device(device)

        self.dtypes: Dict[str, str] = {
            'tokens': 'int32',
            **{f'{mn}.{ln}': 'float32' for mn in self.models for ln in self.latent_names}
        }

        self.pbars: Dict[str, Any] = {
            'batch': tqdm(position=0, desc='batch', total=CONFIG.batch.n_batches, unit='batch'),
            'seq'  : tqdm(position=1, desc='seq  ', total=CONFIG.batch.n_seqs, unit='seq'),
            'token': tqdm(position=2, desc='token', total=CONFIG.batch.n_tokens , unit='token'),
        }

        self.batch_out: Dict[str, np.ndarray] = {}


    def __call__(self, batch: Dict[str, np.ndarray], convert_to_np_first = False) -> Dict[str, np.ndarray]:

        # if self.pbars['batch']._closed:
        #     for pbar in self.pbars.values():
        #         pbar._dump_state(force_flush=True)
        #     raise RuntimeError

        tokenizer_model = next(iter(self.models.values()))['model']


        self.batch_out['tokens'] = tokenizer_model.to_tokens(batch['text'].tolist())
        self.batch_out['tokens'] = self.batch_out['tokens'][:,:CONFIG.batch.max_seq_len]


        for mn, model_info in self.models.items():

            # compose tensors for desired latent_names, add to latents dict
            for ln in self.latent_names:
                self.batch_out[f'{mn}.{ln}'] = torch.empty(
                    (*self.batch_out['tokens'].shape, model_info['model'].cfg.n_layers, model_info['model'].cfg.d_model),
                    device=CONFIG.globals.device
                )

            with torch.inference_mode():
                logits = model_info['model'].run_with_hooks(
                    self.batch_out['tokens'],
                    fwd_hooks=model_info['hooks']
                )


        bos_token = tokenizer_model.to_single_token(tokenizer_model.tokenizer.bos_token)


        # # TODO: so hacky

        # # convert tensors to cpu/numpy to be serialized for ray comms
        # for k in self.out:
        #     if isinstance(self.out[k], torch.Tensor):

        #         cpu_tensor = self.out[k].cpu()

        #         # TODO: replace this with specified types per key
        #         match k:
        #             case 'tokens':
        #                 target_dtype = np.int32
        #             case _:
        #                 target_dtype = np.float32

        #         if cpu_tensor.dtype == torch.float32 and target_dtype == np.float32:
        #             self.out[k] = cpu_tensor.numpy()
        #         else:
        #             self.out[k] = cpu_tensor.numpy().astype(target_dtype, copy=False)

        #         del cpu_tensor


        match CONFIG.batch.type:

            case 'token':

                ...

                # if convert_to_np_first:
                #     col_indices = np.arange(self.out['tokens'].shape[1])[np.newaxis, :]
                #     col_indices = np.broadcast_to(col_indices, self.out['tokens'].shape)

                # else:
                #     col_indices = torch.arange(self.out['tokens'].shape[1]).unsqueeze(0).expand_as(self.out['tokens']).to(self.out['tokens'].device)


                # # mask = (col_indices == 0) | (self.out['tokens'] != bos_token)
                # mask = (col_indices != 0) & (self.out['tokens'] != bos_token)

                # for k in self.out:
                #     self.out[k] = self.out[k][mask].cpu().numpy().astype(self.dtypes[k])

                # n_tokens = self.out['tokens'].shape[0]
                # n_seqs = n_tokens

            case 'sequence':

                mask = (self.batch_out['tokens'] != bos_token)

                if convert_to_np_first:
                    last_idx = np.maximum(np.argmax(mask[:, ::-1], axis=1), 0)

                else:
                    last_idx = mask.flip(1).int().argmax(dim=1)


                last_idx = self.batch_out['tokens'].shape[1] - 1 - last_idx

                all_padding = ~mask.any(1)
                assert (~all_padding).all().item()


                for k in self.batch_out:
                    # self.batch_out[k] = self.batch_out[k][:,:CONFIG.batch.max_seq_len]
                    self.batch_out[k] = self.batch_out[k].cpu().numpy().astype(self.dtypes[k])
                    self.batch_out[k] = [self.batch_out[k][i , :idx + 1] for i, idx in enumerate(last_idx)]

                n_tokens = sum(len(_) for _ in self.batch_out['tokens'])

        n_seqs = n_tokens / CONFIG.batch.max_seq_len
        n_batches = n_tokens / (CONFIG.batch.batch_size * CONFIG.batch.max_seq_len)


        # # convert tensors to cpu/numpy to be serialized for ray comms
        # for k in self.out:
        #     if isinstance(self.out[k], torch.Tensor):

        #         cpu_tensor = self.out[k].cpu()

        #         # TODO: replace this with specified types per key
        #         match k:
        #             case 'tokens':
        #                 target_dtype = np.int32
        #             case _:
        #                 target_dtype = np.float32

        #         if cpu_tensor.dtype == torch.float32 and target_dtype == np.float32:
        #             self.out[k] = cpu_tensor.numpy()
        #         else:
        #             self.out[k] = cpu_tensor.numpy().astype(target_dtype, copy=False)

        #         del cpu_tensor



        # print({k: (type(v), (len(v), '*', *v[0].shape[1:])) for k, v in self.out.items()}, flush=True)

        # raise NotImplementedError({k: (type(v), (len(v), '*', *v[0].shape[1:])) for k, v in self.out.items()})

        # for k, in self.out:
        #     assert self.out(v)

        batch_out = self.batch_out
        del self.batch_out
        self.batch_out = {}
        # torch.cuda.empty_cache()


        self.pbars['batch'].update(n_batches)
        self.pbars['seq'].update(n_seqs)
        self.pbars['token'].update(n_tokens)


        return batch_out


    def store_activation_hook(self, activation, hook, model_name, latent_name, layer_idx):

        # layer_idx, latent_name = (lambda _: [int(_[1]), _[2][5:]])(hook.name.split('.'))

        self.batch_out[f'{model_name}.{latent_name}'][...,layer_idx,:] = activation.detach()