



import torch
import einops


from crosscoders.abc import AutoencoderABC
from crosscoders.configs import ModelConfig




class AcausalAutoencoder(AutoencoderABC, torch.nn.Module):

    def __init__(self, cfg: ModelConfig):

        super().__init__(cfg)


        # TODO: either init random (or other init options) or load pretrained weights
        self.W_enc = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty
            ((self.cfg.N_LAYERS, self.cfg.D_MODEL, self.cfg.D_CODER),
            **self.cfg.HARDWARE.asdict())))

        self.b_enc = torch.nn.Parameter(torch.zeros(
            (self.cfg.D_CODER,),
            **self.cfg.HARDWARE.asdict()))

        self.W_dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty
            ((self.cfg.D_CODER, self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict())))

        self.W_dec.data = einops.rearrange(
            self.W_enc.data.clone(),
            'n_layers d_model d_coder -> d_coder n_layers d_model'
        )

        # self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * 0.1

        self.b_dec = torch.nn.Parameter(torch.zeros(
            (self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict()))


    def _encode(self, x):

        x_enc = einops.einsum(
            x, self.W_enc,
            '... n_layers d_model , n_layers d_model d_coder -> ... d_coder'
        ) + self.b_enc

        x_enc = torch.nn.functional.relu(x_enc)
        self.x_enc = x_enc      # save memory costs using hooked model?

        return x_enc


    def _decode(self, x_enc):

        x_dec = einops.einsum(
            x_enc, self.W_dec,
            '... d_coder , d_coder n_layers d_model -> ... n_layers d_model'
        ) + self.b_dec

        return x_dec


    # def forward(self, x):

    #     x = super().forward(x)
    #     x = torch.nn.functional.relu(x)

    #     return x
