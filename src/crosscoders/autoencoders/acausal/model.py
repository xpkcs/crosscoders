



import torch
import einops


from crosscoders.abc import AutoencoderABC
from crosscoders.dataclasses.configs.runner import ModelConfig




class JumpReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, t, eps):
        threshold = torch.exp(t)
        # dx_mask = input > threshold
        # dt_mask = ((input - threshold) / eps).abs() < 0.5
        # ctx.save_for_backward(dx_mask, dt_mask, t, torch.tensor(eps, device=input.get_device()))     # save for backward computation

        ctx.save_for_backward(input, t, eps)     # save for backward computation
        output = torch.where(input > threshold, input, 0)       # if input > exp(t), pass through

        return output


    @staticmethod
    def backward(ctx, grad_output):
        input, t, eps = ctx.saved_tensors

        threshold = torch.exp(t)
        dx_mask = input > threshold
        dt_mask = ((input - threshold) / eps).abs() < 0.5

        # derivative wrt input: 1 if input > threshold, 0 o/w
        grad_input = grad_output * dx_mask.type_as(grad_output)

        # derivative wrt t: exp(t) if input <= threshold, 0 o/w
        grad_t = grad_output * (dt_mask.type_as(grad_output) * (-threshold / eps))

        return grad_input, grad_t, None


class AcausalAutoencoder(AutoencoderABC):

    def __init__(self, cfg: ModelConfig):

        super().__init__(cfg)


        # TODO: either init random (or other init options) or load pretrained weights

        # W_dec
        self.W_dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty
            ((self.cfg.D_CODER, self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict())))

        # self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * 0.1

        self.b_dec = torch.nn.Parameter(torch.zeros(
            (self.cfg.N_LAYERS, self.cfg.D_MODEL),
            **self.cfg.HARDWARE.asdict()))

        # W_enc
        self.W_enc = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty
            ((self.cfg.D_MODEL, self.cfg.N_LAYERS, self.cfg.D_CODER),
            **self.cfg.HARDWARE.asdict())))

        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            'd_coder n_layers d_model -> d_model n_layers d_coder'
        ).contiguous()

        self.b_enc = torch.nn.Parameter(torch.zeros(
            (self.cfg.D_CODER,),
            **self.cfg.HARDWARE.asdict()))


        match self.cfg.ACTIVATION_FUNCTION:
            case 'relu':
                self.activation_function = torch.nn.functional.relu
                self.args = ()

            case 'jumprelu':
                self.activation_function = JumpReLU.apply

                # t
                self.t = torch.nn.Parameter(0.1 * torch.ones(
                    (self.cfg.D_CODER,),
                    **self.cfg.HARDWARE.asdict()))

                self.args = (
                    self.t,
                    torch.tensor(self.cfg.eps, requires_grad=False, **self.cfg.HARDWARE.asdict())
                )


    def _encode(self, x):

        x_enc = einops.einsum(
            x, self.W_enc,
            '... n_layers d_model , n_layers d_model d_coder -> ... d_coder'
        ) + self.b_enc


        x_enc = self.activation_function(x_enc, *self.args)
        self.x_enc = x_enc      # save memory costs using hooked model?

        return x_enc


    def _decode(self, x_enc):

        x_dec = einops.einsum(
            x_enc, self.W_dec,
            '... d_coder , d_coder n_layers d_model -> ... n_layers d_model'
        ) + self.b_dec

        return x_dec
