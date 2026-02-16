"""Diffusion autoencoder model"""

import torch
from torch import nn, Tensor
import math
from dataclasses import dataclass
import torch.nn.functional as F

from jaxtyping import Float
from typing import Optional

from .attention import SelfAttention, TransformerBlockSPDA
from .fsq import FSQ
from .cfm import ConditionalFlowMatcher

from math import prod

from timm.layers import Mlp
from torchdiffeq import odeint
from pathlib import Path
from scipy.spatial.transform import Rotation
import warnings

# jaxtyping hack to keep on old numpy version because proteina uses it for some unknown reason
Array = Tensor


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Array, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)




@dataclass
class DAEConfig:
    n_channels_decoder: int
    n_channels_encoder: int
    n_layers_encoder: int
    n_layers_decoder: int
    n_heads: int
    mlp_factor: int
    sigma: float = 0.0
    levels: tuple[int] = (8, 8, 8, 8)  # 4096
    share_adaLN: bool = True # by default False for backwards compatibility, but significantly improves flow matching results
    max_seq_len: int = 256
    dropout: float = 0.0
    n_tokens: int = 64
    n_size_toks: int = 1


class DAE(nn.Module):
    """DAE has encoder, decoder, VQ, and diffusion loss."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        levels = list(cfg.levels)
        self.idx_embed = nn.Embedding(256, cfg.n_channels_encoder)
        self.quantize = FSQ(
            levels=levels,
            dim_out=cfg.n_channels_encoder,
            dim=cfg.n_channels_encoder,
        )
        self.to_decoder = nn.Linear(cfg.n_channels_encoder, cfg.n_channels_decoder)
        self.codebook_size = prod(levels)

        # self.input_type = nn.Embedding(2, cfg.n_channels_encoder)
        self.cfm = ConditionalFlowMatcher(cfg.sigma, "uniform_beta")
        n_channels_in = 3# if not cfg.full_backbone else 3*3

        self.up = nn.Sequential(
            nn.Linear(n_channels_in, cfg.n_channels_encoder),
            nn.SiLU(),
            nn.Linear(cfg.n_channels_encoder, cfg.n_channels_encoder),
            nn.LayerNorm(cfg.n_channels_encoder),
        )

        self.encoder = nn.ModuleList([
            TransformerBlockSPDA(
                n_channels=cfg.n_channels_encoder,
                n_heads=cfg.n_heads,
                mlp_factor=cfg.mlp_factor,
                dropout=cfg.dropout,
                pos_embed=False,
            )
            for _ in range(cfg.n_layers_encoder)
        ])



        self.net = DiT(
            n_channels=cfg.n_channels_decoder,
            n_channels_pair=-1, # no pair bias
            channels_in=3,
            n_layers=cfg.n_layers_decoder,
            n_heads=cfg.n_heads,
            mlp_factor=cfg.mlp_factor,
            share_adaLN=cfg.share_adaLN,
            attn_backend='spda',
        )

        self.size_proj = nn.Sequential(
            nn.Linear(self.cfg.n_size_toks * cfg.n_channels_decoder, 4 * cfg.n_channels_decoder),
            nn.SiLU(),
            nn.Linear(4 * cfg.n_channels_decoder, self.cfg.max_seq_len),
        )


    @classmethod
    def from_pretrained(cls, ckpt_pth):
        assert isinstance(ckpt_pth, (str, Path)), "ckpt_pth must be a string or Path"
        map_location = "cpu" if not torch.cuda.is_available() else None
        ckpt = torch.load(ckpt_pth, map_location=map_location)
        cfg = DAEConfig(**ckpt["model_cfg"])
        # another backwards compatibility thing
        if 'n_size_toks' not in ckpt['model_cfg']:
            cfg.n_size_toks = 4
        use_strict = True
        model = cls(cfg)

        # this is obviously super brittle but it's literally just the one checkpoint, we can do this properly by inspecting state dict later
        
        if 'toasty-glade' in str(ckpt_pth):
            warnings.warn("Size proj not present. This should ONLY be the case for toasty-glade. Proceed with caution")
            use_strict=False
        

        model.load_state_dict(ckpt["model"], strict=use_strict)
        return model

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _preprocess(self, x_BLD):
        x_BLD = x_BLD - x_BLD.size(1)
        return x_BLD / 10.
    
  
    @torch._dynamo.disable
    def encode(self, x_BLD):
        # this is going to ASSUME that the thing is already mean centered
        B, L, D = x_BLD.shape
        s_BLD = self.up(x_BLD)
        s_BLD = s_BLD + self.idx_embed(torch.arange(L, device=x_BLD.device))

        # insert auxiliary tokens
        for enc_block in self.encoder:
            s_BLD = enc_block(s_BLD, attn_mask=None)

        c_BLD, idx_BL = self.quantize(s_BLD)
        c_BLD = self.to_decoder(c_BLD)

        return s_BLD, c_BLD, idx_BL

    @torch.no_grad()
    def decode_with_cfg_fn(self, idx_BL, cfg_fn=None, true_length=None, n_steps=100, noise_weight=0.1, score_weight=1.0):

        if cfg_fn is None:
            cfg_fn = lambda t: 1 - t
        c_BLD = self.quantize.indices_to_codes(idx_BL)
        c_BLD = self.to_decoder(c_BLD)

        if true_length is None:
            length_readout = self.size_proj(c_BLD[:, :4, :].view(c_BLD.size(0), -1)).argmax(-1)
            assert length_readout.allclose(length_readout[0])
            true_length = length_readout[0] + 1 # zero center

        c0_BLD = torch.zeros((c_BLD.size(0), true_length, c_BLD.size(-1)), device=c_BLD.device)

        # this shitty thing is necessary because sometimes true_length reads out too few tokens...
        # but it's usually p accurate so i don't thiink this should change too much
        cutoff = min(c_BLD.size(1), true_length)
        c0_BLD[:, :cutoff, :] = c_BLD[:, :cutoff]
        c_BLD = c0_BLD
        c0_BLD = torch.zeros_like(c_BLD)

        device = c_BLD.device
        B, L = c_BLD.shape[:2]
        dim_out = 3

        x_BLD = torch.randn(B, true_length, dim_out, device=device)
        x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)

        t = torch.linspace(0, 1, n_steps, device=device)
        gt = self.get_gt(t)
        dt = t[1] - t[0]

        for step in range(n_steps):
            t_B = torch.full((B,), t[step], device=device)
            v = self.net(x_BLD, t_B, z_BLD=c_BLD, attn_mask=None)
            v -= v.mean(dim=1, keepdim=True)
            # cfg info
            v0 = self.net(x_BLD, t_B, z_BLD=c0_BLD, attn_mask=None)
            v0 -= v0.mean(dim=1, keepdim=True)


            if t[step] >= 0.99:
                v = v0 + cfg_fn(t[step]) * (v - v0) 
                x_BLD = x_BLD + v * dt
            else:
                v = v0 + cfg_fn(t[step]) * (v - v0) 
                t_BL = t_B[:, None].expand(B, L)
                score = self.vf_to_score(x_BLD, v, t_BL)
                score0 = self.vf_to_score(x_BLD, v0, t_BL)

                score = score0 + cfg_fn(t[step]) * (score - score0)

                eps = torch.randn_like(x_BLD)
                eps -= eps.mean(dim=1, keepdim=True)
                std_eps = torch.sqrt(2 * gt[step] * noise_weight * dt)
                delta_x = (v + gt[step] * score * score_weight) * dt + std_eps * eps
                x_BLD = x_BLD + delta_x

        return x_BLD


    
    def decode(self, idx_BL, true_length=None, n_steps=100, noise_weight=0.45, score_weight=1.0):
        c_BLD = self.quantize.indices_to_codes(idx_BL)
        c_BLD = self.to_decoder(c_BLD)

        if true_length is None:
            true_length = c_BLD.size(1)

        c0_BLD = torch.zeros((c_BLD.size(0), true_length, c_BLD.size(-1)), device=c_BLD.device)
        c0_BLD[:, :c_BLD.size(1), :] = c_BLD
        c_BLD = c0_BLD

        device = c_BLD.device
        B, L = c_BLD.shape[:2]
        dim_out = 3

        x_BLD = torch.randn(B, true_length, dim_out, device=device)
        x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)

        t = torch.linspace(0, 1, n_steps, device=device)
        gt = self.get_gt(t)
        dt = t[1] - t[0]

        for step in range(n_steps):
            t_B = torch.full((B,), t[step], device=device)
            v = self.net(x_BLD, t_B, z_BLD=c_BLD, attn_mask=None)
            v -= v.mean(dim=1, keepdim=True)

            if t[step] >= 0.99:
                x_BLD = x_BLD + v * dt
            else:
                t_BL = t_B[:, None].expand(B, L)
                score = self.vf_to_score(x_BLD, v, t_BL)
                eps = torch.randn_like(x_BLD)
                eps -= eps.mean(dim=1, keepdim=True)
                std_eps = torch.sqrt(2 * gt[step] * noise_weight * dt)
                delta_x = (v + gt[step] * score * score_weight) * dt + std_eps * eps
                x_BLD = x_BLD + delta_x

        return x_BLD

    def forward(self, x_BLD, x0):
        # center data
        B, L = x_BLD.shape[:2]

        _, c_BLD, idx_BL = self.encode(x_BLD)
        D = c_BLD.size(-1)
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, x_BLD)

        # now, here is the SAUCE
        # we need to mask out some fraction of tokens. our hope is this learns a hiearchical representation
        # without explicit downsampling

        # Random cutoff per batch (number of 1s)
        # Classifier-free guidance is the special case where we mask out all tokens
        # for simplicity, we'll let this happen with standard probability 1 / L
        # use number of tokens
        # keep a minimum of 16 tokens
        cutoffs = torch.randint(self.cfg.n_size_toks, min(L, self.cfg.n_tokens), (B,), device=x_BLD.device)
        idx = torch.arange(L, device=x_BLD.device).unsqueeze(0)
        cmask = (idx < cutoffs.unsqueeze(1)).to(torch.float32)
        c_BLD = c_BLD * cmask.unsqueeze(-1)

        # classifier free guidance in addition to the above mask
        cfg_mask = (torch.rand((B,), device=x_BLD.device) > 0.05)[
            :, None, None
        ]
        c_BLD = c_BLD * cfg_mask

        # notice right now, we do readout using the full sequence length _after_ masking. 
        # you could do it before, because you autoregressively generate the full sequence,
        # but I tend to think of the sequence length as a "high level" property

        hL_BLD = c_BLD.clone()

        # for block in self.to_size:
        #     hL_BLD = block(hL_BLD, attn_mask=None)

        # hL_BLV = self.proj_to_size(hL_BLD[:, [0], :]) # put the sequence length on the first vector

        vt = self.net(xt, t, z_BLD=c_BLD, attn_mask=None)
        loss = ((ut[:, :L, :] - vt[:, :L, :]) ** 2).mean()

        # compute size loss
        inp_BLD = c_BLD[:, :self.cfg.n_size_toks].reshape(B, -1)
        inp_BS = self.size_proj(inp_BLD)
        size_B = torch.full((B,), fill_value=L - 1, device=x_BLD.device, dtype=torch.long)
        size_loss = F.cross_entropy(inp_BS, size_B)
        # size_tgt = torch.full((B,), fill_value=L - 1, device=x_BLD.device, dtype=torch.long)
        # size_loss = F.cross_entropy(hL_BLV.reshape(-1, self.cfg.max_seq_len), size_tgt)


        loss_dict = {"flow_loss": loss, "size_loss": size_loss}

        return idx_BL, loss_dict

    def get_gt(
        self,
        t: Float[Tensor, "s"],
        mode: str = "us",
        param: float = 1.0,
        clamp_val: Optional[float] = None,
        eps: float = 1e-2,
    ) -> Float[Tensor, "s"]:
        """
        Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is

        Returns
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized**f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - 1e-5)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]

    def vf_to_score(
        self,
        x_t: Float[Tensor, "* n 3"],
        v: Float[Tensor, "* n 3"],
        t: Float[Tensor, "* n"],
        scale_ref: float = 1.0,
    ):
        """
        Compute score of noisy density given the vector field learned by flow matching. With
        our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*]

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        num = t[..., None] * v - x_t  # [*, n, 3]
        den = (1.0 - t)[..., None] * scale_ref**2  # [*, n, 1]
        score = num / den
        return score  # [*, dim]
    



@dataclass
class RnFlowMatcherConfig:
    n_channels: int = 256
    n_channels_pair: int = 64
    channels_in: int = 3
    n_layers: int = 16
    n_heads: int = 8
    mlp_factor: int = 4
    sigma: float = 0.0
    use_qknorm: bool = False
    share_adaLN: bool = False


class RnFlowMatcher(nn.Module):
    def __init__(self, cfg: RnFlowMatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.net = DiT(
            n_channels=cfg.n_channels,
            channels_in=cfg.channels_in,
            n_channels_pair=cfg.n_channels_pair,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_factor=cfg.mlp_factor,
            use_qknorm=cfg.use_qknorm,
            share_adaLN=cfg.share_adaLN,
        )
        self.cfm = ConditionalFlowMatcher(cfg.sigma, "uniform_beta")

    def forward(self, x_BLD, z_BLD=None):
        """
        x1 is clean data
        """
        x0 = torch.randn_like(x_BLD)
        x0 = x0 - x0.mean(dim=1, keepdim=True)
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, x_BLD)
        vt = self.net(xt, t)
        loss = (ut - vt) ** 2
        loss_dict = {"flow_loss": loss.mean()}
        return loss_dict

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def sample(self, x_BLD, n_steps):
        device = x_BLD.device

        def _ode_func(t, x):
            x = x - x.mean(dim=1, keepdim=True)
            t = torch.ones(x.size(0)).to(x.device) * t
            return self.net(x, t)

        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps, device=device)
        samples = odeint(_ode_func, x, t)
        return samples

    @torch.no_grad()
    def euler_sample(self, x_BLD, n_steps):
        device = x_BLD.device
        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps, device=device)
        dt = t[1] - t[0]
        for t_ in t[:-1]:
            x = x + self.net(x, torch.tensor([t_]).to(x.device)) * dt
            x -= x.mean(dim=1, keepdim=True)
        return x

    @torch.no_grad()
    def euler_maruyama_sample(
        self, x_BLD, n_steps, score_weight=1.0, noise_weight=0.45
    ):
        device = x_BLD.device
        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]
        gt = self.get_gt(t)
        dt = t[1] - t[0]
        for step in range(n_steps):
            t_ = t[step]
            v = self.net(x, torch.tensor([t_]).to(x.device))
            v -= v.mean(dim=1, keepdim=True)

            if t_ >= 0.99:
                x = x + v * dt
            else:
                score = self.vf_to_score(
                    x, v, t_ * torch.ones(x.shape[:-1], device=x.device)
                )  # get score from v, [*, dim]
                eps = torch.randn_like(x)
                eps -= eps.mean(dim=1, keepdim=True)
                std_eps = torch.sqrt(2 * gt[step] * noise_weight * dt)
                delta_x = (v + gt[step] * score * score_weight) * dt + std_eps * eps
                x = x + delta_x

            # x -= x.mean(dim=1, keepdim=True)

        return x

    def vf_to_score(
        self,
        x_t: Float[Tensor, "* n 3"],
        v: Float[Tensor, "* n 3"],
        t: Float[Tensor, "* n"],
        scale_ref: float = 1.0,
    ):
        """
        Compute score of noisy density given the vector field learned by flow matching. With
        our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*]

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        num = t[..., None] * v - x_t  # [*, n, 3]
        den = (1.0 - t)[..., None] * scale_ref**2  # [*, n, 1]
        score = num / den
        return score  # [*, dim]

    def get_gt(
        self,
        t: Float[Tensor, "s"],
        mode: str = "us",
        param: float = 1.0,
        clamp_val: Optional[float] = None,
        eps: float = 1e-2,
    ) -> Float[Tensor, "s"]:
        """
        Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is

        Returns
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized**f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - 1e-5)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]


##################################################################################
#                             Components for RnFlowMatcher                       #
##################################################################################


class PairEmbedderNoCond(nn.Module):
    def __init__(self, n_channels, n_buckets, full_backbone=False):
        """
        Pair embedding -- this assumes that ALL elements in a batch are the same, just rotated. Thus, the inter pair
        distances won't change. This will save a ton of memory and we've proven we can train this way anyway.
        """
        super().__init__()
        self.embed = nn.Embedding(n_buckets, n_channels)
        self.pos_embed = nn.Embedding(128, n_channels)
        # 30 Angstoms = 3 nm, inputs are in nm
        # we compute bins on the square, avoids a sqrt which can get nasty with gradients
        self.register_buffer("bins", torch.linspace(0, 3**2, n_buckets - 1))
        self.mlp = Mlp(
            n_channels, 4 * n_channels, n_channels, act_layer=nn.GELU, norm_layer=None
        )
        self.ln = nn.LayerNorm(n_channels)
        self.nc = n_channels
        self.full_backbone = full_backbone

    def forward(self, x_BLD):
        B, L, D = x_BLD.shape
        # just use the first element, assumes that these are repeated
        if self.training:
            if self.full_backbone:
                x_BLD = x_BLD.view(B, L, -1, 3)[:, :, 1, :] # pull out atom dimension
            d_BLL = ((x_BLD[0, :, None] - x_BLD[0, None, :]) ** 2).sum(dim=-1)
            tmp = ((x_BLD[1, :, None] - x_BLD[1, None, :]) ** 2).sum(dim=-1)
            assert (d_BLL - tmp).abs().max() < 1e-2
            d_BLL = d_BLL.unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
            d_BLL = ((x_BLD[:, :, None] - x_BLD[:, None, :]) ** 2).sum(dim=-1)

        idx_L = torch.arange(L, device=x_BLD.device)
        idx_LL = (idx_L[:, None] - idx_L[None, :]).clip(min=-64, max=63) + 64
        pos_LLD = self.pos_embed(idx_LL)

        d_BLL = torch.bucketize(d_BLL, self.bins)
        s_BLLD = self.embed(d_BLL) + pos_LLD
        s_BLLD = self.mlp(self.ln(s_BLLD))

        return s_BLLD


class PairEmbedder(nn.Module):
    def __init__(self, n_channels, n_buckets):
        """
        Pair embedding -- this assumes that ALL elements in a batch are the same, just rotated. Thus, the inter pair
        distances won't change. This will save a ton of memory and we've proven we can train this way anyway.
        """
        super().__init__()
        self.embed = nn.Embedding(n_buckets, n_channels)
        self.pos_embed = nn.Embedding(128, n_channels)
        # 30 Angstoms = 5 nm, inputs are in nm
        self.register_buffer("bins", torch.linspace(0, 3, n_buckets - 1))
        self.mlp = Mlp(
            n_channels, 4 * n_channels, n_channels, act_layer=nn.GELU, norm_layer=None
        )
        self.ln = nn.LayerNorm(n_channels)
        self.nc = n_channels

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_channels, n_channels * 3, bias=True),
        )

    @torch.compile()
    def forward(self, x_BLD, c_BD):
        B, L, D = x_BLD.shape
        # just use the first element, assumes that these are repeated
        if self.training:
            d_BLL = (x_BLD[0, :, None] - x_BLD[0, None, :]) ** 2
            tmp = d_BLL = (x_BLD[1, :, None] - x_BLD[1, None, :]) ** 2
            assert d_BLL.allclose(tmp)
            d_BLL = d_BLL.unsqueeze(0).expand(B, -1, -1)
        else:
            d_BLL = ((x_BLD[:, :, None] - x_BLD[:, None, :]) ** 2).sum(dim=-1)

        idx_L = torch.arange(L, device=x_BLD.device)
        idx_LL = (idx_L[:, None] - idx_L[None, :]).clip(min=-64, max=63) + 64
        pos_LLD = self.pos_embed(idx_LL)

        shift_time, scale_time, gate_time = self.adaLN_modulation(c_BD).chunk(3, dim=-1)
        shift_time = shift_time.view(B, 1, 1, self.nc)
        scale_time = scale_time.view(B, 1, 1, self.nc)
        gate_time = gate_time.view(B, 1, 1, self.nc)

        d_BLL = torch.bucketize(d_BLL, self.bins)
        s_BLLD = self.embed(d_BLL) + pos_LLD
        s_BLLD = s_BLLD * (1 + scale_time) + shift_time

        s_BLLD = s_BLLD + gate_time * self.mlp(self.ln(s_BLLD))

        return s_BLLD


class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_channels_pair,
        n_heads,
        mlp_factor,
        dropout=0.1,
        shared_adaLN=None,
        attn_backend='spda',
    ):
        super().__init__()
        self.attn = SelfAttention(
            n_channels,
            n_heads,
            attn_backend=attn_backend,
            dropout=dropout,
        )
        self.mlp = Mlp(
            n_channels,
            n_channels * mlp_factor,
            n_channels,
            act_layer=nn.GELU,
            norm_layer=None,
            drop=dropout,
        )
        self.norm1 = nn.LayerNorm(n_channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(n_channels, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(n_channels, elementwise_affine=False)

        if shared_adaLN is not None:
            self.adaLN_modulation = shared_adaLN
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_channels, n_channels * 6, bias=True),
            )

    def forward(self, s_BLD, c_BD, **attn_params):
        adaLN_output = self.adaLN_modulation(c_BD)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaLN_output.chunk(6, dim=-1)
        )
        s_BLD = s_BLD + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(s_BLD), shift_msa, scale_msa), **attn_params
        )

        s_BLD = s_BLD + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(s_BLD), shift_mlp, scale_mlp)
        )
        return s_BLD


class DiT(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_channels_pair,
        channels_in,
        n_layers,
        n_heads,
        mlp_factor=4,
        share_adaLN=True,
        attn_backend='spda',
    ):
        super().__init__()

        assert attn_backend in ['flex', 'spda', 'none']

        self.up = InitialLinear(channels_in, n_channels)
        self.share_adaLN = share_adaLN

        if share_adaLN:
            self.shared_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_channels, n_channels * 6, bias=True),
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    n_channels=n_channels,
                    n_channels_pair=n_channels_pair,
                    n_heads=n_heads,
                    mlp_factor=mlp_factor,
                    attn_backend=attn_backend,
                    shared_adaLN=self.shared_adaLN_modulation if share_adaLN else None,
                )
                for _ in range(n_layers)
            ]
        )
        self.time_embedder = TimestepEmbedder(n_channels)

        # I don't like modules that conditionally change the state dict
        self.cond_embed = nn.Embedding(2, n_channels)

        self.final = FinalLinear(n_channels, channels_in)

    @classmethod
    def sequence_packed(cls, doc_ids):
        def _sequence_packed(b, h, q_idx, kv_idx):
            return doc_ids[q_idx] == doc_ids[kv_idx]
        return _sequence_packed

    def forward(self, x_BLD, t_B, z_BLD, **attn_params):
        """
        z_BLD is concatenated to the input as conditioning vector.
        """
        c_BD = self.time_embedder(t_B)
        s_BLD = self.up(x_BLD, c_BD)

        # c_BD = t_BD
        # in context conditioning
        assert z_BLD is not None, (
            "z_BLD must be provided if has_conditioning is True"
        )
        shape_inp = s_BLD.shape[:-1]
        shape, dev = z_BLD.shape[:-1], z_BLD.device
        cond_BL = torch.cat(
            (
                torch.zeros(shape_inp, dtype=torch.long, device=dev),
                torch.ones(shape, dtype=torch.long, device=dev),
            ),
            dim=-1,
        )
        s_BLD = torch.cat([s_BLD, z_BLD], dim=-2)
        s_BLD = s_BLD + self.cond_embed(cond_BL)

        for block in self.blocks:
            s_BLD = block(s_BLD, c_BD, **attn_params)

        L = x_BLD.size(1)
        s_BLD = s_BLD[:, :L, :]
        xout_BLD = self.final(s_BLD, c_BD)
        return xout_BLD


class InitialLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.norm_initial = nn.LayerNorm(
            out_channels, elementwise_affine=False, eps=1e-6
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(out_channels, 2 * out_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(x)
        x = modulate(self.norm_initial(x), shift, scale)
        return x


class FinalLinear(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )

        out_layer = nn.Linear(model_channels, out_channels, bias=True)

        self.linear = out_layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


################################################################################
#                                Augmentations                                 #
################################################################################
