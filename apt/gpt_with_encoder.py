"""
GPT model with attached tokenizer.

This file intentionally stays separate from models.py to allow loading
tokenizers from different checkpoints without cross-branch coupling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from dataclasses import dataclass
from pathlib import Path
import time
from collections import OrderedDict



from .attention import UnifiedTransformerBlock
from torch.nn.attention.flex_attention import create_block_mask, and_masks, BlockMask
from typing import Tuple
from .dae import DAE
from math import prod
import inspect

# warmup block mask cache

##################################################################
#                          GPT class                              #
##################################################################
@dataclass
class GPTWithEncoderConfig:
    n_channels: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    mlp_factor: int = 4
    pth_to_tokenizer: str = None
    block_size: int = 256
    pth_to_tokenizer = 'denim-fog-265/best_model.pt'

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

@functools.lru_cache(maxsize=None)
def get_block_mask(L: int, device: str):
    # Compile once per (B,H,L); super fast to reuse thereafter
    dev = torch.device(device)
    return create_block_mask(
        causal, None, None, L, L, device=dev
    )

def get_block_mask_with_protids(protids_BL: torch.Tensor, device: str):
    dev = torch.device(device)
    def protid(b, h, q_idx, kv_idx):
        return protids_BL[b, q_idx] == protids_BL[b, kv_idx]
    
    causal_and_protid = and_masks(causal, protid)

    # compile false for inference, causes memory error, not sure why

    return create_block_mask(
        causal_and_protid, None, None, protids_BL.shape[1], protids_BL.shape[1], device=dev, _compile=False
    )

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def make_attention_params(prot_ids_BL, attn_backend: str):
    if attn_backend == 'flex':
        return {
            'block_mask': get_block_mask_with_protids(prot_ids_BL, device=str(prot_ids_BL.device))
        }
    elif attn_backend == 'spda':
        idx = torch.arange(prot_ids_BL.size(-1), device=prot_ids_BL.device).unsqueeze(0).expand(prot_ids_BL.size(0), -1)
        attn_mask = idx.unsqueeze(-1) >= idx.unsqueeze(-2)
        attn_mask = attn_mask & (prot_ids_BL.unsqueeze(-1) == prot_ids_BL.unsqueeze(-2))
        return {'attn_mask': attn_mask.unsqueeze(1)}


class GPTWithEncoder(nn.Module):
    def __init__(self, cfg: GPTWithEncoderConfig, attn_backend: str = 'spda'):
        super().__init__()
        self.cfg = cfg
        self.attn_backend = attn_backend
        self.tokenizer = DAE.from_pretrained(cfg.pth_to_tokenizer).eval()

        # unclear if these are doing anything
        # self.tokenizer.up = torch.compile(self.tokenizer.up)
        # self.tokenizer.quantize = torch.compile(self.tokenizer.quantize)
        # self.tokenizer.to_decoder = torch.compile(self.tokenizer.to_decoder)

        self.n_channels_in = 3
        vocab_size = prod(self.tokenizer.cfg.levels) + 2

        # pad vocab size up to nearest multiple of 64
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.vocab_size = vocab_size

        # self.pos_embed = nn.Embedding(260, cfg.n_channels)

        for param in self.tokenizer.parameters():
            param.requires_grad = False

        # +2 for bos and eos
        self.embed = nn.Embedding(self.vocab_size, cfg.n_channels)
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    n_channels=cfg.n_channels,
                    n_heads=cfg.n_heads,
                    mlp_factor=cfg.mlp_factor,
                    dropout=cfg.dropout,
                    attn_backend=attn_backend,
                )
                for _ in range(cfg.n_layers)
            ]
        )


        self.proj = nn.Linear(cfg.n_channels, self.vocab_size, bias=False)
        self.embed.weight = self.proj.weight

        self.ln = nn.LayerNorm(cfg.n_channels)

        bos = torch.tensor([self.vocab_size - 2], dtype=torch.long, device=self.embed.weight.device).unsqueeze(0)
        eos = torch.tensor([self.vocab_size - 1], dtype=torch.long, device=self.embed.weight.device).unsqueeze(0)
        self.register_buffer("bos", bos)
        self.register_buffer("eos", eos)

    @staticmethod
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, ckpt_pth):
        # some syntax sugar to pull this from remote
        assert isinstance(ckpt_pth, (str, Path)), "ckpt_pth must be a string or Path"
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        cfg = GPTWithEncoderConfig(**ckpt["model_cfg"])
        # some monkey patching real quick, since we set this manually earlier
        model = cls(cfg)
        state_dict = ckpt["ema_model"]
        # ugh, hack to fix the fact I did torch.compile instead of model.compile :eyeroll:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k_ = k
            if ('_orig_mod' in k) and ('tokenizer' in k):
                k_ = k.replace('_orig_mod.', '')

            new_state_dict[k_] = v

        model.load_state_dict(new_state_dict)
        return model
    
    def custom_compile(self):
        """
        custom compile accommodates the fact we like to take inputs as sequences of ragged tensors,
        which are concatenated and stacked.
        """
        for block in self.blocks:
            block.compile()
        self.embed.compile()
        self.proj.compile()
    
    def forward_tokens(self, inp_BL, prot_ids_BL, inference=False, **attn_params):
        """ forward just tokens to separate out encoding from training loss"""
        s_BLD = self.embed(inp_BL)
        L = inp_BL.size(-1)
        device = s_BLD.device
        L = s_BLD.size(-2)

        attn_params = make_attention_params(prot_ids_BL, self.attn_backend)
        for block in self.blocks:
            s_BLD = block(s_BLD, **attn_params)
        s_BLD = self.ln(s_BLD)

        if inference:
            logits_BLV = self.proj(s_BLD[:, [-1], :])
        else:
            logits_BLV = self.proj(s_BLD)
        # note: we can micro optimize this by just passing the last token, but
        # we are probably bottlenecked by decoding anyway and this simplifies the code a bit
        return logits_BLV, s_BLD
    
    
    def compute_loss(self, logits_BLV, tgt_BL):
        loss = F.cross_entropy(
            logits_BLV.view(-1, self.vocab_size),
            tgt_BL.reshape(-1),
            reduction="none",
        ).mean()
        acc = (logits_BLV.argmax(dim=-1) == tgt_BL).float().mean()
        return loss, acc

    @torch._dynamo.disable 
    def batch_sequence(self, x_BLD: list[torch.Tensor]):
        tok_lst_BL = []
        lens = []
        with torch.inference_mode():
            for m in range(len(x_BLD)):
                *_, tok_BL = self.tokenizer.encode(x_BLD[m].view(x_BLD[m].size(0), -1, self.n_channels_in))
                tok_BL = tok_BL[:, :min(self.tokenizer.cfg.n_tokens, x_BLD[m].size(1))] # only the first n_tokens contribute
                tok_BL = torch.cat(
                    (self.bos.expand(tok_BL.size(0), -1), tok_BL, self.eos.expand(tok_BL.size(0), -1)), dim=-1
                ).long()
                tok_lst_BL.append(tok_BL)
                lens.append(tok_BL.size(-1))
        lens = torch.tensor(lens, device=self.embed.weight.device)
        prot_ids_BL = torch.arange(1, lens.size(0) + 1, device=self.embed.weight.device).repeat_interleave(lens)
        prot_ids_BL = prot_ids_BL.unsqueeze(0).expand(x_BLD[0].shape[0], -1)

        # make prot ids
        tok_BL = torch.cat(tok_lst_BL, dim=-1)

        # cut down to block size
        tok_BL = tok_BL[:, :self.cfg.block_size]
        prot_ids_BL = prot_ids_BL[:, :self.cfg.block_size]
        return tok_BL, prot_ids_BL

    def forward(self, x_BLD, compute_loss=False):
        tok_BL, prot_ids_BL = self.batch_sequence(x_BLD)
        prot_ids_BL = prot_ids_BL[:, :-1].contiguous()
        inp_BL = tok_BL[:, :-1].contiguous()
        tgt_BL = tok_BL[:, 1:].contiguous()

        logits_BLV, s_BLD = self.forward_tokens(inp_BL, prot_ids_BL)

        loss, acc = None, None
        if compute_loss:
            loss, acc = self.compute_loss(logits_BLV, tgt_BL)

        return logits_BLV, tok_BL, loss, acc

    @torch.no_grad()
    def minp_filter(self, logits, p_base=0.1):
        probs = logits.softmax(dim=-1)
        max_token = logits.argmax(dim=-1)
        p_scaled = probs.max(dim=-1).values * p_base
        prob_new = torch.where(
            probs > p_scaled.unsqueeze(-1), probs, torch.zeros_like(probs)
        )
        prob_new = prob_new / prob_new.sum(dim=-1, keepdim=True)

        return prob_new

    @torch.no_grad()
    def nucleus_filter(self, logits, p=0.9):
        """Set logits of tokens outside top-p to -inf."""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        # Set filtered logits to -inf
        sorted_logits = logits.gather(-1, sorted_indices)
        sorted_logits[mask] = float("-inf")

        # Restore original order
        filtered_logits = torch.empty_like(logits).scatter(
            -1, sorted_indices, sorted_logits
        )
        return filtered_logits
    
    @torch.no_grad()
    def forward_sampled_tokens(self, toks_BL):
        # mostly for debugging
        tok_w_bos_eos_BL = torch.cat((self.bos.expand(toks_BL.size(0), -1), toks_BL, self.eos.expand(toks_BL.size(0), -1)), dim=-1)

        inp_BL = tok_w_bos_eos_BL[:, :-1].contiguous()
        protids_BL = torch.ones_like(inp_BL)
        block_mask = get_block_mask_with_protids(protids_BL, device=str(toks_BL.device))
        logits, _ = self.forward_tokens(inp_BL, block_mask, inference=False)
        loss = self.compute_loss(logits, tok_w_bos_eos_BL[:, 1:])
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        *,
        max_output_size: int = 256,
        temperature: float = 1.0,
        sampling_method: str = "minp",
        threshold: float = 0.15,
        decode: bool = True,
        noise_weight: float = 0.45,
        score_weight: float = 1.0,
        cfg_weight: float = 2.0,
    ):
        device = self.embed.weight.device
        idx = self.bos.clone()
        start_time = time.time()

        block_mask = create_block_mask(causal, 1, 1, max_output_size, max_output_size, device=device)

        # this might need to be a 0...
        input_pos = torch.tensor([1], device=device).unsqueeze(-1)

        while idx.size(-1) < max_output_size:
            logits = self.decode_one_token(idx, input_pos, block_mask, max_seq_length=max_output_size)
            logits = logits / temperature

            if sampling_method == "minp":
                assert threshold < 0.8
                probs = self.minp_filter(logits, p_base=threshold)
            elif sampling_method == "nucleus":
                filtered_logits = self.nucleus_filter(logits, p=threshold)
                probs = F.softmax(filtered_logits, dim=-1)
            else:
                raise ValueError(f"Invalid sampling method: {sampling_method}")

            idx_next = torch.multinomial(probs[:, 0, :], num_samples=1)
            if idx_next.item() == self.eos.item():
                break
            idx = torch.cat((idx, idx_next), dim=-1)
            if idx.size(-1) >= max_output_size:
                break
            
            input_pos += 1
        
        idx = idx[:, 1:]
        recon = None
        if decode:
            recon = self.tokenizer.decode(idx, noise_weight=noise_weight, score_weight=score_weight, cfg_weight=cfg_weight)
        return idx, recon


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {fused_available}")

        return optimizer
