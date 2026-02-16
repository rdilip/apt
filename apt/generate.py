"""Sampling helpers for GPT-with-encoder."""

from __future__ import annotations

from typing import Tuple, List

import torch


def _logits_to_probs(
    logits: torch.Tensor, *, method: str, threshold: float, temperature: float = 1.0
) -> torch.Tensor:
    logits = logits / temperature
    if method == "minp":
        probs = logits.softmax(dim=-1)
        p_scaled = probs.max(dim=-1).values * threshold
        prob_new = torch.where(
            probs > p_scaled.unsqueeze(-1), probs, torch.zeros_like(probs)
        )
        return prob_new / prob_new.sum(dim=-1, keepdim=True)
    if method in {"nucleus", "topp"}:
        probs = logits.softmax(dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = cumprobs > threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        return torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    if method == "argmax":
        probs = torch.zeros_like(logits)
        indices = logits.argmax(dim=-1)
        probs[torch.arange(logits.size(0)), indices] = 1
        return probs
    raise ValueError(f"Unknown sampling method: {method}")


def _sample_from_logits(
    logits: torch.Tensor, *, method: str, threshold: float, temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = _logits_to_probs(logits[:, -1, :], method=method, threshold=threshold, temperature=temperature)
    w = torch.empty_like(probs).exponential_(1)
    return torch.argmin(w / probs, dim=-1, keepdim=True).to(torch.long), probs


@torch.no_grad()
def generate_spda(
    model,
    *,
    batch_size: int,
    sampling_method: str = "minp",
    threshold: float = 0.15,
    temperature: float = 1.0,
    return_logits: bool = False,
) -> List[torch.Tensor] | Tuple[List[torch.Tensor], list[torch.Tensor]]:
    """Generate token sequences using SPDA attention."""
    device = next(model.parameters()).device
    if device.type == "cpu":
        raise ValueError("Generation on CPU is not recommended.")

    model.attn_backend = "spda"
    seq_BL = torch.full((batch_size, 1), fill_value=model.bos.item(), device=device, dtype=torch.long)
    eos = torch.full((batch_size, 1), fill_value=model.eos.item(), device=device)

    all_logits = []
    for _ in range(model.tokenizer.cfg.n_tokens):
        prot_ids_BL = torch.ones_like(seq_BL)
        logits, _ = model.forward_tokens(seq_BL, prot_ids_BL=prot_ids_BL, inference=True)
        new_tok, _ = _sample_from_logits(
            logits, method=sampling_method, threshold=threshold, temperature=temperature
        )
        seq_BL = torch.cat((seq_BL, new_tok), dim=-1)
        if return_logits:
            all_logits.append(logits.cpu().clone())

    seq_BL = torch.cat((seq_BL, eos), dim=-1)
    seqs = []
    for seq in seq_BL:
        cutoff = (seq == model.eos.item()).nonzero(as_tuple=True)[0][0].item() + 1
        seqs.append(seq[:cutoff])

    seqs = [seq[1:-1] for seq in seqs]
    if return_logits:
        return seqs, all_logits
    return seqs
