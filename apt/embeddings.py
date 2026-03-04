from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class _TokenizerLike(Protocol):
    cfg: object
    quantize: object

    def encode(self, x_BLD: Tensor):
        ...


def _validate_point_cloud(x_L3: Tensor) -> None:
    if x_L3.ndim != 2 or x_L3.shape[-1] != 3:
        raise ValueError(f"Expected shape (L, 3), got {tuple(x_L3.shape)}")


def canonicalize_point_cloud(x_L3: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Move a point cloud to a deterministic canonical orientation.

    Steps:
    1) zero-center,
    2) align to principal axes (PCA),
    3) fix axis signs from the farthest point,
    4) enforce right-handed orientation.
    """
    _validate_point_cloud(x_L3)
    orig_dtype = x_L3.dtype
    x = x_L3.to(dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)

    # Degenerate case (single point or no spread): only centering is meaningful.
    if x.shape[0] < 2 or x.square().sum() <= eps:
        return x.to(dtype=orig_dtype)

    cov = x.transpose(0, 1) @ x
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    basis = evecs[:, order]

    # Deterministic sign disambiguation from the farthest point.
    anchor = x[x.square().sum(dim=-1).argmax()]
    signs = torch.sign(anchor @ basis)
    signs = torch.where(signs.abs() < eps, torch.ones_like(signs), signs)
    basis = basis * signs.unsqueeze(0)

    # Keep a proper rotation (det +1).
    if torch.det(basis) < 0:
        basis[:, -1] = -basis[:, -1]

    return (x @ basis).to(dtype=orig_dtype)


def canonicalize_batch(x_BLD: Tensor, eps: float = 1e-8) -> Tensor:
    if x_BLD.ndim != 3 or x_BLD.shape[-1] != 3:
        raise ValueError(f"Expected shape (B, L, 3), got {tuple(x_BLD.shape)}")
    return torch.stack([canonicalize_point_cloud(x_L3, eps=eps) for x_L3 in x_BLD], dim=0)


def fsq_table_from_indices(quantizer, idx_BL: Tensor) -> Tensor:
    """
    Convert token ids to FSQ table values per token.
    Returns shape (B, T, n_levels).
    """
    if idx_BL.ndim == 1:
        idx_BL = idx_BL.unsqueeze(0)
    if idx_BL.ndim != 2:
        raise ValueError(f"Expected shape (B, T) or (T,), got {tuple(idx_BL.shape)}")

    device = quantizer._basis.device
    idx_BL = idx_BL.to(device=device, dtype=torch.long)
    return quantizer.indices_to_codes(idx_BL, project_out=False)


def fsq_embedding_from_table(fsq_BTL: Tensor, ntoks: int = 32, pad_value: float = 0.0) -> Tensor:
    """
    Build fixed-size embeddings from the first ntoks FSQ rows and flatten.
    Output shape: (B, ntoks * n_levels).
    """
    if ntoks <= 0:
        raise ValueError(f"ntoks must be > 0, got {ntoks}")
    if fsq_BTL.ndim != 3:
        raise ValueError(f"Expected shape (B, T, n_levels), got {tuple(fsq_BTL.shape)}")

    bsz, seq_len, n_levels = fsq_BTL.shape
    if seq_len >= ntoks:
        table = fsq_BTL[:, :ntoks, :]
    else:
        pad = torch.full(
            (bsz, ntoks - seq_len, n_levels),
            fill_value=pad_value,
            dtype=fsq_BTL.dtype,
            device=fsq_BTL.device,
        )
        table = torch.cat([fsq_BTL, pad], dim=1)

    return table.reshape(bsz, ntoks * n_levels)


def fsq_embedding_from_indices(quantizer, idx_BL: Tensor, ntoks: int = 32, pad_value: float = 0.0) -> Tensor:
    fsq_BTL = fsq_table_from_indices(quantizer, idx_BL)
    return fsq_embedding_from_table(fsq_BTL, ntoks=ntoks, pad_value=pad_value)


@torch.no_grad()
def embed_point_clouds(
    tokenizer: _TokenizerLike,
    x_BLD: Tensor,
    *,
    ntoks: int = 32,
    canonicalize: bool = True,
    pad_value: float = 0.0,
    return_indices: bool = False,
):
    """
    Embed point clouds by:
    1) optional canonical rotation,
    2) tokenization,
    3) first-ntoks FSQ table flattening.
    """
    if x_BLD.ndim == 2:
        x_BLD = x_BLD.unsqueeze(0)
    if x_BLD.ndim != 3 or x_BLD.shape[-1] != 3:
        raise ValueError(f"Expected shape (B, L, 3) or (L, 3), got {tuple(x_BLD.shape)}")

    n_tokens_cap = getattr(tokenizer.cfg, "n_tokens", x_BLD.shape[1])
    x_in = canonicalize_batch(x_BLD) if canonicalize else x_BLD
    x_in = x_in.to(device=tokenizer.quantize._basis.device, dtype=torch.float32)
    *_, idx_BL = tokenizer.encode(x_in)
    idx_BL = idx_BL[:, : min(idx_BL.shape[1], n_tokens_cap)]

    emb_BD = fsq_embedding_from_indices(tokenizer.quantize, idx_BL, ntoks=ntoks, pad_value=pad_value)
    if return_indices:
        return emb_BD, idx_BL
    return emb_BD
