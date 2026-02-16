from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from .dae import DAE
from .gpt_with_encoder import GPTWithEncoder, GPTWithEncoderConfig

HF_REPO_ID = "rdilip/apt"
TOKENIZER_FILENAME = "tokenizer128.pt"
LANGUAGE_MODEL_FILENAME = "lm128.pt"


def _checkpoint_root() -> Path:
    return resources.files("apt") / "checkpoints"


def default_tokenizer_checkpoint() -> Path:
    return _checkpoint_root() / TOKENIZER_FILENAME


def default_language_model_checkpoint() -> Path:
    return _checkpoint_root() / LANGUAGE_MODEL_FILENAME


@dataclass(frozen=True)
class CheckpointPaths:
    tokenizer: Path
    language_model: Path


def default_checkpoints() -> CheckpointPaths:
    return CheckpointPaths(
        tokenizer=default_tokenizer_checkpoint(),
        language_model=default_language_model_checkpoint(),
    )


class APTTokenizer:
    """Thin wrapper around the DAE tokenizer."""

    @staticmethod
    def from_pretrained(checkpoint_path: Optional[Path | str] = None) -> DAE:
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else default_tokenizer_checkpoint()
        if not ckpt.exists():
            download_model_weights(ckpt)
        return DAE.from_pretrained(ckpt)


class APTLanguageModel:
    """Thin wrapper around GPTWithEncoder with tokenizer override support."""

    @staticmethod
    def from_pretrained(
        checkpoint_path: Optional[Path | str] = None,
        tokenizer_checkpoint: Optional[Path | str] = None,
    ) -> GPTWithEncoder:
        lm_ckpt = Path(checkpoint_path) if checkpoint_path is not None else default_language_model_checkpoint()
        tok_ckpt = (
            Path(tokenizer_checkpoint)
            if tokenizer_checkpoint is not None
            else default_tokenizer_checkpoint()
        )
        if not tok_ckpt.exists():
            download_model_weights(tok_ckpt)
        if not lm_ckpt.exists():
            download_model_weights(lm_ckpt)

        ckpt = torch.load(lm_ckpt, map_location="cpu")
        cfg = GPTWithEncoderConfig(**ckpt["model_cfg"])
        cfg.pth_to_tokenizer = str(tok_ckpt)

        model = GPTWithEncoder(cfg)
        state_dict = ckpt.get("ema_model", ckpt.get("model"))
        if state_dict is None:
            raise KeyError("Checkpoint missing both 'ema_model' and 'model' keys.")

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k_ = k.replace("_orig_mod.", "") if "_orig_mod" in k else k
            new_state_dict[k_] = v

        model.load_state_dict(new_state_dict)
        return model


__all__ = [
    "APTLanguageModel",
    "APTTokenizer",
    "CheckpointPaths",
    "default_checkpoints",
    "default_language_model_checkpoint",
    "default_tokenizer_checkpoint",
    "download_model_weights",
]


def download_model_weights(destination: Path | str) -> Path:
    """
    Download model weights from the Hugging Face Hub into the given destination path.

    The filename determines which artifact is fetched:
    - tokenizer128.pt
    - lm128.pt
    """
    dest = Path(destination)
    filename = dest.name
    if filename not in {TOKENIZER_FILENAME, LANGUAGE_MODEL_FILENAME}:
        raise ValueError(
            f"Unknown filename '{filename}'. Expected '{TOKENIZER_FILENAME}' or '{LANGUAGE_MODEL_FILENAME}'."
        )
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=dest.parent,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)
