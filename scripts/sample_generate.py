from __future__ import annotations

from pathlib import Path

import torch

from apt.models import APTLanguageModel, APTTokenizer
from apt.generate import generate_spda


def write_ca_pdb(path, coords):
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = APTTokenizer.from_pretrained()
    model = APTLanguageModel.from_pretrained()

    tokenizer = tokenizer.to(device).eval()
    model = model.to(device).eval()

    seqs = generate_spda(
        model,
        batch_size=1,
        sampling_method="minp",
        threshold=0.15,
        temperature=1.0,
    )
    idx_BL = seqs[0].unsqueeze(0)

    cfg_fn = lambda t: 1 - t
    with torch.no_grad():
        coords = tokenizer.decode_with_cfg_fn(
            idx_BL,
            cfg_fn=cfg_fn,
            true_length=idx_BL.size(1),
            n_steps=100,
            noise_weight=0.1,
            score_weight=1.0,
        )
    coords = coords.squeeze(0).cpu() * 10.0
    out_path = "examples/generated_sample.pdb"
    write_ca_pdb(Path(out_path), coords)
    print(f"Sampled coords shape: {coords.shape}")
    print(f"Wrote PDB: {out_path}")


if __name__ == "__main__":
    main()
