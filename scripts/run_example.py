from __future__ import annotations

from pathlib import Path

import torch

from apt.models import APTLanguageModel, APTTokenizer
from apt.utils import kabsch_rmsd


def load_ca_coords(pdb_path: Path) -> torch.Tensor:
    coords = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17]
            if atom_name != "CA":
                continue
            if altloc not in (" ", "A"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    x_L3 = torch.tensor(coords, dtype=torch.float32)
    # zero-center and scale
    x_L3 = x_L3 - x_L3.mean(dim=0, keepdim=True)
    x_L3 = x_L3 / 10.0
    return x_L3


def main() -> None:
    pdb_path = Path("examples/1CRN.pdb")
    x_L3 = load_ca_coords(pdb_path)
    x_BLD = x_L3.unsqueeze(0)

    print(f"Loaded {pdb_path} with {x_L3.shape[0]} residues")

    tokenizer = APTTokenizer.from_pretrained().cuda(0)
    model = APTLanguageModel.from_pretrained().cuda(0)
    x_BLD = x_BLD.cuda(0)

    max_toks = tokenizer.cfg.n_tokens

    with torch.no_grad():
        _, tok_BLD, idx_BL = tokenizer.encode(x_BLD)
        idx_BL = idx_BL[:, :max_toks]
        recon = tokenizer.decode(idx_BL)
        rmsd = kabsch_rmsd(recon.cpu(), x_BLD.cpu()) * 10 
    
    print("Reconstruction RMSD (CA, Å):", rmsd.item())



if __name__ == "__main__":
    main()
