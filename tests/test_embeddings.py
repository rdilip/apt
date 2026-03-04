from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from apt.embeddings import canonicalize_point_cloud, embed_point_clouds
from apt.fsq import FSQ
from apt.vector_db import SimpleVectorDB, build_vector_db_from_point_clouds


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(m)
    d = torch.sign(torch.diag(r))
    d[d == 0] = 1
    q = q @ torch.diag(d)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _make_cloud(seed: int, n_points: int = 72) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_points, 3, generator=g)
    # Anisotropic scaling reduces PCA axis ambiguities.
    x = x * torch.tensor([1.7, 0.8, 2.4]) + torch.tensor([0.2, -0.3, 1.1])
    return x


def _load_ca_coords(pdb_path: Path) -> torch.Tensor:
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
    out = torch.tensor(coords, dtype=torch.float32)
    out = out - out.mean(dim=0, keepdim=True)
    return out / 10.0


class DummyTokenizer:
    """
    Small deterministic tokenizer-like object for embedding tests.
    """

    def __init__(self, levels=(8, 8, 8, 8, 8), n_tokens: int = 64):
        self.cfg = SimpleNamespace(n_tokens=n_tokens)
        n_levels = len(levels)
        self.quantize = FSQ(levels=list(levels), dim=n_levels, dim_out=n_levels)

    def encode(self, x_BLD: torch.Tensor):
        if x_BLD.ndim != 3 or x_BLD.shape[-1] != 3:
            raise ValueError(f"Expected (B, L, 3), got {tuple(x_BLD.shape)}")
        radius = torch.linalg.norm(x_BLD, dim=-1, keepdim=True)
        xy = x_BLD[..., :1] * x_BLD[..., 1:2]
        feats = torch.cat([x_BLD, radius, xy], dim=-1)
        q_BLD, idx_BL = self.quantize(feats)
        return feats, q_BLD, idx_BL

    def embed(self, x_BLD: torch.Tensor, **kwargs):
        return embed_point_clouds(self, x_BLD, **kwargs)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = DummyTokenizer(levels=(8, 8, 8, 8, 8), n_tokens=64)

    def test_default_embedding_dimension_matches_ntoks_times_levels(self):
        cloud = _make_cloud(seed=10, n_points=80)
        emb = self.tokenizer.embed(cloud, ntoks=32, canonicalize=True)
        self.assertEqual(tuple(emb.shape), (1, 160))  # 32 * 5

    def test_canonicalization_is_rotation_invariant(self):
        cloud = _make_cloud(seed=11, n_points=80)
        rot = _random_rotation(seed=3)
        rotated = cloud @ rot

        canonical_a = canonicalize_point_cloud(cloud)
        canonical_b = canonicalize_point_cloud(rotated)
        self.assertTrue(torch.allclose(canonical_a, canonical_b, atol=1e-5, rtol=1e-5))

        emb_a = self.tokenizer.embed(cloud, ntoks=32, canonicalize=True)
        emb_b = self.tokenizer.embed(rotated, ntoks=32, canonicalize=True)
        self.assertTrue(torch.allclose(emb_a, emb_b, atol=1e-6, rtol=1e-6))

    def test_vector_db_returns_same_and_similar_neighbors(self):
        base = _make_cloud(seed=21, n_points=80)
        similar = base + 0.01 * _make_cloud(seed=22, n_points=80)
        different = _make_cloud(seed=99, n_points=80) * torch.tensor([0.3, 2.3, 1.2])

        db, _ = build_vector_db_from_point_clouds(
            self.tokenizer,
            [base, similar, different],
            ntoks=32,
            canonicalize=True,
            metric="cosine",
            ids=["base", "similar", "different"],
        )

        query = base @ _random_rotation(seed=7)
        query_emb = self.tokenizer.embed(query, ntoks=32, canonicalize=True)
        results = db.search(query_emb, k=3)[0]
        ranked_ids = [r.id for r in results]

        self.assertEqual(ranked_ids[0], "base")
        self.assertIn("similar", ranked_ids[:2])
        self.assertEqual(ranked_ids[-1], "different")

        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertGreater(results[0].score, results[-1].score)

    def test_simple_vector_db_euclidean_mode(self):
        db = SimpleVectorDB(metric="euclidean")
        db.add([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]], ids=["a", "b", "c"])
        out = db.search([0.2, 0.2], k=2)[0]
        self.assertEqual([r.id for r in out], ["a", "b"])

    def test_real_protein_example_neighbors(self):
        pdb_path = Path(__file__).resolve().parents[1] / "examples" / "1CRN.pdb"
        base = _load_ca_coords(pdb_path)
        g = torch.Generator().manual_seed(42)
        similar = base + 0.002 * torch.randn(base.shape, generator=g)
        different = _make_cloud(seed=404, n_points=base.shape[0])

        db, _ = build_vector_db_from_point_clouds(
            self.tokenizer,
            [base, similar, different],
            ntoks=32,
            canonicalize=True,
            metric="cosine",
            ids=["1CRN", "1CRN_noisy", "random"],
        )

        query = base @ _random_rotation(seed=123)
        query_emb = self.tokenizer.embed(query, ntoks=32, canonicalize=True)
        results = db.search(query_emb, k=3)[0]
        ranked_ids = [r.id for r in results]

        self.assertEqual(ranked_ids[0], "1CRN")
        self.assertIn("1CRN_noisy", ranked_ids[:2])


if __name__ == "__main__":
    unittest.main()
