from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class Neighbor:
    id: Any
    index: int
    score: float
    payload: Any = None


def _to_2d_float_tensor(x: Tensor | Sequence[float] | Sequence[Sequence[float]]) -> Tensor:
    t = x if torch.is_tensor(x) else torch.tensor(x)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {tuple(t.shape)}")
    return t.to(dtype=torch.float32, device="cpu")


class SimpleVectorDB:
    """
    In-memory vector database with exact nearest-neighbor search.

    Supported metrics:
    - cosine: higher score is better
    - euclidean: lower score is better
    """

    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric}")
        self.metric = metric
        self._embeddings: Tensor | None = None
        self._ids: list[Any] = []
        self._payloads: list[Any] = []

    def __len__(self) -> int:
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])

    @property
    def embeddings(self) -> Tensor:
        if self._embeddings is None:
            return torch.empty(0, 0, dtype=torch.float32)
        return self._embeddings

    @property
    def ids(self) -> list[Any]:
        return list(self._ids)

    def add(
        self,
        embeddings: Tensor | Sequence[Sequence[float]] | Sequence[float],
        *,
        ids: Sequence[Any] | None = None,
        payloads: Sequence[Any] | None = None,
    ) -> None:
        emb = _to_2d_float_tensor(embeddings)
        n_new = emb.shape[0]

        if ids is None:
            ids = [len(self._ids) + i for i in range(n_new)]
        if payloads is None:
            payloads = [None] * n_new

        if len(ids) != n_new:
            raise ValueError(f"ids length ({len(ids)}) must match number of embeddings ({n_new})")
        if len(payloads) != n_new:
            raise ValueError(f"payloads length ({len(payloads)}) must match number of embeddings ({n_new})")

        if self._embeddings is None:
            self._embeddings = emb
        else:
            if emb.shape[1] != self._embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._embeddings.shape[1]}, got {emb.shape[1]}"
                )
            self._embeddings = torch.cat([self._embeddings, emb], dim=0)

        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def _pairwise_scores(self, queries_QD: Tensor) -> Tensor:
        if self._embeddings is None:
            raise ValueError("Vector DB is empty. Add embeddings before searching.")
        if queries_QD.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                f"Query dimension mismatch: expected {self._embeddings.shape[1]}, got {queries_QD.shape[1]}"
            )

        if self.metric == "cosine":
            q = F.normalize(queries_QD, dim=-1)
            d = F.normalize(self._embeddings, dim=-1)
            return q @ d.transpose(0, 1)
        return torch.cdist(queries_QD, self._embeddings, p=2)

    def search(
        self,
        query_embeddings: Tensor | Sequence[float] | Sequence[Sequence[float]],
        *,
        k: int = 5,
    ) -> list[list[Neighbor]]:
        queries = _to_2d_float_tensor(query_embeddings)
        scores_QN = self._pairwise_scores(queries)

        if len(self) == 0:
            return [[] for _ in range(queries.shape[0])]
        k = max(1, min(k, len(self)))
        largest = self.metric == "cosine"
        vals_QK, idx_QK = torch.topk(scores_QN, k=k, dim=-1, largest=largest)

        all_results: list[list[Neighbor]] = []
        for qi in range(queries.shape[0]):
            results_q: list[Neighbor] = []
            for rank in range(k):
                db_idx = int(idx_QK[qi, rank].item())
                results_q.append(
                    Neighbor(
                        id=self._ids[db_idx],
                        index=db_idx,
                        score=float(vals_QK[qi, rank].item()),
                        payload=self._payloads[db_idx],
                    )
                )
            all_results.append(results_q)
        return all_results


@torch.no_grad()
def build_vector_db_from_point_clouds(
    tokenizer,
    point_clouds: Sequence[Tensor],
    *,
    ntoks: int = 32,
    canonicalize: bool = True,
    metric: str = "cosine",
    ids: Sequence[Any] | None = None,
    payloads: Sequence[Any] | None = None,
) -> tuple[SimpleVectorDB, Tensor]:
    """
    Convenience helper:
    - embeds each point cloud with tokenizer.embed(...),
    - inserts embeddings into a SimpleVectorDB,
    - returns (db, embeddings).
    """
    embeddings = []
    for cloud in point_clouds:
        emb = tokenizer.embed(cloud, ntoks=ntoks, canonicalize=canonicalize)
        embeddings.append(emb.squeeze(0).detach().cpu())
    emb_ND = torch.stack(embeddings, dim=0)

    db = SimpleVectorDB(metric=metric)
    db.add(emb_ND, ids=ids, payloads=payloads)
    return db, emb_ND

