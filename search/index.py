"""Thin wrapper around a Qdrant collection of image embeddings (cosine distance)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

COLLECTION = "images"


@dataclass
class Hit:
    id: int
    score: float
    payload: dict


class ImageIndex:
    """location: ':memory:', a directory path (embedded, persisted), or a server URL like http://host:6333."""

    def __init__(self, dim: int, location: str = ":memory:", collection: str = COLLECTION, recreate: bool = False):
        self.dim, self.collection = dim, collection
        self.client = QdrantClient(url=location) if location.startswith("http") else (
            QdrantClient(location=location) if location == ":memory:" else QdrantClient(path=location))
        exists = self.client.collection_exists(collection)
        if exists and recreate:
            self.client.delete_collection(collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        else:
            have = self.client.get_collection(collection).config.params.vectors.size
            if have != dim:
                raise ValueError(f"collection '{collection}' has dim {have}, encoder produces {dim}")

    def add(self, ids: list[int], vectors: np.ndarray, payloads: list[dict], batch: int = 512) -> None:
        assert len(ids) == len(vectors) == len(payloads)
        for i in range(0, len(ids), batch):
            self.client.upsert(self.collection, points=models.Batch(
                ids=[int(x) for x in ids[i:i + batch]],
                vectors=vectors[i:i + batch].astype(np.float32).tolist(),
                payloads=payloads[i:i + batch]))

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, vector: np.ndarray, k: int = 10, exact: bool = False,
               where: dict | None = None) -> list[Hit]:
        """where: payload equality filter, e.g. {"split": "test"}."""
        flt = models.Filter(must=[models.FieldCondition(key=k_, match=models.MatchValue(value=v))
                                  for k_, v in where.items()]) if where else None
        res = self.client.query_points(
            self.collection, query=np.asarray(vector, dtype=np.float32).tolist(), limit=k, with_payload=True,
            query_filter=flt, search_params=models.SearchParams(exact=exact)).points
        return [Hit(int(p.id), float(p.score), dict(p.payload or {})) for p in res]
