"""Text-to-image retrieval metrics: Recall@k, MRR, and latency percentiles.

Every query has a SET of relevant image ids. For caption datasets (Flickr, COCO) that set is the single
source image; for the synthetic set it is every image with the same colour+shape.
Recall@k = fraction of queries with at least one relevant image in the top k (the standard
text-to-image Recall@K definition when there is one ground-truth image).
"""
from __future__ import annotations

import time

import numpy as np

from search.index import ImageIndex


def build_queries(rows: list[dict], relevance: str = "image") -> tuple[list[str], list[set[int]]]:
    """relevance='image': a caption's only relevant image is its source. 'attrs': any image with equal attrs."""
    texts, rel = [], []
    by_attrs: dict[tuple, set[int]] = {}
    if relevance == "attrs":
        for i, r in enumerate(rows):
            by_attrs.setdefault(tuple(r["attrs"]), set()).add(i)
    for i, r in enumerate(rows):
        for cap in r["captions"]:
            texts.append(cap)
            rel.append(by_attrs[tuple(r["attrs"])] if relevance == "attrs" else {i})
    return texts, rel


def recall_at_k(ranked: list[list[int]], relevant: list[set[int]], ks=(1, 5, 10)) -> dict:
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([bool(set(r[:k]) & rel) for r, rel in zip(ranked, relevant)]))
    rr = []
    for r, rel in zip(ranked, relevant):
        pos = next((j for j, x in enumerate(r) if x in rel), None)
        rr.append(0.0 if pos is None else 1.0 / (pos + 1))
    out["mrr"] = float(np.mean(rr))
    return out


def evaluate(encoder, index: ImageIndex, texts: list[str], relevant: list[set[int]], ks=(1, 5, 10),
             exact: bool = False) -> dict:
    kmax = max(ks)
    t0 = time.perf_counter()
    q = encoder.encode_texts(texts)
    encode_s = time.perf_counter() - t0

    ranked, lat = [], []
    for v in q:
        t = time.perf_counter()
        hits = index.search(v, k=kmax, exact=exact)
        lat.append((time.perf_counter() - t) * 1000)
        ranked.append([h.id for h in hits])
    res = recall_at_k(ranked, relevant, ks)
    res.update({
        "n_queries": len(texts),
        "search_ms_p50": float(np.percentile(lat, 50)),
        "search_ms_p95": float(np.percentile(lat, 95)),
        "text_encode_ms_per_query": encode_s / max(len(texts), 1) * 1000,
    })
    return res


def ann_overlap(encoder, index: ImageIndex, texts: list[str], k: int = 10) -> float:
    """Mean fraction of the exact top-k that the approximate (HNSW) search also returns. 1.0 = no recall loss.
    Embedded local Qdrant is brute force, so this is only informative against a Qdrant server."""
    q = encoder.encode_texts(texts)
    ov = []
    for v in q:
        a = {h.id for h in index.search(v, k, exact=False)}
        b = {h.id for h in index.search(v, k, exact=True)}
        ov.append(len(a & b) / max(len(b), 1))
    return float(np.mean(ov))
