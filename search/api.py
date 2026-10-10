"""FastAPI service.

  GET  /health
  GET  /search?q=red circle&k=5        text -> images
  POST /search/image  (file upload)    image -> similar images

Run:  ENCODER=siglip INDEX_PATH=artifacts/qdrant uvicorn search.api:create_app --factory --port 8000
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from search.encoders import Encoder, make_encoder
from search.index import ImageIndex


def create_app(encoder: Encoder | None = None, index: ImageIndex | None = None) -> FastAPI:
    if encoder is None:
        encoder = make_encoder(os.environ.get("ENCODER", "siglip"), os.environ.get("MODEL_ID"))
    if index is None:
        index = ImageIndex(encoder.dim, os.environ.get("INDEX_PATH", "artifacts/qdrant"))
    app = FastAPI(title="Semantic image search")

    def fmt(hits):
        return [{"id": h.id, "score": round(h.score, 5), **h.payload} for h in hits]

    @app.get("/health")
    def health():
        return {"status": "ok", "encoder": encoder.name, "dim": encoder.dim, "images": index.count()}

    @app.get("/search")
    def search_text(q: str = Query(..., min_length=1, max_length=300), k: int = Query(10, ge=1, le=100)):
        if not q.strip():
            raise HTTPException(422, "empty query")
        return {"query": q, "results": fmt(index.search(encoder.encode_texts([q])[0], k))}

    @app.post("/search/image")
    async def search_image(file: UploadFile = File(...), k: int = Query(10, ge=1, le=100)):
        try:
            img = Image.open(io.BytesIO(await file.read()))
            img.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "not a valid image")
        return {"results": fmt(index.search(encoder.encode_images([img])[0], k))}

    return app
