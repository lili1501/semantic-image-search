"""Image/text encoders that map both modalities into one L2-normalised embedding space.

Two implementations:
  * SigLIPEncoder     - the real model (needs torch + transformers + the weights; NOT runnable in the
                        sandbox this repo was built in, see README "Verified vs not verified").
  * ColorShapeEncoder - a tiny hand-written stand-in used only to test the pipeline (indexing, vector
                        search, metrics, API). It understands "red circle"-style captions on synthetic
                        images and says nothing about real-model quality.
"""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    name: str
    dim: int

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def as_feature_tensor(out):
    """transformers' get_*_features returns a tensor in older releases and a ModelOutput (embedding in
    `pooler_output`) in newer ones. Accept both."""
    if hasattr(out, "float"):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"cannot read embeddings from {type(out).__name__}; expected a tensor or an output with pooler_output")
    return pooled


class SigLIPEncoder:
    """Google SigLIP via Hugging Face transformers. Heavy imports happen lazily in __init__."""

    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device: str | None = None,
                 batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.name = model_id
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dim = int(self.encode_texts(["dim probe"]).shape[1])   # measured, not read from config

    def _batches(self, items):
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for b in self._batches(images):
                inp = self.processor(images=[im.convert("RGB") for im in b], return_tensors="pt").to(self.device)
                out.append(as_feature_tensor(self.model.get_image_features(**inp)).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for b in self._batches(texts):
                # SigLIP was trained with fixed-length padding; other padding hurts quality.
                inp = self.processor(text=b, padding="max_length", max_length=64, truncation=True,
                                     return_tensors="pt").to(self.device)
                out.append(as_feature_tensor(self.model.get_text_features(**inp)).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))


PALETTE = {
    "red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 30),
    "orange": (250, 140, 20), "purple": (140, 40, 180), "black": (20, 20, 20), "cyan": (30, 200, 220),
}
COLORS = list(PALETTE)
SHAPES = ["circle", "square", "triangle"]
_SHAPE_SYNONYMS = {"circle": "circle", "round": "circle", "disc": "circle", "square": "square",
                   "box": "square", "triangle": "triangle"}
_SHAPE_FILL = {"circle": np.pi / 4, "square": 1.0, "triangle": 0.5}  # area / bounding-box area


class ColorShapeEncoder:
    """Test stand-in: 8 colour dims + 3 shape dims. Images are classified by mean colour and fill ratio."""

    name = "color-shape-standin"
    dim = len(COLORS) + len(SHAPES)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
            mask = a.min(axis=2) < 230                       # non-white pixels = the object
            if not mask.any():
                continue
            mean = a[mask].mean(axis=0)
            c = int(np.argmin([np.linalg.norm(mean - np.array(PALETTE[k])) for k in COLORS]))
            ys, xs = np.where(mask)
            box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            fill = mask.sum() / box
            s = int(np.argmin([abs(fill - _SHAPE_FILL[k]) for k in SHAPES]))
            out[i, c] = 1.0
            out[i, len(COLORS) + s] = 1.0
        return l2_normalize(out)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                if w in PALETTE:
                    out[i, COLORS.index(w)] = 1.0
                elif w in _SHAPE_SYNONYMS:
                    out[i, len(COLORS) + SHAPES.index(_SHAPE_SYNONYMS[w])] = 1.0
        return l2_normalize(out)


def make_encoder(kind: str, model_id: str | None = None, **kw) -> Encoder:
    if kind == "siglip":
        return SigLIPEncoder(model_id or "google/siglip-base-patch16-224", **kw)
    if kind == "standin":
        return ColorShapeEncoder()
    raise ValueError(f"unknown encoder '{kind}' (use 'siglip' or 'standin')")
