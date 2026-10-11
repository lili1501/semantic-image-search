"""Datasets: a JSONL manifest format, a Flickr8k/30k-style captions.txt converter, and a synthetic generator.

Manifest: one JSON object per line: {"image": "relative/or/absolute/path.jpg", "captions": ["...", "..."]}
Row number (0-based) is the image id used in the index.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageDraw

from search.encoders import COLORS, PALETTE, SHAPES


def read_manifest(path: str) -> list[dict]:
    root = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not os.path.isabs(r["image"]):
                    r["image"] = os.path.join(root, r["image"])
                rows.append(r)
    return rows


def write_manifest(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def flickr_captions_to_manifest(captions_file: str, images_dir: str, out_path: str) -> int:
    """Convert Kaggle-style Flickr8k/30k `captions.txt` (CSV with header `image,caption`) to a manifest.
    Image paths are written relative to the manifest, so keep the manifest next to / above the images folder."""
    caps = defaultdict(list)
    with open(captions_file, newline="") as f:
        for row in csv.DictReader(f):
            caps[row["image"].strip()].append(row["caption"].strip())
    base = os.path.dirname(os.path.abspath(out_path))
    rows = [{"image": os.path.relpath(os.path.join(images_dir, name), base), "captions": c}
            for name, c in sorted(caps.items())]
    write_manifest(rows, out_path)
    return len(rows)


def coco_captions_to_manifest(captions_json: str, images_dir: str, out_path: str) -> int:
    """Convert a COCO captions file (captions_val2017.json) to a manifest. Images that are listed but missing on disk
    are skipped, so a partial image download still works. Paths are relative to the manifest."""
    with open(captions_json) as f:
        coco = json.load(f)
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    caps = defaultdict(list)
    for a in coco["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    base = os.path.dirname(os.path.abspath(out_path))
    rows = []
    for iid in sorted(files):
        path = os.path.join(images_dir, files[iid])
        if caps.get(iid) and os.path.exists(path):
            rows.append({"image": os.path.relpath(path, base), "captions": caps[iid]})
    if not rows:
        raise ValueError("no images with captions found; check --images points at the extracted val2017 folder")
    write_manifest(rows, out_path)
    return len(rows)


_TEMPLATES = ["a {c} {s}", "{c} {s} on a white background", "a {s} that is {c}", "picture of a {c} {s}"]


def make_synthetic(out_dir: str, n: int = 200, size: int = 64, seed: int = 0) -> str:
    """Draw n images of one coloured shape on white. Returns the manifest path.
    Several images share each (colour, shape) pair, so relevance is attribute-based: see `attrs` in each row."""
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rows = []
    for i in range(n):
        color, shape = rng.choice(COLORS), rng.choice(SHAPES)
        im = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(im)
        w = rng.randint(size // 4, size // 2)
        x0, y0 = rng.randint(2, size - w - 2), rng.randint(2, size - w - 2)
        box = [x0, y0, x0 + w, y0 + w]
        fill = PALETTE[color]
        if shape == "circle":
            d.ellipse(box, fill=fill)
        elif shape == "square":
            d.rectangle(box, fill=fill)
        else:
            d.polygon([(x0 + w // 2, y0), (x0, y0 + w), (x0 + w, y0 + w)], fill=fill)
        rel = f"images/{i:05d}.png"
        im.save(os.path.join(out_dir, rel))
        caps = [t.format(c=color, s=shape) for t in rng.sample(_TEMPLATES, 2)]
        rows.append({"image": rel, "captions": caps, "attrs": [color, shape]})
    path = os.path.join(out_dir, "manifest.jsonl")
    write_manifest(rows, path)
    return path
