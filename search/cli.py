"""Command line: make data, build the index, evaluate.

  python -m search.cli synth   --out data/synth --n 300
  python -m search.cli flickr  --captions data/flickr8k/captions.txt --images data/flickr8k/Images --out data/flickr8k/manifest.jsonl
  python -m search.cli index   --manifest data/synth/manifest.jsonl --encoder standin --index artifacts/qdrant
  python -m search.cli eval    --manifest data/synth/manifest.jsonl --encoder standin --index artifacts/qdrant --relevance attrs
"""
from __future__ import annotations

import argparse
import json
import os
import time

from PIL import Image

from search.data import flickr_captions_to_manifest, make_synthetic, read_manifest
from search.encoders import make_encoder
from search.evaluate import build_queries, evaluate
from search.index import ImageIndex


def cmd_index(a) -> dict:
    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    enc = make_encoder(a.encoder, a.model_id)
    index = ImageIndex(enc.dim, a.index, recreate=True)
    t0 = time.time()
    for s in range(0, len(rows), a.batch_size):
        chunk = rows[s:s + a.batch_size]
        imgs = []
        for r in chunk:
            with Image.open(r["image"]) as im:
                imgs.append(im.convert("RGB").copy())
        index.add(list(range(s, s + len(chunk))), enc.encode_images(imgs),
                  [{"path": r["image"], "caption": r["captions"][0] if r["captions"] else ""} for r in chunk])
    info = {"encoder": enc.name, "dim": enc.dim, "images": index.count(), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(info))
    return info


def cmd_eval(a) -> dict:
    rows = read_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    enc = make_encoder(a.encoder, a.model_id)
    index = ImageIndex(enc.dim, a.index)
    if index.count() != len(rows):
        raise SystemExit(f"index holds {index.count()} images but the manifest has {len(rows)} "
                         f"(use the same --manifest/--limit as when indexing)")
    texts, rel = build_queries(rows, a.relevance)
    res = evaluate(enc, index, texts, rel)
    res.update({"gallery_size": len(rows), "encoder": enc.name, "relevance": a.relevance})
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth"); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("flickr"); f.add_argument("--captions", required=True)
    f.add_argument("--images", required=True); f.add_argument("--out", required=True)

    for name in ("index", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--encoder", default="siglip", choices=["siglip", "standin"])
        p.add_argument("--model-id", default=None)
        p.add_argument("--index", default="artifacts/qdrant")
        p.add_argument("--limit", type=int, default=None, help="use only the first N manifest rows")
        if name == "index":
            p.add_argument("--batch-size", type=int, default=64)
        else:
            p.add_argument("--relevance", default="image", choices=["image", "attrs"])
            p.add_argument("--out", default=None, help="write metrics JSON here")

    a = ap.parse_args(argv)
    if a.cmd == "synth":
        print(make_synthetic(a.out, a.n, seed=a.seed))
    elif a.cmd == "flickr":
        print(flickr_captions_to_manifest(a.captions, a.images, a.out), "images")
    elif a.cmd == "index":
        cmd_index(a)
    else:
        cmd_eval(a)


if __name__ == "__main__":
    main()
