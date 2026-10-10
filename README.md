# Semantic Image Search (SigLIP + Qdrant)

Search an image collection with natural language ("a dog catching a frisbee") or with another image.
SigLIP embeds images and text into one space; a Qdrant collection stores the image vectors and answers
cosine nearest-neighbour queries; a FastAPI service exposes it and a Streamlit page is the UI.

> **Status.** The pipeline, index, metrics, API and UI are tested here with a tiny stand-in encoder on
> synthetic coloured-shape images. **The real SigLIP model and real photo datasets were never run**
> (the build sandbox cannot reach Hugging Face), so this repo contains **no retrieval-quality numbers for
> SigLIP**. See "Verified vs not verified" and "Running with SigLIP".

```
images ──SigLIP image tower──▶ L2-normalised vectors ──▶ Qdrant (cosine) ◀── SigLIP text tower ◀── "query text"
                                                              │
                                                  FastAPI  /search  /search/image ──▶ Streamlit UI
```

| File | Role |
|---|---|
| `search/encoders.py` | `SigLIPEncoder` (real), `ColorShapeEncoder` (test stand-in), `make_encoder` |
| `search/index.py` | Qdrant wrapper: batched upsert, top-k cosine search, payload filter, dimension check |
| `search/data.py` | JSONL manifest I/O, Flickr8k/30k `captions.txt` converter, synthetic image generator |
| `search/evaluate.py` | Recall@1/5/10, MRR, search-latency p50/p95, ANN-vs-exact overlap |
| `search/api.py` | FastAPI app factory: `/health`, `/search?q=&k=`, `POST /search/image` |
| `search/cli.py` | `synth`, `flickr`, `index`, `eval` commands |
| `app/streamlit_app.py` | Text and image search UI |

## Quick start (stand-in encoder, works anywhere)

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests -q                                   # 15 tests
python -m search.cli synth --out data/synth --n 300
python -m search.cli index --manifest data/synth/manifest.jsonl --encoder standin --index artifacts/qdrant
python -m search.cli eval  --manifest data/synth/manifest.jsonl --encoder standin --index artifacts/qdrant --relevance attrs
ENCODER=standin INDEX_PATH=artifacts/qdrant uvicorn search.api:create_app --factory --port 8000
curl "localhost:8000/search?q=green%20square&k=3"
```

## Running with SigLIP (what you need to do; untested here)

```bash
pip install -r requirements-siglip.txt        # torch, transformers, streamlit
# 1. Get a captioned dataset, e.g. Flickr8k (8,091 images, 5 captions each) or Flickr30k, with a captions.txt
#    (CSV header `image,caption`). Then:
python -m search.cli flickr --captions data/flickr8k/captions.txt --images data/flickr8k/Images --out data/flickr8k/manifest.jsonl
# 2. Embed + index (first run downloads google/siglip-base-patch16-224 from Hugging Face; use --model-id for another size)
python -m search.cli index --manifest data/flickr8k/manifest.jsonl --encoder siglip --index artifacts/qdrant --batch-size 64
# 3. Evaluate: each caption's only correct image is its source image
python -m search.cli eval --manifest data/flickr8k/manifest.jsonl --encoder siglip --index artifacts/qdrant \
    --relevance image --out reports/flickr8k_siglip.json
# 4. Serve + UI
ENCODER=siglip INDEX_PATH=artifacts/qdrant uvicorn search.api:create_app --factory --port 8000
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

Notes for reporting numbers: Recall@K depends on gallery size, so quote it with the number of images
(`gallery_size` in the JSON). `search_ms_*` is the Qdrant query only; `text_encode_ms_per_query` is
reported separately, and end-to-end API latency has to be measured against the running service.
The model weights and datasets carry their own licences; check them before redistributing anything.
Embedded Qdrant (a local folder) is brute-force exact search; to measure real HNSW approximation loss
run a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`), pass `--index http://localhost:6333`, and use
`search.evaluate.ann_overlap`.

## Verified vs not verified

**Verified by running it (15 passing tests plus live runs):**
- Recall@k and MRR against hand-computed values; a shuffled-label control scores near chance; exact-image relevance scores far below attribute relevance on the synthetic set, so the metric is not inflated.
- Qdrant index: correct neighbour order and scores, payload filtering, on-disk persistence, dimension-mismatch rejection.
- The stand-in encoder classifies all 240 synthetic test images to their true colour and shape.
- The CLI `synth → index → eval` chain, including rejection of a manifest/index size mismatch.
- A real `uvicorn` server answering `/health`, `/search` (correct results, 422 on empty query) and `/search/image` over HTTP.
- The Streamlit page loads, reads `/health`, and renders the 9 result images for a text query (Streamlit's `AppTest` harness against the live API; not viewed in a browser).
- The Flickr `captions.txt` converter on a small hand-made file.

**Not verified:**
- **Anything involving SigLIP**: model loading, the embedding dimension, GPU use, embedding throughput, and retrieval quality. `SigLIPEncoder` has never been executed; expect to fix small issues on first run.
- Real datasets (Flickr, COCO), including the converter on a real file.
- Qdrant server mode, HNSW recall, and behaviour at hundreds of thousands of images.
- Latency under load; the in-process timings here (~0.2 ms search over 300 vectors) say nothing about SigLIP-scale deployments.
- Hugging Face Spaces deployment; no Dockerfile is included.
- The stand-in is a hand-written lookup, not a trained model. Its perfect attribute recall only shows the plumbing is correct.

## Known limitations / next steps
- Query images are not stored: `/search/image` embeds the upload and discards it.
- No re-ranking, query expansion or hybrid (keyword + vector) search.
- Add a Dockerfile and Spaces app once a real run works; add fine-tuning only if zero-shot Recall is the weak point.
