import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from search import cli
from search.api import create_app
from search.data import flickr_captions_to_manifest, make_synthetic, read_manifest
from search.encoders import ColorShapeEncoder, SigLIPEncoder, make_encoder
from search.evaluate import build_queries, evaluate, recall_at_k
from search.index import ImageIndex


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    path = make_synthetic(str(d), n=240, seed=1)
    return read_manifest(path), path


@pytest.fixture(scope="module")
def built(synth):
    rows, _ = synth
    enc = ColorShapeEncoder()
    imgs = [Image.open(r["image"]).convert("RGB") for r in rows]
    idx = ImageIndex(enc.dim)
    idx.add(list(range(len(rows))), enc.encode_images(imgs), [{"path": r["image"]} for r in rows])
    return enc, idx, rows


# ---- metrics against hand-computed answers --------------------------------------------------
def test_recall_and_mrr_known_values():
    ranked = [[5, 1, 2], [7, 8, 9], [3, 4, 0]]
    relevant = [{1}, {42}, {3}]
    r = recall_at_k(ranked, relevant, ks=(1, 2, 3))
    assert r["recall@1"] == pytest.approx(1 / 3)       # only query 3
    assert r["recall@2"] == pytest.approx(2 / 3)       # queries 1 and 3
    assert r["recall@3"] == pytest.approx(2 / 3)       # query 2 never found
    assert r["mrr"] == pytest.approx((1 / 2 + 0 + 1) / 3)


# ---- stand-in encoder ------------------------------------------------------------------------
def test_standin_encoder_classifies_every_synthetic_image(synth):
    rows, _ = synth
    enc = ColorShapeEncoder()
    v = enc.encode_images([Image.open(r["image"]).convert("RGB") for r in rows])
    t = enc.encode_texts([" ".join(r["attrs"]) for r in rows])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose((v * t).sum(1), 1, atol=1e-5), "image embedding should equal its own attribute text"


def test_encoder_factory_and_unknown_text():
    assert make_encoder("standin").dim == 11
    with pytest.raises(ValueError):
        make_encoder("nope")
    assert not np.any(ColorShapeEncoder().encode_texts(["gibberish words"]))   # zero vector, no crash


# ---- index -----------------------------------------------------------------------------------
def test_index_returns_known_neighbours_in_order():
    idx = ImageIndex(3)
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx.add([10, 11, 12, 13], vecs, [{"n": i} for i in range(4)])
    hits = idx.search(np.array([1, 0, 0]), k=3)
    assert [h.id for h in hits][:2] == [10, 11]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5) and hits[0].score >= hits[1].score >= hits[2].score
    assert idx.count() == 4


def test_index_payload_filter_and_dim_mismatch(tmp_path):
    idx = ImageIndex(2, str(tmp_path / "q"))
    idx.add([1, 2], np.array([[1, 0], [1, 0]], dtype=np.float32), [{"split": "a"}, {"split": "b"}])
    assert [h.id for h in idx.search(np.array([1, 0]), 5, where={"split": "b"})] == [2]
    del idx.client                       # release the on-disk lock before reopening
    with pytest.raises(ValueError):
        ImageIndex(5, str(tmp_path / "q"))


def test_index_persists_on_disk(tmp_path):
    p = str(tmp_path / "q")
    a = ImageIndex(2, p)
    a.add([1], np.array([[0, 1]], dtype=np.float32), [{"x": 1}])
    a.client.close()
    b = ImageIndex(2, p)
    assert b.count() == 1 and b.search(np.array([0, 1]), 1)[0].id == 1


# ---- end to end retrieval --------------------------------------------------------------------
def test_end_to_end_attribute_recall_is_perfect_and_chance_is_not(built):
    enc, idx, rows = built
    texts, rel = build_queries(rows, "attrs")
    res = evaluate(enc, idx, texts, rel)
    assert res["recall@1"] == 1.0 and res["recall@10"] == 1.0 and res["mrr"] == 1.0
    assert res["n_queries"] == 480 and res["search_ms_p95"] >= res["search_ms_p50"] > 0


def test_single_image_relevance_is_harder_than_attribute_relevance(built):
    """With 240 images over 24 attribute combos, ~10 images tie per query, so exact-image Recall@1 must be low.
    Guards against an evaluation that silently reports inflated numbers."""
    enc, idx, rows = built
    texts, rel = build_queries(rows, "image")
    res = evaluate(enc, idx, texts, rel)
    assert res["recall@1"] < 0.3 and res["recall@10"] < 1.0


def test_wrong_encoder_gets_chance_level(built):
    """Shuffled text->image pairing must score near chance, proving the metric measures something."""
    enc, idx, rows = built
    texts, rel = build_queries(rows, "attrs")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rel))
    res = evaluate(enc, idx, texts, [rel[i] for i in perm])
    assert res["recall@1"] < 0.25


def test_exact_and_default_search_agree_locally(built):
    enc, idx, _ = built
    q = enc.encode_texts(["red circle"])[0]
    assert [h.id for h in idx.search(q, 10, exact=True)] == [h.id for h in idx.search(q, 10)]


# ---- API -------------------------------------------------------------------------------------
def test_api_text_and_image_search(built):
    enc, idx, rows = built
    c = TestClient(create_app(enc, idx))
    assert c.get("/health").json() == {"status": "ok", "encoder": enc.name, "dim": 11, "images": 240}

    r = c.get("/search", params={"q": "blue triangle", "k": 5}).json()
    assert len(r["results"]) == 5
    id_to_attrs = {i: set(row["attrs"]) for i, row in enumerate(rows)}
    assert all(id_to_attrs[h["id"]] == {"blue", "triangle"} for h in r["results"])

    src = rows[7]
    buf = io.BytesIO(); Image.open(src["image"]).save(buf, "PNG")
    r2 = c.post("/search/image", params={"k": 3}, files={"file": ("q.png", buf.getvalue(), "image/png")}).json()
    assert all(id_to_attrs[h["id"]] == set(src["attrs"]) for h in r2["results"])


def test_api_validation(built):
    enc, idx, _ = built
    c = TestClient(create_app(enc, idx))
    assert c.get("/search", params={"q": ""}).status_code == 422
    assert c.get("/search", params={"q": "   "}).status_code == 422
    assert c.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert c.get("/search", params={"q": "x", "k": 101}).status_code == 422
    assert c.post("/search/image", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400


# ---- data + CLI ------------------------------------------------------------------------------
def test_flickr_converter(tmp_path):
    cap = tmp_path / "captions.txt"
    cap.write_text('image,caption\na.jpg,"A dog, running"\na.jpg,A brown dog\nb.jpg,Two kids\n')
    out = tmp_path / "m.jsonl"
    assert flickr_captions_to_manifest(str(cap), str(tmp_path / "Images"), str(out)) == 2
    rows = read_manifest(str(out))
    assert rows[0]["captions"] == ["A dog, running", "A brown dog"]
    assert rows[0]["image"].endswith("Images/a.jpg")


def test_cli_index_then_eval(synth, tmp_path, capsys):
    _, manifest = synth
    idx_dir = str(tmp_path / "q")
    cli.main(["index", "--manifest", manifest, "--encoder", "standin", "--index", idx_dir, "--batch-size", "50"])
    out = tmp_path / "res.json"
    cli.main(["eval", "--manifest", manifest, "--encoder", "standin", "--index", idx_dir,
              "--relevance", "attrs", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["recall@1"] == 1.0 and res["gallery_size"] == 240
    with pytest.raises(SystemExit):   # mismatched gallery size is rejected, not silently scored
        cli.main(["eval", "--manifest", manifest, "--encoder", "standin", "--index", idx_dir, "--limit", "10"])


def test_siglip_import_is_lazy():
    """Constructing the real encoder needs torch; without it, failure must be an ImportError, not a crash at import."""
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            SigLIPEncoder()


def test_coco_converter(tmp_path):
    import json as _json
    from PIL import Image
    from search.data import coco_captions_to_manifest
    (tmp_path / "val2017").mkdir()
    for n in ("000000000001.jpg", "000000000002.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / "val2017" / n)
    coco = {"images": [{"id": 1, "file_name": "000000000001.jpg"}, {"id": 2, "file_name": "000000000002.jpg"},
                       {"id": 3, "file_name": "000000000003.jpg"}],                       # id 3: not on disk -> skipped
            "annotations": [{"image_id": 1, "caption": "a dog "}, {"image_id": 1, "caption": "a puppy"},
                            {"image_id": 2, "caption": "a cat"}, {"image_id": 3, "caption": "gone"}]}
    (tmp_path / "c.json").write_text(_json.dumps(coco))
    out = tmp_path / "m" / "manifest.jsonl"; out.parent.mkdir()
    assert coco_captions_to_manifest(str(tmp_path / "c.json"), str(tmp_path / "val2017"), str(out)) == 2
    rows = read_manifest(str(out))
    assert rows[0]["captions"] == ["a dog", "a puppy"] and os.path.normpath(rows[0]["image"]) == str(tmp_path / "val2017" / "000000000001.jpg")
    with pytest.raises(ValueError):
        coco_captions_to_manifest(str(tmp_path / "c.json"), str(tmp_path / "nowhere"), str(out))


def test_write_manifest_creates_missing_folders(tmp_path):
    from search.data import write_manifest
    out = tmp_path / "a" / "b" / "m.jsonl"
    write_manifest([{"image": "x.jpg", "captions": ["c"]}], str(out))
    assert out.exists()
