"""Streamlit front end. Talks to the FastAPI service (default http://localhost:8000).

    API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semantic image search", layout="wide")
st.title("Semantic image search")

try:
    info = httpx.get(f"{API_URL}/health", timeout=5).json()
    st.caption(f"{info['images']:,} images indexed · encoder: {info['encoder']}")
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach the API at {API_URL}: {e}")
    st.stop()

k = st.sidebar.slider("Results", 1, 30, 9)
tab_text, tab_img = st.tabs(["Search by text", "Search by image"])


def show(results):
    if not results:
        st.info("No results.")
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            try:
                st.image(r["path"], caption=f"{r.get('caption', '')} · {r['score']:.3f}")
            except Exception:  # noqa: BLE001  (path not visible to the UI host)
                st.write(f"{r['path']} · {r['score']:.3f}")


with tab_text:
    q = st.text_input("Describe what you are looking for", placeholder="a dog catching a frisbee")
    if q.strip():
        show(httpx.get(f"{API_URL}/search", params={"q": q, "k": k}, timeout=30).json()["results"])

with tab_img:
    f = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if f is not None:
        resp = httpx.post(f"{API_URL}/search/image", params={"k": k}, timeout=30,
                          files={"file": (f.name, f.getvalue(), f.type)})
        if resp.status_code == 200:
            show(resp.json()["results"])
        else:
            st.error(resp.text)
