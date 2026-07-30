import os, time
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# ---------- KONFIGURASI (samakan dengan build_embeddings.py) ----------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL  = "gemini-2.5-flash"   # pilih model Flash gratis di AI Studio
DIM = 768
TOP_K = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.npz")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ---------- MUAT INDEKS PRA-HITUNG (bukan meng-embed korpus di sini) ----------
@st.cache_resource(show_spinner="Memuat indeks pengetahuan...")
def load_index():
    if not os.path.exists(INDEX_PATH):
        return None, None, None
    d = np.load(INDEX_PATH, allow_pickle=True)
    return d["mat"], list(d["sources"]), list(d["texts"])


# ---------- RETRY sederhana untuk 429 ----------
def _retry(fn, tries=5, wait=10):
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Batas kuota API tercapai. Coba lagi beberapa saat.")


# ---------- EMBED PERTANYAAN (hanya 1 request per query) ----------
def embed_query(q):
    def call():
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=q,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=DIM
            ),
        )
        return np.array(r.embeddings[0].values, dtype="float32")
    v = _retry(call)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def retrieve(q, mat, sources, texts, k=TOP_K):
    v = embed_query(q)
    sims = mat @ v
    top = np.argsort(-sims)[:k]
    return [(sources[i], texts[i], float(sims[i])) for i in top]


# ---------- GENERATION ber-grounding + wajib sitasi ----------
def generate_answer(query, contexts):
    blok = "\n\n".join(
        f"[Sumber {n}] ({src})\n{txt}" for n, (src, txt, _) in enumerate(contexts, 1)
    )
    prompt = f"""Anda asisten pengetahuan tata kelola (GCG). Jawab HANYA berdasarkan konteks di bawah.
Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan pada dokumen yang tersedia."
Selalu cantumkan rujukan dalam bentuk [Sumber N] di akhir kalimat yang relevan. Jawab dalam Bahasa Indonesia.

KONTEKS:
{blok}

PERTANYAAN: {query}

JAWABAN:"""
    return _retry(lambda: client.models.generate_content(model=CHAT_MODEL, contents=prompt).text)


# ---------- UI ----------
st.title("Governance Knowledge Management System — Demo")
st.caption("Prototipe RAG. Jawaban selalu merujuk ke dokumen sumber. Hanya dokumen publik.")

mat, sources, texts = load_index()

if mat is None:
    st.warning(
        "File index.npz belum ada. Jalankan build_embeddings.py di Google Colab, "
        "lalu commit index.npz ke folder governance-kms-demo/ dan reboot aplikasi."
    )
    st.stop()

q = st.chat_input("Tanyakan sesuatu tentang GCG, gratifikasi, benturan kepentingan, dst.")
if q:
    st.chat_message("user").write(q)
    hits = retrieve(q, mat, sources, texts)
    answer = generate_answer(q, hits)
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Lihat sumber yang diambil"):
            for n, (src, txt, score) in enumerate(hits, 1):
                st.markdown(f"**[Sumber {n}]** {src}  ·  skor {score:.2f}")
                st.write(txt[:400] + ("..." if len(txt) > 400 else ""))
