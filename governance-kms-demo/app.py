import os, time
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# ---------- KONFIGURASI ----------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL  = "gemini-3.5-flash"   # model Flash GA saat ini (VERIFIKASI di AI Studio bila 404)
DIM = 768
TOP_K = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.npz")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ---------- MUAT INDEKS ----------
@st.cache_resource(show_spinner="Memuat indeks pengetahuan...")
def _load_index(path, mtime):
    d = np.load(path, allow_pickle=True)
    return d["mat"], list(d["sources"]), list(d["texts"])


def load_index():
    if not os.path.exists(INDEX_PATH):
        return None, None, None
    return _load_index(INDEX_PATH, os.path.getmtime(INDEX_PATH))


# ---------- RETRY untuk error sementara ----------
def _retry(fn, tries=5):
    transient = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503",
                 "500", "INTERNAL", "DEADLINE", "overloaded", "high demand")
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if any(t in str(e) for t in transient):
                time.sleep(3 * (i + 1))   # backoff bertambah: 3, 6, 9, 12, 15 dtk
                continue
            raise
    raise RuntimeError("BUSY")  # ditangani di UI dengan pesan ramah


# ---------- EMBED PERTANYAAN  ----------
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


# ---------- GENERATION ----------
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
        "File index.npz belum terbaca. Pastikan sudah di folder governance-kms-demo/ "
        "lalu lakukan Reboot app (bukan hanya Rerun)."
    )
    st.stop()

st.caption(f"Indeks siap: {len(texts)} potongan dokumen.")

q = st.chat_input("Tanyakan sesuatu tentang GCG, gratifikasi, benturan kepentingan, dst.")
if q:
    st.chat_message("user").write(q)
    try:
        hits = retrieve(q, mat, sources, texts)
        answer = generate_answer(q, hits)
    except RuntimeError:
        st.chat_message("assistant").warning(
            "Model sedang sibuk (server Google ramai) atau kuota sesaat penuh. "
            "Coba kirim pertanyaan yang sama lagi beberapa detik lagi."
        )
        st.stop()
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Lihat sumber yang diambil"):
            for n, (src, txt, score) in enumerate(hits, 1):
                st.markdown(f"**[Sumber {n}]** {src}  ·  skor {score:.2f}")
                st.write(txt[:400] + ("..." if len(txt) > 400 else ""))
