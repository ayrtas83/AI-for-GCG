import os, time
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# ---------- KONFIGURASI (samakan dengan build_embeddings.py) ----------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL  = "gemini-3.5-flash"   # model Flash GA saat ini (VERIFIKASI di AI Studio bila 404)
DIM = 768
TOP_K = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.npz")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ---------- MUAT INDEKS (cek keberadaan di luar cache; cache di-key mtime) ----------
@st.cache_resource(show_spinner="Memuat indeks pengetahuan...")
def _load_index(path, mtime):
    d = np.load(path, allow_pickle=True)
    return d["mat"], list(d["sources"]), list(d["texts"])


def load_index():
    if not os.path.exists(INDEX_PATH):
        return None, None, None
    return _load_index(INDEX_PATH, os.path.getmtime(INDEX_PATH))


# ---------- RETRY untuk error sementara (429 kuota + 503/500 server sibuk) ----------
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
    raise RuntimeError("BUSY")


# ---------- 1) EMBED PERTANYAAN ----------
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


# ---------- 2) SIMILARITY SEARCH -> TOP-K ----------
def search(qvec, mat, sources, texts, k=TOP_K):
    sims = mat @ qvec
    top = np.argsort(-sims)[:k]
    return [(sources[i], texts[i], float(sims[i])) for i in top]


# ---------- 3) SUSUN PROMPT (di-grounding + wajib sitasi) ----------
def build_prompt(query, contexts):
    blok = "\n\n".join(
        f"[Sumber {n}] ({src})\n{txt}" for n, (src, txt, _) in enumerate(contexts, 1)
    )
    return f"""Anda asisten pengetahuan tata kelola (GCG). Jawab HANYA berdasarkan konteks di bawah.
Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan pada dokumen yang tersedia."
Selalu cantumkan rujukan dalam bentuk [Sumber N] di akhir kalimat yang relevan. Jawab dalam Bahasa Indonesia.

KONTEKS:
{blok}

PERTANYAAN: {query}

JAWABAN:"""


# ---------- 4) GENERATION ----------
def generate_answer(prompt):
    return _retry(lambda: client.models.generate_content(model=CHAT_MODEL, contents=prompt).text)


# ---------- UI ----------
st.title("Governance Knowledge Management System — Demo")
st.caption("Prototipe RAG. Jawaban selalu merujuk ke dokumen sumber.")

mat, sources, texts = load_index()
if mat is None:
    st.warning(
        "File index.npz belum terbaca. Pastikan sudah di folder governance-kms-demo/ "
        "lalu lakukan Reboot app (bukan hanya Rerun)."
    )
    st.stop()

st.caption(f"Indeks siap: {len(texts)} potongan dokumen.")
show_trace = st.sidebar.checkbox("Tampilkan proses RAG (mode demonstrasi)", value=True)

q = st.chat_input("Tanyakan sesuatu tentang GCG, gratifikasi, benturan kepentingan, dst.")
if q:
    st.chat_message("user").write(q)
    try:
        qvec = embed_query(q)                          # 1. embedding query
        hits = search(qvec, mat, sources, texts)       # 2. similarity search -> top-k
        prompt = build_prompt(q, hits)                 # 3. prompt
        answer = generate_answer(prompt)               # 4. generation
    except RuntimeError:
        st.chat_message("assistant").warning(
            "Model sedang sibuk (server Google ramai) atau kuota sesaat penuh. "
            "Coba kirim pertanyaan yang sama lagi beberapa detik lagi."
        )
        st.stop()

    with st.chat_message("assistant"):
        st.write(answer)

        with st.expander("Sumber rujukan"):
            for n, (src, txt, score) in enumerate(hits, 1):
                st.markdown(f"**[Sumber {n}]** {src}  ·  skor {score:.3f}")
                st.write(txt[:400] + ("..." if len(txt) > 400 else ""))

        if show_trace:
            with st.expander("Proses RAG (untuk pembahasan skripsi)", expanded=False):
                st.markdown("**1. Pertanyaan**")
                st.code(q, language=None)

                st.markdown(f"**2. Embedding query** — vektor berdimensi {DIM} (5 nilai pertama):")
                st.code(np.round(qvec[:5], 4).tolist(), language=None)

                st.markdown(f"**3. Similarity search — Top-{TOP_K} potongan (skor cosine)**")
                for n, (src, txt, score) in enumerate(hits, 1):
                    st.markdown(f"- [Sumber {n}] `{src}` · skor **{score:.3f}**")

                st.markdown("**4. Prompt yang dikirim ke Gemini**")
                st.code(prompt, language=None)

                st.markdown("**5. Jawaban**")
                st.write(answer)
